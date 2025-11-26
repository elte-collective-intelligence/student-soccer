import json
import os
import time 
from typing import Union
import torch
from tqdm import tqdm
from tensordict import TensorDict
from collections import deque
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.data.tensor_specs import BoxList, CategoricalBox
from dataclasses import dataclass
import logging
logger = logging.getLogger('algorithm')



ACTIVATION_CLASS_MAP = {
    "relu": torch.nn.ReLU,
    "tanh": torch.nn.Tanh,
    "elu": torch.nn.ELU,
    "leaky_relu": torch.nn.LeakyReLU,
    "silu": torch.nn.SiLU,
}

DEFAULT_DEPTH = 2
DEFAULT_NUM_CELLS = 128

@dataclass
class MLPConfig:
    model_class: str = "mlp"
    depth: int = DEFAULT_DEPTH
    num_cells: int = DEFAULT_NUM_CELLS
    activation_class: str = "tanh"

    def __post_init__(self):
        # If user passed a string, convert it to the actual class
        if isinstance(self.activation_class, str):
            if self.activation_class not in ACTIVATION_CLASS_MAP:
                raise ValueError(
                    f"Unknown activation '{self.activation_class}'. "
                    f"Must be one of {list(ACTIVATION_CLASS_MAP.keys())}"
                )
            self.activation_class = ACTIVATION_CLASS_MAP[self.activation_class]

    

class BaseAlgorithm:
    """
    Base class for all MARL algorithms.
    """

    def __init__(self, 
                 n_agents: int = None,
                 device: str = "cpu", 
                 seed: int = 42,
                 continuous_actions: bool = False,
                 
                 root_folder: str = None, 
                 display_moving_average_window_size: int = 100,
                 evaluation_interval: int = 1_000_000, 
                 evaluation_episodes: int = 10,
                 evaluation_max_steps: int = 100,
                 team_reward_aggregation_fn: Union[str,callable]= "mean",

                 checkpoint_interval: int = 1_000_000,

                 group: str =  None, 
                 action_key: tuple = 'action',
                 observation_key: tuple = 'observation',
                 reward_key: tuple = 'reward',
                 done_key: tuple = 'done',
                 terminated_key: tuple = 'terminated',
                 truncated_key: tuple = 'truncated',
                 episode_reward_key: tuple = 'episode_reward',
                 do_absolute_evaluation: bool = True,
                 
                 metric_dashboard: bool = False,
                 ):
        
        self.n_agents = n_agents
        self.continuous_actions = continuous_actions
         
        self.base_action_key = action_key
        self.base_observation_key = observation_key
        self.base_reward_key = reward_key
        self.base_done_key = done_key
        self.base_terminated_key = terminated_key
        self.base_truncated_key = truncated_key
        self.base_episode_reward_key = episode_reward_key
        self.group = group
        self.device = device
        self.seed = seed
        self.root_folder = root_folder or f"{self.__name__}/"
        self.checkpoints_folder = os.path.join(self.root_folder, "checkpoints")
        self.checkpoint_interval = checkpoint_interval
        self.evaluation_interval = evaluation_interval
        self.evaluation_episodes = evaluation_episodes
        self.evaluation_max_steps = evaluation_max_steps
        self.do_absolute_evaluation = do_absolute_evaluation
        self.metric_dashboard = None

        if isinstance(team_reward_aggregation_fn, str):
            self.team_reward_aggregation_fn = getattr(torch, team_reward_aggregation_fn)
        elif callable(team_reward_aggregation_fn):
            self.team_reward_aggregation_fn = team_reward_aggregation_fn
        else:
            raise NotImplementedError("Passed team_reward_aggregation_fn is neither a method of torch nor a callable.")

        self._agent_dim = -2
        
        self._losses = deque()
        
        self._episode_rewards = deque()
        self._recent_episode_rewards = deque(maxlen=display_moving_average_window_size)
        self._display_moving_average_window_size = display_moving_average_window_size
        
        # Logging and bookkeeping
        self._eval_results = {}
        self._eval_steps = 0
        self._optimization_metrics = []
        self._current_timesteps_taken = 0
        self._current_batch_number = 0      # relative
        self._current_minibatch_number = 0  # relative
        self._current_epochs_number = 0   # relative

    @property
    def _prefix(self):
        return f"{'Ho' if self.share_parameters else 'He'}{'I' if self.independent_critic else 'MA'}"
    
    
    def get_group_key(self, key: str) -> tuple:
        if self.group is None:
            return key
        return (self.group, key)
    
    def get_next_key(self, key: str| tuple) -> tuple:
        if isinstance(key, tuple):
            return ("next",) + key
        return ("next", key)
    
    def _setup_algorithm(self,):
        raise NotImplementedError("Setup method must be implemented by subclasses.")

    def _create_folder_structure(self):
        os.makedirs(self.root_folder, exist_ok=True)
        os.makedirs(self.checkpoints_folder, exist_ok=True)

    def setup(self,make_env_fn,**kwargs):
        self._create_folder_structure()
        self.make_env_fn = make_env_fn
        dummy_env = self.make_env_fn(num_envs=1)
        self._setup_base_keys(dummy_env)
        self._setup_additional_keys(dummy_env)
        self._setup_algorithm(dummy_env,**kwargs)
        self._setup_loss()
        self._setup_replay_buffer()
        self._setup_optimizer()
        self._setup_collector(self.make_env_fn)
        
        self._pre_train_check()
    

    def _setup_base_keys(self,env):
        if self.n_agents is None:
            if hasattr(env, 'group_map'):
                self.n_agents = len(env.group_map[self.group])
            else:
                raise ValueError("Number of agents (n_agents) must be specified if the environment does not have a group_map attribute.")

        self.action_key = self.get_group_key(self.base_action_key)
        self.observation_key = self.get_group_key(self.base_observation_key)
        self.reward_key = self.get_group_key(self.base_reward_key)
       
        self.done_key = self.base_done_key                  # self.done_key = self.get_group_key(self.base_done_key)
        self.terminated_key = self.base_terminated_key      # self.terminated_key = self.get_group_key(self.base_terminated_key)
        self.truncated_key = self.base_truncated_key        # self.truncated_key = self.get_group_key(self.base_truncated_key)

        self.episode_reward_key = self.get_next_key(self.get_group_key(self.base_episode_reward_key))

        self.n_observations: int = env.full_observation_spec_unbatched[
            self.observation_key
        ].shape[-1]

        
        if self.continuous_actions:
            self.n_actions: int = get_continuous_action_space_size(env,self.action_key)
        else:
            self.n_actions: int = get_discrete_action_space_size(env,self.action_key)

            
    @property
    def policy(self):
        raise NotImplementedError("Policy property must be implemented by subclasses.")
    
    def setup_policy_for_evaluation(self,make_env_fn,**kwargs):
        dummy_env = make_env_fn(num_envs=1)
        self._setup_base_keys(dummy_env)
        self._setup_additional_keys(dummy_env)
        self._setup_algorithm(dummy_env,**kwargs)
    
    def process_batch(self, batch_td: TensorDict):
        raise NotImplementedError("Process batch method must be implemented by subclasses.")
    
    def optimization_step(self):
        raise NotImplementedError("Optimization step method must be implemented by subclasses.")
    
    def _setup_additional_keys(self):
        pass

    def _setup_collector(self,env):
        self.collector = None
        raise NotImplementedError("Setup collector method must be implemented by subclasses.")

    def _setup_loss(self):
        self.loss_module = None
        raise NotImplementedError("Setup loss method must be implemented by subclasses.")

    def _setup_replay_buffer(self):
        self.replay_buffer = None
        raise NotImplementedError("Setup replay buffer method must be implemented by subclasses.")
    
    def _check_replay_buffer_ready(self):
        if self.replay_buffer is None:
            raise ValueError("Replay buffer is not set up. Please call setup_replay_buffer() before training.")

    def _setup_optimizer(self):
        raise NotImplementedError("Setup optimizer method must be implemented by subclasses.")

    def _pre_train_check(self):
        if self.collector is None:
            raise ValueError("Collector is not set up. Please call setup_collector() before training.")
        self._check_replay_buffer_ready()
        
        if self.loss_module is None:
            raise ValueError("Loss function is not set up. Please call setup_loss() before training.")
        if not hasattr(self, 'policy'):
            raise ValueError("Policy is not set up. Please define self.policy before training.")

    def __collection_time_check(self):

        total_frames = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(self.collector):
            num_frames = batch.numel()  # or batch["next"]["done"].numel() depending on batch structure
            total_frames += num_frames

            elapsed = time.time() - start_time
            fps = total_frames / elapsed
            print(f"Batch {batch_idx}: {num_frames} frames, total {total_frames}, FPS={fps:.2f}")
    
    def train(self,*args, **kwargs):
        self._episode_reward_mean_list = []
        self._pbar = tqdm(total=self.total_frames, desc=f"'{self.group}' group episode mean reward = 0.0")
        episode_returns, episode_lengths = self.evaluate(make_env_fn=self.make_env_fn, absolute_evaluation=False,max_steps=self.evaluation_max_steps,num_envs=self.evaluation_episodes)
        self._eval_results[f"step_{self._eval_steps}"] = {
            "step_count":self._current_timesteps_taken,
            "episode_length":episode_lengths.tolist(),
            "episode_return":episode_returns.tolist()
        }
        self._eval_steps += 1
        logger.debug(f"Evaluation at {self._current_timesteps_taken} timesteps: Mean Return = {episode_returns.mean():.2f}")
        logger.debug(f"Evaluation at {self._current_timesteps_taken} timesteps: Mean Length = {episode_lengths.mean():.2f}")
        for batch_td in self.collector:
            self._current_batch_number += 1
            processed_batch_td = self.process_batch(batch_td)
            if processed_batch_td.dim() >= 2:
                processed_batch_td = processed_batch_td.flatten()
            self._current_timesteps_taken += batch_td.numel()
            self.replay_buffer.extend(processed_batch_td)
            self.optimization_step(num_frames=processed_batch_td.numel())
            self.collector.update_policy_weights_()
            self._train_step(processed_batch_td)
            
            if self._current_timesteps_taken % self.evaluation_interval == 0:
                episode_returns, episode_lengths = self.evaluate(make_env_fn=self.make_env_fn, absolute_evaluation=False,max_steps=self.evaluation_max_steps,num_envs=self.evaluation_episodes)
                self._eval_results[f"step_{self._eval_steps}"] = {
                    "step_count":self._current_timesteps_taken,
                    "episode_length":episode_lengths.tolist(),
                    "episode_return":episode_returns.tolist()
                }
                self._eval_steps += 1
                logger.debug(f"Evaluation at {self._current_timesteps_taken} timesteps: Mean Return = {episode_returns.mean():.2f}")
                logger.debug(f"Evaluation at {self._current_timesteps_taken} timesteps: Mean Length = {episode_lengths.mean():.2f}")
                
            if self._current_timesteps_taken % self.checkpoint_interval == 0:
                self.save(self.checkpoints_folder, f"checkpoint_{self._current_timesteps_taken}.pth")
                
        if self.do_absolute_evaluation:
            episode_returns, episode_lengths = self.evaluate(make_env_fn=self.make_env_fn, absolute_evaluation=True,max_steps=self.evaluation_max_steps,num_envs=self.evaluation_episodes)
            self._eval_results['absolute_metrics'] = {
                    "episode_length":episode_lengths.tolist(),
                    "episode_return":episode_returns.tolist()
                }
            logger.debug(f"Evaluation at {self._current_timesteps_taken} timesteps: Mean Return = {episode_returns.mean():.2f}")
            logger.debug(f"Evaluation at {self._current_timesteps_taken} timesteps: Mean Length = {episode_lengths.mean():.2f}")

        self.save(self.root_folder, f"last.pth")
        return self._eval_results


    def _train_step(self, batch_td: TensorDict):
        # self.__collection_time_check()
        num_frames = batch_td.numel()
        self._current_timesteps_taken += num_frames
       
        done_mask = batch_td.get(self.get_next_key(self.done_key))
        
        logger.debug(f'Shape of done_mask: {done_mask.shape}')
       
        if done_mask.dim() == 2:
            shaped_done = done_mask[:,0]
        else:
            shaped_done= done_mask.squeeze(-1)[:,0]
        logger.debug(f'Shape of shaped_done: {shaped_done.shape}')
        logger.debug(f'Shape of episode_rewards: {batch_td.get(self.episode_reward_key).shape}')
        
        episode_rewards = self.team_reward_aggregation_fn(batch_td.get(self.episode_reward_key).squeeze(-1), dim=1)
        logger.debug(f'Shape of episode_rewards aggregated: {episode_rewards.shape}')
        finished_rewards = episode_rewards[shaped_done]
        
        if finished_rewards.numel() > 0:
            self._episode_rewards.extend(finished_rewards.tolist())
            self._recent_episode_rewards.extend(finished_rewards.tolist())
        if len(self._episode_rewards) > 0:
            moving_avg = sum(self._recent_episode_rewards) / len(self._recent_episode_rewards)
        else:
            moving_avg = 0.0
        self._pbar.set_description(
            f"'{self.group}' group episode mean) = {moving_avg:.2f}",
            refresh=False
        )
        self._pbar.update(num_frames)

    def evaluate(self, make_env_fn, absolute_evaluation=False,num_envs=1,max_steps=None, callback=None):
        if max_steps is None:
            max_steps = self.evaluation_max_steps
        episode_multiplier = 10 if absolute_evaluation else 1
        env = make_env_fn(num_envs=num_envs * episode_multiplier,device = self.device)
        with torch.no_grad():
            rollouts= env.rollout(
                policy=self.policy,
                auto_cast_to_device=True,
                break_when_any_done=False,
                max_steps=max_steps,
                callback=callback
                )
            
            
        first_done_mask = self._get_first_done_indices(rollouts)
        
        episode_rewards = self.team_reward_aggregation_fn(rollouts.get(self.episode_reward_key), dim=-1)

        row_idx = torch.arange(episode_rewards.size(0))
        rewards = episode_rewards[row_idx, first_done_mask.squeeze(-1)]
        return rewards, first_done_mask.squeeze(-1).float()
    
    

        
    def _get_first_done_indices(self, rollouts):
        done_mask = rollouts[self.get_next_key(self.done_key)]
        first_done_mask=done_mask.int().argmax(dim=1)
        return first_done_mask

    def save(self, folder: str, filename: str):
        """
        Save the algorithm's state to a file.
        
        Args:
            path (str): The file path where the state should be saved.
        """
        raise NotImplementedError("Save method must be implemented by subclasses.")
    
    def load(self, path: str):
        """
        Load the algorithm's state from a file.
        
        Args:
            path (str): The file path from which the state should be loaded.
        """
        raise NotImplementedError("Load method must be implemented by subclasses.")
    
    
def get_discrete_action_space_size(env,action_key):
    action_space = env.full_action_spec_unbatched[action_key].space
    if isinstance(action_space, BoxList):
        return action_space.boxes[0].n
    elif isinstance(action_space, CategoricalBox):
        return action_space.n

def get_continuous_action_space_size(env,action_key):
    action_space = env.full_action_spec_unbatched[action_key].shape
    return action_space[-1]
    
        
        