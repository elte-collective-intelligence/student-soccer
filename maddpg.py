import os
import torch

# Tensordict / TorchRL
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import RewardSum, TransformedEnv,EnvBase
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal,TanhDelta,AdditiveGaussianModule
from torchrl.objectives import DDPGLoss,ValueEstimators, SoftUpdate
from torchrl.envs.transforms import VecNorm
from dataclasses import dataclass
from torch.distributions import Categorical


from matplotlib import pyplot as plt
from tqdm import tqdm
import logging

logger = logging.getLogger('algorithm')


from base_algorithm import BaseAlgorithm,MLPConfig



class MADDPG(BaseAlgorithm):
    def __init__(
        self,
        # Optimisation
        learning_rate: float = 3e-4,       # override if supplied,
        gamma: float = 0.99,
        max_grad_norm: float = 1.0,
        polyak_tau: float = 0.995,
        memory_size: int=1_000_000,
        #Exploration
        sigma_init: float = 0.9,
        sigma_end: float = 0.1,
        annealing_steps_ratio: float = 0.5,
        
        #Loss
        discrepancy_loss:str = "l2",
        use_critic_target_network = True,
        use_policy_target_network = False,

        # Training loop
        total_frames: int = 60_000,
        frames_per_batch: int = 6_000,
        minibatch_size: int = 400,
        num_epochs: int = 50,

        independent_critic: bool = False,
        share_parameters: bool = True,
        continuous_actions: bool = True, 
        
        base_state_action_value_key: str = "state_action_value",
        base_actor_logits_keys: str = "param",
        base_observation_action_key: str = "observation_n_action",
        actor_net_cfg: MLPConfig | dict = MLPConfig(),
        critic_net_cfg: MLPConfig | dict = MLPConfig(),


        **kwargs
    ):
        #       # Network parameters
        self.share_parameters = share_parameters
        self.independent_critic = independent_critic
        if not continuous_actions:
            logger.info("Discrete action space is not supported in MADDPG.")
            raise NotImplementedError("Discrete action space is not supported in MADDPG.")
        super().__init__(**kwargs,continuous_actions=continuous_actions)


        # Hyper-parameters
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.polyak_tau = polyak_tau
        self.memory_size = memory_size
        self.learning_rate = learning_rate
        
        self.sigma_init = sigma_init
        self.sigma_end = sigma_end
        self.annealing_steps_ratio = annealing_steps_ratio

        self.discrepancy_loss = discrepancy_loss
        self.use_critic_target_network = use_critic_target_network
        self.use_policy_target_network = use_policy_target_network
        # Training parameters
        self.total_frames = total_frames
        self.frames_per_batch = frames_per_batch
        self.minibatch_size = minibatch_size
        self.num_epochs = num_epochs
        
        
        #Key names
        self.base_state_action_value_key = base_state_action_value_key
        self.base_actor_logits_key = base_actor_logits_keys
        self.base_observation_action_key = base_observation_action_key
        
        if isinstance(actor_net_cfg,dict):
            actor_net_cfg = MLPConfig(**actor_net_cfg)
        self.actor_net_cfg = actor_net_cfg

        if isinstance(critic_net_cfg,dict):
            critic_net_cfg = MLPConfig(**critic_net_cfg)
        self.critic_net_cfg = critic_net_cfg
        self.metric_dashboard = None
    
    @property
    def __name__(self):
        return self._prefix + "DDPG"
        
    @property
    def policy(self):
        return self._policy
    

    def _setup_additional_keys(self,env: EnvBase):
        self.state_action_value_key = self.get_group_key(self.base_state_action_value_key)
        self.logits_key = self.get_group_key(self.base_actor_logits_key)
        self.observation_action_key = self.get_group_key(self.base_observation_action_key)

    def _setup_algorithm(self, env: EnvBase):
        self._setup_policy(env)
        self._setup_exploration_policy(env)
        self._setup_critic()
     

    def setup(self,make_env_fn,**kwargs):
        self._create_folder_structure()
        self.make_env_fn = make_env_fn
        dummy_env = self.make_env_fn(num_envs=1)
        self._setup_base_keys(dummy_env)
        self._setup_additional_keys(dummy_env)
        self._setup_algorithm(dummy_env,**kwargs)
        self._setup_loss()
        self._setup_target_updaters()
        self._setup_replay_buffer()
        self._setup_optimizer()
        self._setup_collector(self.make_env_fn)
        
        self._pre_train_check()

    def _setup_optimizer(self):
        
        self.optimizer = torch.optim.Adam(
                self.loss_module.parameters(),
                lr=self.learning_rate
            )

    def _setup_policy(self, env: EnvBase):
    
        net_module = self._setup_continuous_actor_net_module(env=env)
    
    

        self._policy = ProbabilisticActor(
            module=net_module,
            spec=env.full_action_spec_unbatched[self.action_key],
            in_keys=[self.logits_key],
            out_keys=[self.action_key],
            distribution_class=TanhDelta,
            distribution_kwargs={
                "low": env.full_action_spec_unbatched[self.action_key].space.low,
                "high": env.full_action_spec_unbatched[self.action_key].space.high,
        },
        return_log_prob=False,
        )
        
    def _setup_exploration_policy(self, env: EnvBase):
        self._exploration_policy = TensorDictSequential(
            self._policy,
            AdditiveGaussianModule(
                spec=self.policy.spec,
                annealing_num_steps=int(self.total_frames * self.annealing_steps_ratio),  # Number of frames after which sigma is sigma_end
                action_key=self.action_key,
                sigma_init=self.sigma_init,  # Initial value of the sigma
                sigma_end=self.sigma_end,  # Final value of the sigma
            )
        )

    def _setup_continuous_actor_net_module(self,env: EnvBase):
        self._actor_net = MultiAgentMLP(
                n_agent_inputs=self.n_observations,
                n_agent_outputs=self.n_actions,
                n_agents=self.n_agents,
                centralised=False,
                share_params=self.share_parameters,
                device=self.device,
                depth=self.actor_net_cfg.depth,
                num_cells=self.actor_net_cfg.num_cells,
                activation_class=self.actor_net_cfg.activation_class
            )
        net_module = TensorDictModule(
            self._actor_net,
            in_keys=[self.observation_key],
            out_keys=[self.logits_key],
        )
        
        return net_module


    def _setup_critic(self):
        
        self._critic_net = MultiAgentMLP(
            n_agent_inputs=self.n_observations+self.n_actions,
            n_agent_outputs=1,
            n_agents=self.n_agents,
            centralised=not self.independent_critic,
            share_params=self.share_parameters,
            device=self.device,
            depth=self.critic_net_cfg.depth,
            num_cells=self.critic_net_cfg.num_cells,
            activation_class=self.critic_net_cfg.activation_class,
        )
        self._critic_net_module = TensorDictModule(
            module=self._critic_net,
            in_keys=[self.observation_action_key],
            out_keys=[self.state_action_value_key],
        )
        
        self._concat_module = TensorDictModule(
            lambda obs, action: torch.cat([obs, action], dim=-1),
            in_keys=[self.observation_key, self.action_key],
            out_keys=[self.observation_action_key],
        )
        self.critic = TensorDictSequential(
            self._concat_module,
            self._critic_net_module,
        )




    def _setup_replay_buffer(self):
        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(self.memory_size, device=self.device),
            sampler=SamplerWithoutReplacement(),
            batch_size=self.minibatch_size,
        )

    def _setup_loss(self):
        self.loss_module = DDPGLoss(
            actor_network=self.policy,
            value_network=self.critic,
            delay_actor=self.use_policy_target_network,
            delay_value=self.use_critic_target_network,
            loss_function=self.discrepancy_loss
        )
        self.loss_module.set_keys(
            state_action_value=self.state_action_value_key,        # (group, "state_action_value")
            reward=self.reward_key,                                # (group, "reward")
            done=self.get_group_key(self.base_done_key),           # (group, "done")
            terminated=self.get_group_key(self.base_terminated_key) # (group, "terminated")
        )
        self.loss_module.make_value_estimator(
            ValueEstimators.TD0, gamma=self.gamma,
        )

    def _setup_target_updaters(self):
        self.target_updater = SoftUpdate(self.loss_module, tau=self.polyak_tau)
    
    def _setup_optimizer(self):
        self.actor_optimizer = torch.optim.Adam(
            self.loss_module.actor_network_params.flatten_keys().values(), lr=self.learning_rate
        )
        self.critic_optimizer =  torch.optim.Adam(
            self.loss_module.value_network_params.flatten_keys().values(), lr=self.learning_rate
        )

    def _setup_collector(self, make_env: callable):
        self.collector = SyncDataCollector(
            make_env,
            self._exploration_policy,
            device=self.device,
            storing_device=self.device,
            policy_device=self.device,
            env_device=self.device,
            frames_per_batch=self.frames_per_batch,
            total_frames=self.total_frames,
        )

    def save(self, folder: str, filename: str):
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
        }, f"{folder}/{filename}")

    def load(self, path: str | None = None):
        """
        Load the actor and critic state dictionaries from a file.
        
        Args:
            path (str | None): The file path to load the state dictionaries from.
                               If None, uses the default path set during initialization.
        """            
        checkpoint = torch.load(path,map_location=torch.device(self.device))
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])

    def process_batch(self, batch_td):
        """
        Match the tutorial's semantics:
        - Create (group, "done") and (group, "terminated") from global keys
        - Ensure reward has a trailing singleton dim
        - Then flatten leading batch dims
        """
        group = self.group

        # 1) Expand global done/terminated into per-agent group tensors
        keys = list(batch_td.keys(True, True))
        group_shape = batch_td.get_item_shape(group)  # (..., n_agents_in_group)

        nested_done_key = ("next", group, "done")
        nested_terminated_key = ("next", group, "terminated")

        if nested_done_key not in keys:
            batch_td.set(
                nested_done_key,
                batch_td.get(("next", "done"))
                .unsqueeze(-1)
                .expand((*group_shape, 1)),
            )

        if nested_terminated_key not in keys:
            batch_td.set(
                nested_terminated_key,
                batch_td.get(("next", "terminated"))
                .unsqueeze(-1)
                .expand((*group_shape, 1)),
            )

        # 2) Ensure reward is (..., n_agents_in_group, 1)
        nested_reward_key = ("next", group, "reward")
        reward = batch_td.get(nested_reward_key)
        if reward.dim() == len(group_shape):  # e.g. (..., n_agents_in_group)
            reward = reward.unsqueeze(-1)
            batch_td.set(nested_reward_key, reward)

        # 3) Now flatten leading batch dims, like you already do
        flat_td = batch_td.reshape(-1)
        return flat_td


    def _optimize_dual(self,loss_dict,mini_td):
        actor_loss_value = (
            loss_dict["loss_objective"] + loss_dict["loss_entropy"]
        )
        critic_loss_value = loss_dict["loss_critic"]
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss_value.backward(retain_graph=True)
        actor_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.max_grad_norm
        )
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss_value.backward()
        critic_norm = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.max_grad_norm
        )
        self.critic_optimizer.step()
        self._log_optimization_metrics(mini_td,loss_dict=loss_dict, actor_loss=actor_loss_value, critic_loss=critic_loss_value, actor_norm=actor_norm, critic_norm=critic_norm)

    def _optimize_single(self,loss_dict,mini_td):
            loss_value = (
                loss_dict["loss_objective"]
                + loss_dict["loss_critic"]
                + loss_dict["loss_entropy"]
            )

            loss_value.backward()

            total_norm = torch.nn.utils.clip_grad_norm_(
                self.loss_module.parameters(), self.max_grad_norm
            )  

            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._log_optimization_metrics(mini_td,loss_dict=loss_dict, loss_value=loss_value, total_norm=total_norm)
          
    def optimization_step(self,num_frames,*args, **kwargs):
        self._current_epochs_number = 0
        for _ in range(self.num_epochs):
            self._current_epochs_number += 1
            self._current_minibatch_number = 0
            for _ in range(self.frames_per_batch // self.minibatch_size):
                self._current_minibatch_number += 1
                mini_td = self.replay_buffer.sample()
                loss_dict = self.loss_module(mini_td)    
                actor_loss = loss_dict.get("loss_actor")
                actor_loss.backward()
                params = self.actor_optimizer.param_groups[0]["params"]
                policy_norm = torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
                self.actor_optimizer.step()
                self.actor_optimizer.zero_grad()
                
                critic_loss = loss_dict.get("loss_value")
                critic_loss.backward()
                params = self.critic_optimizer.param_groups[0]["params"]
                critic_norm = torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
                self.critic_optimizer.step()
                self.critic_optimizer.zero_grad()
                
                # self._log_optimization_metrics(mini_td,loss_dict=loss_dict, policy_loss=actor_loss, critic_loss=critic_loss, policy_norm=policy_norm, critic_norm=critic_norm)
            self.target_updater.step()
                
                
            if self.metric_dashboard is not None:
                self.metric_dashboard.update(self._optimization_metrics)
        self._exploration_policy[-1].step(num_frames)

    def _log_optimization_metrics(self,mini_td,loss_dict,**kwargs):
        metrics = {
            "batch_num": self._current_batch_number,
            "epoch_num": self._current_epochs_number,
            "minibatch_num": self._current_minibatch_number,
            "timesteps": self._current_timesteps_taken,
        }
        
        td_error = mini_td.get("td_error")
        if td_error is not None:
            metrics["td_error"] = td_error.detach().abs().mean().item()
        
        for k,v in loss_dict.items():
            metrics[k] = v.cpu().item()

        for k,v in kwargs.items():
            metrics[k] = v.cpu().item()
        
       
        self._optimization_metrics.append(metrics)


 