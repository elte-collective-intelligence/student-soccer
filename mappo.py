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
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from torchrl.envs.transforms import VecNorm
from dataclasses import dataclass
from torch.distributions import Categorical


from matplotlib import pyplot as plt
from tqdm import tqdm
import logging

logger = logging.getLogger('algorithm')


from base_algorithm import BaseAlgorithm,MLPConfig


LOG_EXPLAINED_VARIANCE = False

class MAPPO(BaseAlgorithm):
    def __init__(
        self,
        # Optimisation
        actor_lr: float = 3e-4,       # override if supplied,
        critic_lr: float = 1e-4,      # override if supplied,
        learning_rate: float | None = None,   
        gamma: float = 0.99,
        lmbda: float = 0.9,
        entropy_eps: float = 1e-4,
        max_grad_norm: float = 1.0,
        clip_epsilon: float = 0.2,
        normalize_advantage: bool = False,

        # Training loop
        total_frames: int = 60_000,
        frames_per_batch: int = 6_000,
        minibatch_size: int = 400,
        num_epochs: int = 50,

        independent_critic: bool = False,
        share_parameters: bool = True,
        continuous_actions: bool = False, # NOT SUPPORTED YET
        base_state_value_key: str = "state_value",
        base_actor_logits_keys: str = None,
        actor_net_cfg: MLPConfig | dict = MLPConfig(),
        critic_net_cfg: MLPConfig | dict = MLPConfig(),


        **kwargs
    ):
        #       # Network parameters
        self.share_parameters = share_parameters
        self.independent_critic = independent_critic
        super().__init__(**kwargs,continuous_actions=continuous_actions)


        # Hyper-parameters
        self.gamma = gamma
        self.lmbda = lmbda
        self.entropy_eps = entropy_eps
        self.max_grad_norm = max_grad_norm
        self.clip_epsilon = clip_epsilon
        self.normalize_advantage = normalize_advantage

        # LR handling
        self.actor_lr = actor_lr 
        self.critic_lr = critic_lr
        if learning_rate is not None:
            logger.info(f"When `learning_rate` is supplied only one optimizer is used for both actor and critic.")
            self.actor_lr = None
            self.critic_lr = None
            self.learning_rate = learning_rate

        # Training parameters
        self.total_frames = total_frames
        self.frames_per_batch = frames_per_batch
        self.minibatch_size = minibatch_size
        self.num_epochs = num_epochs
        
        self.continuous_actions = continuous_actions
        if self.continuous_actions:
            if base_actor_logits_keys is None:
                self.base_actor_loc_key,self.base_actor_scale_key = "loc","scale"
            elif len(base_actor_logits_keys) == 2:
                self.base_actor_loc_key,self.base_actor_scale_key = base_actor_logits_keys
            else:
                raise ValueError(f"When using continuous actions, `base_actor_logits_keys` must be a list of two strings or tuples of strings, the first for the mean and the second for the scale. Got {base_actor_logits_keys}")
        else:
            if base_actor_logits_keys is None:
                self.base_actor_logits_key = "logits"
            elif isinstance(base_actor_logits_keys,str) or isinstance(base_actor_logits_keys,tuple):
                self.base_actor_logits_key = base_actor_logits_keys
            else:
                raise ValueError(f"When using discrete actions, `base_actor_logits_keys` must be a string or a tuple of strings. Got {base_actor_logits_keys}")
            self.base_actor_logits_keys = base_actor_logits_keys
                   
        self.base_state_value_key = base_state_value_key
        
        if isinstance(actor_net_cfg,dict):
            actor_net_cfg = MLPConfig(**actor_net_cfg)
        self.actor_net_cfg = actor_net_cfg

        if isinstance(critic_net_cfg,dict):
            critic_net_cfg = MLPConfig(**critic_net_cfg)
        self.critic_net_cfg = critic_net_cfg
        self.metric_dashboard = None
    
    @property
    def __name__(self):
        return self._prefix + "PPO"
        
    @property
    def policy(self):
        return self._actor
    

    def _setup_additional_keys(self,env: EnvBase):
        self.state_value_key = self.get_group_key(self.base_state_value_key)
        if self.continuous_actions:
            self.loc_key = self.get_group_key(self.base_actor_loc_key)
            self.scale_key = self.get_group_key(self.base_actor_scale_key)
        else:
            self.logits_key = self.get_group_key(self.base_actor_logits_key)

    def _setup_algorithm(self, env: EnvBase):
        self._setup_actor(env)
        self._setup_critic()



    def _setup_optimizer(self):
        if self.learning_rate is not None:
            self.optimizer = torch.optim.Adam(
                self.loss_module.parameters(),
                lr=self.learning_rate
            )
        else:
            self.actor_optimizer = torch.optim.Adam(
                self.policy.parameters(), lr=self.actor_lr
            )
            self.critic_optimizer = torch.optim.Adam(
                self.critic.parameters(), lr=self.critic_lr
            )

    def _setup_actor(self, env: EnvBase):
        if self.continuous_actions:
            net_module, output_keys, distribution_class, distribution_kwargs = self._setup_continuous_actor_net_module(env=env)
        else:
            net_module, output_keys, distribution_class, distribution_kwargs = self._setup_discrete_actor_net_module(env=env)

        self._actor = ProbabilisticActor(
            module=net_module,
            spec=env.full_action_spec_unbatched[self.action_key],
            in_keys=output_keys,
            out_keys=[self.action_key],
            distribution_class=distribution_class,
            distribution_kwargs=distribution_kwargs,
            return_log_prob=True,
        )

    def _setup_discrete_actor_net_module(self, env: EnvBase):
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
        
        distribution_class = Categorical
        distribution_kwargs = {}

        return net_module,[self.logits_key],distribution_class,distribution_kwargs

    def _setup_continuous_actor_net_module(self,env: EnvBase):
        self._actor_net = torch.nn.Sequential(
            MultiAgentMLP(
                n_agent_inputs=self.n_observations,
                n_agent_outputs=self.n_actions * 2,
                n_agents=self.n_agents,
                centralised=False,
                share_params=self.share_parameters,
                device=self.device,
                depth=self.actor_net_cfg.depth,
                num_cells=self.actor_net_cfg.num_cells,
                activation_class=self.actor_net_cfg.activation_class
            ),
            NormalParamExtractor()
        )
        net_module = TensorDictModule(
            self._actor_net,
            in_keys=[self.observation_key],
            out_keys=[self.loc_key, self.scale_key],
        )
        
        distribution_class = TanhNormal
        distribution_kwargs = {
            "low": env.full_action_spec_unbatched[self.action_key].space.low,
            "high": env.full_action_spec_unbatched[self.action_key].space.high,
        }

        return net_module, [self.loc_key, self.scale_key], distribution_class, distribution_kwargs


    def _setup_critic(self):
        self._critic_net = MultiAgentMLP(
            n_agent_inputs=self.n_observations,
            n_agent_outputs=1,
            n_agents=self.n_agents,
            centralised=not self.independent_critic,
            share_params=self.share_parameters,
            device=self.device,
            depth=self.critic_net_cfg.depth,
            num_cells=self.critic_net_cfg.num_cells,
            activation_class=self.critic_net_cfg.activation_class,
        )
        self.critic = TensorDictModule(
            module=self._critic_net,
            in_keys=[self.observation_key],
            out_keys=[self.state_value_key],
        )

    def _setup_replay_buffer(self):
        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(self.frames_per_batch, device=self.device),
            sampler=SamplerWithoutReplacement(),
            batch_size=self.minibatch_size,
        )

    def _setup_loss(self):
        self.loss_module = ClipPPOLoss(
            actor_network=self.policy,
            critic_network=self.critic,
            clip_epsilon=self.clip_epsilon,
            entropy_coeff=self.entropy_eps,
            normalize_advantage=self.normalize_advantage,
            log_explained_variance=LOG_EXPLAINED_VARIANCE,
        )
        self.loss_module.set_keys(
            reward=self.reward_key,
            action=self.action_key,
            value=self.state_value_key,
            done=self.done_key,
            terminated=self.terminated_key,
        )
        self.loss_module.make_value_estimator(
            ValueEstimators.GAE, gamma=self.gamma, lmbda=self.lmbda
        )
        self.GAE = self.loss_module.value_estimator

    def _setup_collector(self, make_env: callable):
        self.collector = SyncDataCollector(
            make_env,
            self.policy,
            device=self.device,
            storing_device=self.device,
            policy_device=self.device,
            env_device=self.device,
            frames_per_batch=self.frames_per_batch,
            total_frames=self.total_frames,
        )

    def save(self, folder: str, filename: str):
        torch.save({
            "actor_state_dict": self.policy.state_dict(),
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
        self.policy.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])

    def process_batch(self, batch_td):
        flat_td = batch_td.reshape(-1)
        logger.debug(f"Batch TD shape: {flat_td.shape}")
        logger.debug(f"Reward shape: {flat_td.get(self.get_next_key(self.reward_key)).shape}")
        logger.debug(f"Done shape: {flat_td.get(self.get_next_key(self.done_key)).shape}")
        logger.debug(f"Terminated shape: {flat_td.get(self.get_next_key(self.terminated_key)).shape}")
       
        
        reward = flat_td.get(self.get_next_key(self.reward_key))
        reward = reward.reshape(*reward.shape[:2], 1)  # normalize to (B, A, 1) no matter if it was (B, A) or (B, A, 1)
        flat_td.set(self.get_next_key(self.reward_key), reward)

        flat_td.set(
            self.get_next_key(self.done_key),
            flat_td.get(self.get_next_key(self.done_key))
            .unsqueeze(-1)
            .expand((-1,self.n_agents,1)),
        )
        flat_td.set(
            self.get_next_key(self.terminated_key),
            flat_td.get(self.get_next_key(self.terminated_key))
            .unsqueeze(-1)
            .expand((-1,self.n_agents,1)),
        )
        with torch.no_grad():
            self.GAE(
                flat_td,
                params=self.loss_module.critic_network_params,
                target_params=self.loss_module.target_critic_network_params,
            )
        return flat_td.unsqueeze(0)


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
          
    def optimization_step(self,*args, **kwargs):
        self._current_epochs_number = 0
        for _ in range(self.num_epochs):
            self._current_epochs_number += 1
            self._current_minibatch_number = 0
            for _ in range(self.frames_per_batch // self.minibatch_size):
                self._current_minibatch_number += 1
                mini_td = self.replay_buffer.sample()
                loss_dict = self.loss_module(mini_td)    
                if self.learning_rate is not None:
                    self._optimize_single(loss_dict,mini_td)
                else:
                    self._optimize_dual(loss_dict,mini_td)
            if self.metric_dashboard is not None:
                self.metric_dashboard.update(self._optimization_metrics)

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


 