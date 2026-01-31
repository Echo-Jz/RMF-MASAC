import numpy as np
import torch as th
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F
from copy import deepcopy
from typing import Tuple, List, Optional

from .AgentBase import AgentBase
from .AgentBase import ActorBase, CriticBase
from .AgentBase import build_mlp, layer_init_with_orthogonal
from ..train import Config
from ..train import ReplayBuffer
from torch.optim.lr_scheduler import StepLR

TEN = th.Tensor


class AgentTD3(AgentBase):
    """Twin Delayed DDPG algorithm.
    Addressing Function Approximation Error in Actor-Critic Methods. 2018.
    """

    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        super().__init__(net_dims, state_dim, action_dim, gpu_id, args)
        self.update_freq = getattr(args, 'update_freq', 2)  # standard deviation of exploration noise
        self.num_ensembles = getattr(args, 'num_ensembles', 8)  # the number of critic networks
        self.policy_noise_std = getattr(args, 'policy_noise_std', 0.10)  # standard deviation of exploration noise
        self.explore_noise_std = getattr(args, 'explore_noise_std', 0.05)  # standard deviation of exploration noise

        self.act = Actor(net_dims, state_dim, action_dim).to(self.device)
        self.cri = CriticTwin(net_dims, state_dim, action_dim, num_ensembles=self.num_ensembles).to(self.device)
        self.act_target = deepcopy(self.act)
        self.cri_target = deepcopy(self.cri)
        self.act_optimizer = th.optim.Adam(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = th.optim.Adam(self.cri.parameters(), self.learning_rate)

    def update_objectives(self, buffer: ReplayBuffer, update_t: int) -> Tuple[float, float]:
        assert isinstance(update_t, int)
        with th.no_grad():
            if self.if_use_per:
                (state, action, reward, undone, unmask, next_state,
                 is_weight, is_index) = buffer.sample_for_per(self.batch_size)
            else:
                state, action, reward, undone, unmask, next_state = buffer.sample(self.batch_size)
                is_weight, is_index = None, None

            next_action = self.act.get_action(next_state, action_std=self.policy_noise_std)  # deterministic policy
            next_q = self.cri_target.get_q_values(next_state, next_action).min(dim=1)[0]

            q_label = reward + undone * self.gamma * next_q

        q_values = self.cri.get_q_values(state, action)
        q_labels = q_label.view((-1, 1)).repeat(1, q_values.shape[1])
        td_error = self.criterion(q_values, q_labels).mean(dim=1) * unmask
        if self.if_use_per:
            obj_critic = (td_error * is_weight).mean()
            buffer.td_error_update_for_per(is_index.detach(), td_error.detach())
        else:
            obj_critic = td_error.mean()
        if self.lambda_fit_cum_r != 0:
            cum_reward_mean = buffer.cum_rewards[buffer.ids0, buffer.ids1].detach_().mean().repeat(q_values.shape[1])
            obj_critic += self.criterion(cum_reward_mean, q_values.mean(dim=0)).mean() * self.lambda_fit_cum_r
        self.optimizer_backward(self.cri_optimizer, obj_critic)
        self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

        if update_t % self.update_freq == 0:  # delay update
            action_pg = self.act(state)  # action to policy gradient
            obj_actor = self.cri(state, action_pg).mean()
            self.optimizer_backward(self.act_optimizer, -obj_actor)
            self.soft_update(self.act_target, self.act, self.soft_update_tau)
        else:
            obj_actor = th.tensor(th.nan)
        return obj_critic.item(), obj_actor.item()


class AgentDDPG(AgentBase):
    """DDPG(Deep Deterministic Policy Gradient)
    Continuous control with deep reinforcement learning. 2015.
    """

    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=gpu_id, args=args)
        self.explore_noise_std = getattr(args, 'explore_noise', 0.05)  # set for `self.get_policy_action()`
        self.n_agent = args.arglist.numAgent

        self.act = Actor(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim).to(self.device)
        # self.cri = Critic(net_dims=net_dims, state_dim=state_dim*self.n_agent, action_dim=action_dim*2).to(self.device)
        self.cri = Critic(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, num_agents=self.n_agent).to(self.device)
        self.act_target = deepcopy(self.act)
        self.cri_target = deepcopy(self.cri)
        self.act_optimizer = th.optim.Adam(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = th.optim.Adam(self.cri.parameters(), self.learning_rate)
        # Add learning rate schedulers
        self.act_scheduler = StepLR(self.act_optimizer, step_size=6000,
                                    gamma=0.95)  # Reduce LR every 10 steps by a factor of 0.95
        self.cri_scheduler = StepLR(self.cri_optimizer, step_size=6000, gamma=0.95)



class OrnsteinUhlenbeckNoise:
    def __init__(self, size: int, theta=0.15, sigma=0.3, ou_noise=0.0, dt=1e-2):
        """
        The noise of Ornstein-Uhlenbeck Process

        Source: https://github.com/slowbull/DDPG/blob/master/src/explorationnoise.py
        It makes Zero-mean Gaussian Noise more stable.
        It helps agent explore better in an inertial system.
        Don't abuse OU Process. OU process has too many hyperparameters and over fine-tuning make no sense.

        int size: the size of noise, shape = (-1, action_dim)
        float theta: related to the not independent of OU-noise
        float sigma: related to action noise std
        float ou_noise: initialize OU-noise
        float dt: derivative
        """
        self.theta = theta
        self.sigma = sigma
        self.ou_noise = ou_noise
        self.dt = dt
        self.size = size

    def __call__(self) -> float:
        """
        output a OU-noise

        return array ou_noise: a noise generated by Ornstein-Uhlenbeck Process
        """
        noise = self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.ou_noise -= self.theta * self.ou_noise * self.dt + noise
        return self.ou_noise


'''network'''


class SelfAttentionLayer(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, key_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.key_dim = key_dim
        # 将状态-动作对映射到Query/Key空间
        self.query = nn.Linear(state_dim + action_dim, key_dim)
        self.key = nn.Linear(state_dim + action_dim, key_dim)

    def forward(
            self,
            self_state: TEN,
            self_action: TEN,
            neighbor_states: TEN,
            neighbor_actions: TEN
    ) -> Tuple[TEN, TEN]:

        # 拼接自身状态-动作对
        self_sa = th.cat([self_state, self_action], dim=-1)  # [batch, state_dim + action_dim]

        # 拼接邻居状态-动作对
        neighbor_sa = th.cat([neighbor_states, neighbor_actions],
                                dim=-1)  # [batch, num_neighbors, state_dim + action_dim]

        # 计算Query和Key
        Q = self.query(self_sa).unsqueeze(1)  # [batch, 1, key_dim]
        K = self.key(neighbor_sa)  # [batch, num_neighbors, key_dim]

        # 计算注意力权重
        attention_scores = th.matmul(K, Q.transpose(-1, -2)).squeeze(-1)  # [batch, num_neighbors]
        attention_scores = attention_scores / (self.key_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch, num_neighbors]

        # 加权聚合邻居动作
        aggregated_actions = th.sum(
            neighbor_actions * attention_weights.unsqueeze(-1),
            dim=1
        )  # [batch, action_dim]

        return aggregated_actions, attention_weights

class Actor(ActorBase):
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        # self.net = build_mlp(dims=[state_dim, *net_dims, action_dim])
        # layer_init_with_orthogonal(self.net[-1], std=0.1)
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.mean = nn.Linear(64, action_dim)  # 输出均值
        self.log_std = nn.Linear(64, action_dim)  # 输出对数标准差
        # 正交初始化
        layer_init_with_orthogonal(self.mean, std=0.1)
        layer_init_with_orthogonal(self.log_std, std=1.0)

    def get_action(self, state: TEN, action_std: float) -> TEN:  # for exploration
        # action_avg = self.net(state).tanh()
        # dist = self.ActionDist(action_avg, action_std)
        # action = dist.sample()
        # entropy = dist.entropy().sum(dim=-1, keepdim=True)
        x = self.gelu(self.fc1(state))
        x = self.gelu(self.fc2(x))
        mu = th.tanh(self.mean(x))  # Mean action in [-1, 1]
        log_std = th.clamp(self.log_std(x), -20, 2)  # Log std deviation
        std = th.exp(log_std)
        dist = D.Normal(mu, std)
        action = dist.rsample()
        return action.clip(-1.0, 1.0)
        # return action.clip(-1.0, 1.0), mu, std


class Critic(CriticBase):
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int, num_agents: int):
        super().__init__(state_dim=state_dim * num_agents, action_dim=action_dim * 2)
        self.num_agents = num_agents
        self.attention = SelfAttentionLayer(state_dim, action_dim)
        self.net = build_mlp(dims=[state_dim*self.num_agents + action_dim*2, *net_dims, 1])
        layer_init_with_orthogonal(self.net[-1], std=0.5)

        self.last_attention_weights = None

    def get_attention_weights(self) -> Optional[TEN]:
        return self.last_attention_weights

    # 新增方法处理邻居逻辑
    def forward_with_neighbors(
            self,
            state: TEN,
            self_state: TEN,
            self_action: TEN,
            neighbor_states: TEN,
            neighbor_actions: TEN
    ) -> TEN:
        aggregated_action, attention_weights = self.attention(
            self_state, self_action,
            neighbor_states, neighbor_actions
        )
        self.last_attention_weights = attention_weights  # 存储权重
        combined_action = th.cat((self_action, aggregated_action), dim=-1)
        return self(state, combined_action)  # 调用原始forward


class CriticTwin(CriticBase):  # shared parameter
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int, num_ensembles: int = 2):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net = build_mlp(dims=[state_dim + action_dim, *net_dims, num_ensembles])
        layer_init_with_orthogonal(self.net[-1], std=0.5)
