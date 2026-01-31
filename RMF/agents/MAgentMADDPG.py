import torch
import numpy as np
import torch.distributions as D

from .AgentBase import AgentBase
from .AgentTD3 import AgentDDPG
from .AgentTD3 import Actor, Critic



class AgentMADDPG(AgentBase):
    """
    Bases: ``AgentBase``

    Multi-Agent DDPG algorithm. “Multi-Agent Actor-Critic for Mixed Cooperative-Competitive”. R Lowe. et al.. 2017.

    :param net_dim[int]: the dimension of networks (the width of neural networks)
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    :param learning_rate[float]: learning rate of optimizer
    :param gamma[float]: learning rate of optimizer
    :param n_agents[int]: number of agents
    :param if_per_or_gae[bool]: PER (off-policy) or GAE (on-policy) for sparse reward
    :param num_envs[int]: the env number of VectorEnv. num_envs == 1 means don't use VectorEnv
    :param agent_id[int]: if the visible_gpu is '1,9,3,4', agent_id=1 means (1,9,4,3)[agent_id] == 9
    """

    def __init__(
            self,
            net_dim,
            state_dim,
            action_dim,
            args,
            gpu_id: int = 0,
    ):
        super().__init__(net_dims=net_dim,
                                state_dim=state_dim,
                                action_dim=action_dim)
        self.ClassAct = Actor
        self.ClassCri = Critic
        self.if_use_cri_target = True
        self.if_use_act_target = True


        self.agents = [AgentDDPG(net_dims=net_dim,
                                state_dim=state_dim,
                                action_dim=action_dim,
                                 gpu_id = gpu_id,
                                args=args) for i in range(args.arglist.numAgent)]
        self.explore_env = self._explore_one_env
        self.if_off_policy = True
        self.n_agents = args.arglist.numAgent

        self.batch_size = args.batch_size  # num of transitions sampled from replay buffer.
        self.n_states = state_dim
        self.n_actions = action_dim
        self.repeat_times = args.repeat_times
        self.net_dim = net_dim
        self.gamma = args.gamma
        self.update_tau = args.soft_update_tau
        # self.update_tau = 0
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")


    def update_agent(self, rewards, dones, actions, observations, next_obs, index, k1, k2):
        """
        Update the single agent neural networks, called by update_net.

        :param rewards: reward list of the sampled buffer
        :param dones: done list of the sampled buffer
        :param actions: action list of the sampled buffer
        :param observations: observation list of the sampled buffer
        :param next_obs: next_observation list of the sampled buffer
        :param index: ID of the agent
        """
        curr_agent = self.agents[index]
        curr_agent.cri_optimizer.zero_grad()
        all_target_actions = []
        for i in range(self.n_agents):
            if i == index:
                target_action = curr_agent.act_target.get_action(next_obs[:, index], 0)
                # target_action, _ = curr_agent.act_target(next_obs[:, index])
                all_target_actions.append(target_action)
            if i != index:
                action = self.agents[i].act_target.get_action(next_obs[:, i], 0)
                # action, _ = self.agents[i].act_target(next_obs[:, i], 0)
                all_target_actions.append(action)
        action_target_all = (
            torch.cat(all_target_actions, dim=1)
            .to(self.device)
        ).unsqueeze(-1)

        indices = [
            (index - 2) % self.n_agents,  # i-2
            (index - 1) % self.n_agents,  # i-1
            (index + 1) % self.n_agents,  # i+1
            (index + 2) % self.n_agents  # i+2
            # slice((index - 2) % self.n_agents * 2, (index - 2) % self.n_agents * 2 + 2),
            # slice((index - 1) % self.n_agents * 2, (index - 1) % self.n_agents * 2 + 2),
            # slice((index + 1) % self.n_agents * 2, (index + 1) % self.n_agents * 2 + 2),
            # slice((index + 2) % self.n_agents * 2, (index + 2) % self.n_agents * 2 + 2),
        ]

        # neighbor_actions_target = action_target_all[:, indices]
        # avg_neighbor_action_target = neighbor_actions_target.mean(dim=1)
        # current_action_target = action_target_all[:, index]
        # combined_action_target = torch.stack((avg_neighbor_action_target, current_action_target), dim=1)

        # neighbor_actions_target = torch.cat(
        #     [action_target_all[:, idx] for idx in indices], dim=1
        # )
        # neighbor_actions_target = neighbor_actions_target.view(-1, len(indices), 2)
        # avg_neighbor_action_target = neighbor_actions_target.mean(dim=1)  # (batch_size, action_dim)
        # current_action_target = action_target_all[:,
        #                         index * 2: (index + 1) * 2]  # (batch_size, action_dim)
        # combined_action_target = torch.cat((avg_neighbor_action_target, current_action_target), dim=1)

        target_value = rewards[:, index] + self.gamma * curr_agent.cri_target.forward_with_neighbors(
            next_obs.reshape(next_obs.shape[0], next_obs.shape[1] * next_obs.shape[2]),
            next_obs[:, index],
            action_target_all[:, index],
            next_obs[:, indices],
            action_target_all[:, indices]
        ).detach().squeeze(dim=1)

        # neighbor_actions = actions[:, indices, :]
        # avg_neighbor_action = neighbor_actions.mean(dim=1)
        # current_action = actions[:, index, :]
        # combined_action = torch.cat((avg_neighbor_action, current_action), dim=1)
        # indices_1 = [
        #     (index - 2) % self.n_agents,  # i-2
        #     (index - 1) % self.n_agents,  # i-1
        #     (index + 1) % self.n_agents,  # i+1
        #     (index + 2) % self.n_agents  # i+2
        # ]
        # neighbor_actions = actions[:, indices_1, :]
        # avg_neighbor_action = neighbor_actions.mean(dim=1)
        # current_action = actions[:, index, :]
        # combined_action = torch.cat((avg_neighbor_action, current_action), dim=1)
        actual_value = curr_agent.cri.forward_with_neighbors(
            observations.reshape(
                next_obs.shape[0], next_obs.shape[1] * next_obs.shape[2]
            ),
            observations[:, index],
            actions[:, index],
            observations[:, indices],
            actions[:, indices]
        ).squeeze(dim=1)

        vf_loss = curr_agent.criterion(actual_value, target_value.detach()).mean()
        curr_agent.act_optimizer.zero_grad()
        all_mu=[]
        all_std=[]
        for i in range(self.n_agents):
            if i == index:
                mu, std = curr_agent.act(observations[:, index])
                all_mu.append(mu)
                all_std.append(std)
            if i != index:
                mu_j, std_j = self.agents[i].act(observations[:, index])
                all_mu.append(mu_j)
                all_std.append(std_j)
        mu_all = (
            torch.cat(all_mu, dim=1)
            .to(self.device)
        )
        std_all = (
            torch.cat(all_std, dim=1)
            .to(self.device)
        )
        dist = D.Normal(mu, std)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)

        curr_pol_vf_in = curr_agent.act.get_action(observations[:, index], 0)
        # curr_pol_vf_in = curr_pol_out
        all_pol_acs = []
        for i in range(self.n_agents):
            if i == index:
                all_pol_acs.append(curr_pol_vf_in)
            else:
                all_pol_acs.append(actions[:, i])
        actions_act = (
            torch.cat(all_pol_acs, dim=1)
            .to(self.device)
        ).unsqueeze(-1)
        # neighbor_actions_act = actions_act[:, indices]
        # avg_neighbor_action_act = neighbor_actions_act.mean(dim=1)
        # current_action_act = actions_act[:, index]
        # combined_action_act = torch.stack([current_action_act, avg_neighbor_action_act], dim=1)

        # neighbor_actions_act = torch.cat(
        #     [actions_act[:, idx] for idx in indices], dim=1
        # )
        # neighbor_actions_act = neighbor_actions_act.view(-1, len(indices), 2)
        # avg_neighbor_action_act = neighbor_actions_act.mean(dim=1)
        # current_action_act = actions_act[:,
        #                         index * 2: (index + 1) * 2]  # (batch_size, action_dim)
        # combined_action_act = torch.cat((avg_neighbor_action_act, current_action_act), dim=1)
        pol_loss_Q = -torch.mean(
            curr_agent.cri.forward_with_neighbors(
                observations.reshape(
                    observations.shape[0], observations.shape[1] * observations.shape[2]
                ),
                observations[:, index],
                actions_act[:, index],
                observations[:, indices],
                actions_act[:, indices]
            )
        )
        attention_weights = curr_agent.cri.get_attention_weights()
        neighbor_mu = mu_all[:, indices]
        # avg_neighbor_mu = neighbor_mu.mean(dim=1).unsqueeze(1)
        avg_neighbor_mu = torch.sum(
            neighbor_mu * attention_weights,
            dim=1
        ).unsqueeze(1)  # [batch, action_dim]
        # dot_product = torch.sum(mu * neighbor_mu, dim=1)  # [batch_size]
        dot_product = torch.sum(mu * avg_neighbor_mu, dim=1)  # [batch_size]
        norm_i = torch.norm(mu, dim=1)  # [batch_size]
        # norm_j = torch.norm(neighbor_mu, dim=1)  # [batch_size]
        norm_j = torch.norm(avg_neighbor_mu, dim=1)  # [batch_size]
        similarity = dot_product / (norm_i * norm_j + 1e-8)  # 防止除以零

        # pol_loss = pol_loss_Q - 0.01 * entropy.mean() + 0.002 * similarity.mean() ##初始版本
        # pol_loss = pol_loss_Q - 0.02 * entropy.mean() + 0.001 * similarity.mean()
        pol_loss = pol_loss_Q - k1 * entropy.mean() + k2 * similarity.mean()
        curr_agent.act_optimizer.zero_grad()
        pol_loss.backward()
        curr_agent.act_optimizer.step()
        curr_agent.cri_optimizer.zero_grad()
        vf_loss.backward()
        curr_agent.cri_optimizer.step()
        # Update learning rate schedulers
        curr_agent.act_scheduler.step()
        curr_agent.cri_scheduler.step()

        return vf_loss,pol_loss

    def update_net(self, buffer, k1, k2):
        """
        Update the neural networks by sampling batch data from ``ReplayBuffer``.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :param repeat_times: the re-using times of each trajectory.
        :param soft_update_tau: the soft update parameter.
        """
        update_times = int(buffer.cur_size * self.repeat_times / self.batch_size)
        objs_critic = np.zeros((update_times,self.n_agents))
        objs_actor = np.zeros((update_times,self.n_agents))

        for update_t in range(update_times):
                observations,actions,rewards, dones, next_obs = buffer.sample(
                    self.batch_size
                )
                for index in range(self.n_agents):
                    objs_critic[update_t,index],objs_actor[update_t,index]=self.update_agent(rewards, dones, actions, observations, next_obs, index, k1, k2)

                for agent in self.agents:
                    self.soft_update(agent.cri_target, agent.cri, self.update_tau)
                    self.soft_update(agent.act_target, agent.act, self.update_tau)

        # 计算每个batch中每个智能体的平均loss
        obj_avg_critic = np.nanmean(objs_critic, axis=0) if len(objs_critic) else np.zeros((1,self.n_agents))
        obj_avg_actor = np.nanmean(objs_actor, axis=0) if len(objs_actor) else np.zeros((1,self.n_agents))
        obj_avg_critic = obj_avg_critic.squeeze()
        obj_avg_actor = obj_avg_actor.squeeze()
        return obj_avg_critic,obj_avg_actor

    def _explore_one_env(self, env, target_step) -> list:
        """
        Exploring the environment for target_step.
        param env: the Environment instance to be explored.
        param target_step: target steps to explore.
        """
        traj_temp = []
        k = 0
        if not hasattr(self, 'states'):
            self.states,_ = env.reset()
        for _ in range(target_step):
            k += 1
            actions = []
            for i in range(self.n_agents):
                action = self.agents[i].explore_action(torch.tensor(self.states[i],dtype = torch.float32,device=self.device ))
                actions.append(action.tolist())
            # print(actions)
            next_s, reward, done, *_ = env.step(np.array(actions))
            traj_temp.append((self.states, reward, done, actions))
            global_done = done
            if global_done:
                state = env.reset()
                k = 0
            else:
                state = next_s
        self.states = state
        # states = torch.tensor([a[0] for a in traj_temp],dtype = torch.float32).to(self.device)
        # rewards = torch.tensor([a[1] for a in traj_temp],dtype = torch.float32).to(self.device)
        # dones = torch.tensor([a[2] for a in traj_temp],dtype = torch.float32).to(self.device)
        # actions = torch.tensor([a[3] for a in traj_temp],dtype = torch.float32).to(self.device)
        states = torch.tensor(np.array([a[0] for a in traj_temp]), dtype=torch.float32).to(self.device)
        rewards = torch.tensor(np.array([a[1] for a in traj_temp]), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array([a[2] for a in traj_temp]), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array([a[3] for a in traj_temp]), dtype=torch.float32).to(self.device)
        return states,actions,rewards,dones

    def select_actions(self, states):
        """
        Select continuous actions for exploration

        :param state: states.shape==(n_agents,batch_size, state_dim, )
        :return: actions.shape==(n_agents,batch_size, action_dim, ),  -1 < action < +1
        """
        actions = []
        for i in range(self.n_agents):
            action, _ = self.agents[i].act(states[i])
            actions.append(action)
        return actions

    def save_or_load_agent(self, cwd, if_save):
        """
        save or load training files for Agent

        :param cwd: Current Working Directory. ElegantRL save training files in CWD.
        :param if_save: True: save files. False: load files.
        """
        for i in range(self.n_agents):
            self.agents[i].save_or_load_agent(cwd + "/" + str(i), if_save)
