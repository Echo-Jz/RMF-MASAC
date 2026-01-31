import os
import time
import numpy as np
import torch as th
from typing import Tuple, List

from .config import Config

TEN = th.Tensor


class Evaluator:
    def __init__(self, cwd: str, env, args: Config, if_tensorboard: bool = False):
        self.cwd = cwd  # current working directory to save model
        self.env = env  # the env for Evaluator, `eval_env = env` in default
        self.agent_id = args.gpu_id
        self.total_step = 0  # the total training step
        self.start_time = time.time()  # `used_time = time.time() - self.start_time`
        self.eval_times = args.eval_times  # number of times that get episodic cumulative return
        self.eval_per_step = args.eval_per_step  # evaluate the agent per training steps
        self.eval_step_counter = -self.eval_per_step  # `self.total_step > self.eval_step_counter + self.eval_per_step`

        self.save_gap = args.save_gap
        self.save_counter = 0
        self.if_keep_save = args.if_keep_save
        self.if_over_write = args.if_over_write

        self.recorder_path = f'{cwd}/recorder.npy'
        self.recorder = []  # total_step, r_avg, r_std, critic_value, ...
        self.recorder_step = args.eval_record_step  # start recording after the exploration reaches this step.
        self.max_r = -np.inf
        self.args = args
        self.n_agent = args.arglist.numAgent
        print("| Evaluator:"
              "\n| `step`: Number of samples, or total training steps, or running times of `env.step()`."
              "\n| `time`: Time spent from the start of training to this moment."
              "\n| `avgR`: Average value of cumulative rewards, which is the sum of rewards in an episode."
              "\n| `stdR`: Standard dev of cumulative rewards, which is the sum of rewards in an episode."
              "\n| `avgS`: Average of steps in an episode."
              "\n| `objC`: Objective of Critic network. Or call it loss function of critic network."
              "\n| `objA`: Objective of Actor network. It is the average Q value of the critic network."
              f"\n{'#' * 80}\n", flush=True)
        arg = args
        out_str = f"{'ID':<3}{'Step':>8}{'Time':>8} |"+' '.join([f"{f'avgR{i}':>8}" for i in range(arg.arglist.numAgent)])+' '.join([f"{f'stdR{i}':>8}" for i in range(arg.arglist.numAgent)])
        out_str += ' '.join([f"{'avgS':>7}{'stdS':>6} |"])+' '.join([f"{f'expR{i}':>8}" for i in range(arg.arglist.numAgent)])
        out_str += ' '.join([f"{f'objC{i}':>7}{f'objA{i}':>7}" for i in range(arg.arglist.numAgent)])
        print(out_str, flush=True)

        if getattr(env, 'num_envs', 1) == 1:  # get attribute
            self.get_cumulative_rewards_and_step = self.get_cumulative_rewards_and_step_single_env
        else:  # vectorized environment
            self.get_cumulative_rewards_and_step = self.get_cumulative_rewards_and_step_vectorized_env





    def multiagent_evaluate_and_save(self, agent, steps: int, exp_r: float, logging_tuple: tuple):
        self.total_step += steps  # update total training steps

        if self.total_step < self.recorder_step:
            return
        if self.total_step < self.eval_step_counter + self.eval_per_step:
            return

        self.eval_step_counter = self.total_step
        returns = np.zeros((self.eval_times,agent.n_agents))
        cost = np.zeros((self.eval_times,agent.n_agents))
        income = np.zeros((self.eval_times,agent.n_agents))
        reward_task = np.zeros((self.eval_times,agent.n_agents))
        reward_block = np.zeros((self.eval_times,agent.n_agents))
        actions = np.zeros((self.eval_times,agent.n_agents))
        # actions = np.zeros((self.eval_times, agent.n_agents,2))
        steps = np.zeros(self.eval_times)
        for i in range(self.eval_times):
            return_n,step_n,total_action,cost_n,income_n,reward_task_n,reward_block_n = multiagent_get_rewards_and_steps(self.env, agent)
            returns[i] = return_n.flatten()
            cost[i] = cost_n.flatten()
            income[i] = income_n.flatten()
            reward_task[i] = reward_task_n.flatten()
            reward_block[i] = reward_block_n.flatten()
            actions[i] = total_action
            steps[i] = step_n



        returns = th.tensor(returns, dtype=th.float32)
        cost = th.tensor(cost, dtype=th.float32)
        income = th.tensor(income, dtype=th.float32)
        reward_task = th.tensor(reward_task, dtype=th.float32)
        reward_block = th.tensor(reward_block, dtype=th.float32)
        steps = th.tensor(steps, dtype=th.float32)
        avg_r = returns.mean(dim = 0)
        std_r = returns.std(dim = 0)
        avg_s = steps.mean().item()
        std_s = steps.std().item()
        avg_action = np.mean(actions,axis = 0)
        std_action = np.std(actions,axis = 0)
        avg_cost = cost.mean(dim = 0)
        std_cost = cost.std(dim = 0)
        avg_income = income.mean(dim = 0)
        std_income = income.std(dim=0)
        avg_reward_task = reward_task.mean(dim = 0)
        std_reward_task = reward_task.std(dim=0)
        avg_reward_block = reward_block.mean(dim = 0)
        std_reward_block = reward_block.std(dim=0)

        train_time = int(time.time() - self.start_time)

        '''record the training information'''
        # recorder格式为：step avgR1 avgR2 ... stdR1 stdR2 ... expR1 expR2 ... objc1 obja1 objc2 obja2 ...

        recorder = [self.total_step]+avg_r.tolist()+std_r.tolist()+exp_r.tolist()+logging_tuple[0].tolist()+logging_tuple[1].tolist()+avg_action.tolist()+std_action.tolist()+avg_cost.tolist()+std_cost.tolist()+avg_income.tolist()+std_income.tolist()+avg_reward_task.tolist()+std_reward_task.tolist()+avg_reward_block.tolist()+std_reward_block.tolist()

        self.recorder.append(recorder)  # update recorder


        '''print some information to Terminal'''

        out_str = f"{self.agent_id:<3}{self.total_step:8.2e}{train_time:8.0f} |"+' '.join([f"{a :8.2f}" for a in avg_r.tolist()])+' '.join([f"{a :7.1f}" for a in std_r.tolist()])+f"{avg_s:7.0f}{std_s:6.0f} |"
        out_str += " ".join([f"{a :8.2f}" for a in exp_r.tolist()])
        out_str += " ".join([f"{b :8.2f}" for a in logging_tuple for b in a ])

        print(out_str, flush=True)

        if_save = True
        if if_save:
            self.save_training_curve_jpg()
        if not self.if_keep_save:
            return

        self.save_counter += 1
        # actor_path = None
        # for i in range(agent.n_agents):
        #     if if_save:  # save checkpoint with the highest episode return
        #         actor_path = f"{self.cwd}/actor{i}__{self.total_step:012}.pt"
        #
        #     elif self.save_counter == self.save_gap:
        #         self.save_counter = 0
        #         actor_path = f"{self.cwd}/actor{i}__{self.total_step:012}.pt"
        #
        #     if actor_path:
        #         th.save(agent.agents[i].act.state_dict(), actor_path)  # save policy network in *.pt



    def evaluate_and_save(self, actor: th.nn, steps: int, exp_r: float, logging_tuple: tuple):
        self.total_step += steps  # update total training steps

        if self.total_step < self.recorder_step:
            return
        if self.total_step < self.eval_step_counter + self.eval_per_step:
            return

        self.eval_step_counter = self.total_step

        rewards_step_ten = self.get_cumulative_rewards_and_step(actor)

        returns = rewards_step_ten[:, 0]  # episodic cumulative returns of an
        steps = rewards_step_ten[:, 1]  # episodic step number
        avg_r = returns.mean().item()
        std_r = returns.std().item()
        avg_s = steps.mean().item()
        std_s = steps.std().item()

        train_time = int(time.time() - self.start_time)

        '''record the training information'''

        self.recorder.append((self.total_step, avg_r, std_r, exp_r, *logging_tuple))  # update recorder
        if self.tensorboard:
            self.tensorboard.add_scalar("info/critic_loss_sample", logging_tuple[0], self.total_step)
            self.tensorboard.add_scalar("info/actor_obj_sample", -1 * logging_tuple[1], self.total_step)
            self.tensorboard.add_scalar("reward/avg_reward_sample", avg_r, self.total_step)
            self.tensorboard.add_scalar("reward/std_reward_sample", std_r, self.total_step)
            self.tensorboard.add_scalar("reward/exp_reward_sample", exp_r, self.total_step)

            self.tensorboard.add_scalar("info/critic_loss_time", logging_tuple[0], train_time)
            self.tensorboard.add_scalar("info/actor_obj_time", -1 * logging_tuple[1], train_time)
            self.tensorboard.add_scalar("reward/avg_reward_time", avg_r, train_time)
            self.tensorboard.add_scalar("reward/std_reward_time", std_r, train_time)
            self.tensorboard.add_scalar("reward/exp_reward_time", exp_r, train_time)

        '''print some information to Terminal'''
        prev_max_r = self.max_r
        self.max_r = max(self.max_r, avg_r)  # update max average cumulative rewards
        print(f"{self.agent_id:<3}{self.total_step:8.2e}{train_time:8.0f} |"
              f"{avg_r:8.2f}{std_r:7.1f}{avg_s:7.0f}{std_s:6.0f} |"
              f"{exp_r:8.2f}{''.join(f'{n:7.2f}' for n in logging_tuple)}", flush=True)

        if_save = avg_r > prev_max_r
        if if_save:
            self.save_training_curve_jpg()
        if not self.if_keep_save:
            return

        self.save_counter += 1
        # actor_path = None
        # if if_save:  # save checkpoint with the highest episode return
        #     if self.if_over_write:
        #         actor_path = f"{self.cwd}/actor.pt"
        #     else:
        #         actor_path = f"{self.cwd}/actor__{self.total_step:012}.pt"
        #
        # elif self.save_counter == self.save_gap:
        #     self.save_counter = 0
        #     if self.if_over_write:
        #         actor_path = f"{self.cwd}/actor.pt"
        #     else:
        #         actor_path = f"{self.cwd}/actor__{self.total_step:012}.pt"
        #
        # if actor_path:
        #     th.save(actor.state_dict(), actor_path)  # save policy network in *.pt

    def save_or_load_recoder(self, if_save: bool):
        if if_save:
            recorder_ary = np.array(self.recorder)
            np.save(self.recorder_path, recorder_ary)
        elif os.path.exists(self.recorder_path):
            recorder = np.load(self.recorder_path)
            self.recorder = [tuple(i) for i in recorder]  # convert numpy to list
            self.total_step = self.recorder[-1][0]

    def get_cumulative_rewards_and_step_single_env(self, actor) -> TEN:
        rewards_steps_list = [get_rewards_and_steps(self.env, actor) for _ in range(self.eval_times)]
        rewards_steps_ten = th.tensor(rewards_steps_list, dtype=th.float32)
        return rewards_steps_ten  # rewards_steps_ten.shape[1] == 2

    def get_cumulative_rewards_and_step_vectorized_env(self, actor) -> TEN:
        rewards_step_list = [get_cumulative_rewards_and_step_from_vec_env(self.env, actor)
                             for _ in range(max(1, self.eval_times // self.env.num_envs))]
        rewards_step_list = sum(rewards_step_list, [])
        rewards_step_ten = th.tensor(rewards_step_list)
        return rewards_step_ten  # rewards_steps_ten.shape[1] == 2

    def save_training_curve_jpg(self):
        # recoder格式为：step avgR1 avgR2 ... stdR1 stdR2 ... expR1 expR2 ... objc1 obja1 objc2 obja2 ...

        recorder = np.array(self.recorder)

        train_time = int(time.time() - self.start_time)
        total_step = int(self.recorder[-1][0])
        fig_title = f"step_time_maxR_{int(total_step)}_{int(train_time)}"

        # draw_learning_curve(recorder=recorder, fig_title=fig_title, save_path=f"{self.cwd}", n_agent = self.n_agent)
        np.save(self.recorder_path, recorder)  # save self.recorder for `draw_learning_curve()`


"""util"""


def multiagent_get_rewards_and_steps(env, agent) -> Tuple[float, int]:
    """Usage
    eval_times = 4
    net_dim = 2 ** 7
    actor_path = './LunarLanderContinuous-v2_PPO_1/actor.pt'

    env = build_env(env_class=env_class, env_args=env_args)
    actor = agent(net_dim, env.state_dim, env.action_dim, gpu_id=gpu_id).act
    actor.load_state_dict(th.load(actor_path, map_location=lambda storage, loc: storage))

    r_s_ary = [get_episode_return_and_step(env, act) for _ in range(eval_times)]
    r_s_ary = np.array(r_s_ary, dtype=np.float32)
    r_avg, s_avg = r_s_ary.mean(axis=0)  # average of episode return and episode step
    """
    max_step = env.max_step
    device = agent.device  # net.parameters() is a Python generator.

    state, info_dict = env.reset()
    episode_steps = 0
    cumulative_returns = 0.0  # sum of rewards in an episode
    cumulative_costs = 0.0
    cumulative_incomes = 0.0
    cumulative_rewards_task = 0.0
    cumulative_rewards_block = 0.0
    total_action = np.zeros((max_step,agent.n_agents))
    # total_action = np.zeros((max_step,agent.n_agents,2))
    for episode_steps in range(max_step):
        tensor_state = th.as_tensor(state, dtype=th.float32, device=device)
        actions = agent.select_actions(tensor_state)
        total_action[episode_steps]=np.array([i.item() for i in actions])
        state, reward, terminated, _, cost, income, reward_task, reward_block = env.step([i.item() for i in actions])   # 输入一维列表
        cumulative_returns += reward
        cumulative_costs += cost
        cumulative_incomes += income
        cumulative_rewards_task += reward_task
        cumulative_rewards_block +=reward_block

    total_action=np.mean(total_action, axis=0)
    env_unwrapped = getattr(env, 'unwrapped', env)
    cumulative_returns = getattr(env_unwrapped, 'cumulative_returns', cumulative_returns)
    cumulative_costs = getattr(env_unwrapped, 'cumulative_costs', cumulative_costs)
    cumulative_incomes = getattr(env_unwrapped, 'cumulative_returns', cumulative_returns)
    cumulative_rewards_task = getattr(env_unwrapped, 'cumulative_rewards_task', cumulative_rewards_task)
    cumulative_rewards_block = getattr(env_unwrapped, 'cumulative_rewards_block', cumulative_rewards_block)
    return cumulative_returns,episode_steps + 1,total_action,cumulative_costs,cumulative_incomes,cumulative_rewards_task,cumulative_rewards_block



def get_rewards_and_steps(env, actor, if_render: bool = False) -> Tuple[float, int]:
    """Usage
    eval_times = 4
    net_dim = 2 ** 7
    actor_path = './LunarLanderContinuous-v2_PPO_1/actor.pt'

    env = build_env(env_class=env_class, env_args=env_args)
    actor = agent(net_dim, env.state_dim, env.action_dim, gpu_id=gpu_id).act
    actor.load_state_dict(th.load(actor_path, map_location=lambda storage, loc: storage))

    r_s_ary = [get_episode_return_and_step(env, act) for _ in range(eval_times)]
    r_s_ary = np.array(r_s_ary, dtype=np.float32)
    r_avg, s_avg = r_s_ary.mean(axis=0)  # average of episode return and episode step
    """
    max_step = env.max_step
    device = next(actor.parameters()).device  # net.parameters() is a Python generator.

    state, info_dict = env.reset()
    episode_steps = 0
    cumulative_returns = 0.0  # sum of rewards in an episode
    for episode_steps in range(max_step):
        tensor_state = th.as_tensor(state, dtype=th.float32, device=device).unsqueeze(0)
        tensor_action = actor(tensor_state)
        action = tensor_action.detach().cpu().numpy()[0]  # not need detach(), because using th.no_grad() outside
        state, reward, terminated, truncated, _ = env.step(action)
        cumulative_returns += reward

        if if_render:
            env.render()
        if terminated or truncated:
            break
    else:
        print("| get_rewards_and_step: WARNING. max_step > 12345", flush=True)

    env_unwrapped = getattr(env, 'unwrapped', env)
    cumulative_returns = getattr(env_unwrapped, 'cumulative_returns', cumulative_returns)
    return cumulative_returns, episode_steps + 1


def get_cumulative_rewards_and_step_from_vec_env(env, actor) -> List[Tuple[float, int]]:
    device = env.device
    env_num = env.num_envs
    max_step = env.max_step
    '''get returns and dones (GPU)'''
    returns = th.empty((max_step, env_num), dtype=th.float32, device=device)
    dones = th.empty((max_step, env_num), dtype=th.bool, device=device)

    state, info_dict = env.reset()  # must reset in vectorized env
    for t in range(max_step):
        action = actor(state.to(device))
        # assert action.shape == (num_envs, ) if if_discrete else (num_envs, action_dim)
        state, reward, terminal, truncate, info_dict = env.step(action)

        returns[t] = reward
        dones[t] = th.logical_or(terminal, truncate)

    '''get cumulative returns and step'''
    if hasattr(env, 'cumulative_returns'):  # GPU
        returns_step_list = [(ret, env.max_step) for ret in env.cumulative_returns]
    else:  # CPU
        returns = returns.cpu()
        dones = dones.cpu()

        returns_step_list = []
        for i in range(env_num):
            dones_where = th.where(dones[:, i].eq(1))[0] + 1
            episode_num = len(dones_where)
            if episode_num == 0:
                continue

            j0 = 0
            for j1 in dones_where.tolist():
                reward_sum = returns[j0:j1, i].sum().item()  # cumulative returns of an episode
                steps_num = j1 - j0  # step number of an episode
                returns_step_list.append((reward_sum, steps_num))

                j0 = j1
    return returns_step_list


def draw_learning_curve(recorder: np.ndarray = None,
                        fig_title: str = 'learning_curve',
                        save_path: str = '',n_agent = 1):
    steps = recorder[:, 0]  # x-axis is training steps
    r_avg = recorder[:, 1:1+n_agent]
    r_std = recorder[:, 1+n_agent:1+2*n_agent]
    r_exp = recorder[:, 1+2*n_agent:1+3*n_agent]
    obj_c = recorder[:, 1+3*n_agent:1+5*n_agent:2]
    obj_a = recorder[:, 2+3*n_agent:2+5*n_agent:2]
    action_avg = recorder[:, 1+5*n_agent:1+6*n_agent]
    action_std = recorder[:, 1+6*n_agent:1+7*n_agent]
    avg_cost = recorder[:, 1 + 7 * n_agent:1 + 8 * n_agent]
    std_cost = recorder[:, 1 + 8 * n_agent:1 + 9 * n_agent]
    avg_income = recorder[:, 1 + 9 * n_agent:1 + 10 * n_agent]
    std_income = recorder[:, 1 + 10 * n_agent:1 + 11 * n_agent]
    avg_reward_task = recorder[:, 1 + 11 * n_agent:1 + 12 * n_agent]
    std_reward_task = recorder[:, 1 + 12 * n_agent:1 + 13 * n_agent]
    avg_reward_block = recorder[:, 1 + 13 * n_agent:1 + 14 * n_agent]
    std_reward_block = recorder[:, 1 + 14 * n_agent:1 + 15 * n_agent]

    '''plot subplots'''
    import matplotlib as mpl
    mpl.use('Agg')
    """Generating matplotlib graphs without a running X server [duplicate]
    write `mpl.use('Agg')` before `import matplotlib.pyplot as plt`
    https://stackoverflow.com/a/4935945/9293137
    """

    import matplotlib.pyplot as plt
    for i in range(n_agent):
        save_path1 = save_path + f'/LearningCurve{i}.jpg'
        fig, axs = plt.subplots(2)

        '''axs[0]'''
        ax00 = axs[0]
        ax00.cla()

        ax01 = axs[0].twinx()
        color01 = 'darkcyan'
        ax01.set_ylabel('Explore AvgReward', color=color01)
        ax01.plot(steps, r_exp[:,i], color=color01, alpha=0.5, )
        ax01.tick_params(axis='y', labelcolor=color01)

        color0 = 'lightcoral'
        ax00.set_ylabel('Episode Return', color=color0)
        ax00.plot(steps, r_avg[:,i], label='Episode Return', color=color0)
        ax00.fill_between(steps, r_avg[:,i] - r_std[:,i], r_avg[:,i] + r_std[:,i], facecolor=color0, alpha=0.3)
        ax00.grid()
        '''axs[1]'''
        ax10 = axs[1]
        ax10.cla()

        ax11 = axs[1].twinx()
        color11 = 'darkcyan'
        ax11.set_ylabel('objC', color=color11)
        ax11.fill_between(steps, obj_c[:,i], facecolor=color11, alpha=0.2, )
        ax11.tick_params(axis='y', labelcolor=color11)

        color10 = 'royalblue'
        ax10.set_xlabel('Total Steps')
        ax10.set_ylabel('objA', color=color10)
        ax10.plot(steps, obj_a[:,i], label='objA', color=color10)
        ax10.tick_params(axis='y', labelcolor=color10)

        ax10.legend()
        ax10.grid()

        '''plot save'''
        plt.title(fig_title+f'agent{i}', y=2.3)
        plt.savefig(save_path1)
        plt.close('all')  # avoiding warning about too many open figures, rcParam `figure.max_open_warning`
        # plt.show()  # if use `mpl.use('Agg')` to draw figures without GUI, then plt can't plt.show()

    for i in range(n_agent):
        save_path1 = save_path + f'/actionCurve{i}.jpg'
        fig, axs = plt.subplots(1)



        axs.cla()
        color0 = 'lightcoral'
        axs.set_ylabel('Episode Return', color=color0)
        axs.set_xlabel('Total Steps')
        axs.plot(steps, action_avg[:,i], label='Episode Return', color=color0)
        axs.fill_between(steps, action_avg[:,i] - action_std[:,i], action_avg[:,i] + action_std[:,i], facecolor=color0, alpha=0.3)
        axs.grid()


        '''plot save'''
        plt.title(fig_title+f'agent{i}')
        plt.savefig(save_path1)
        plt.close('all')  # avoiding warning about too many open figures, rcParam `figure.max_open_warning`





"""learning curve"""


def demo_evaluator_actor_pth():
    import gym
    from elegantrl.agents.AgentPPO import AgentPPO
    from elegantrl.train.config import Config, build_env

    gpu_id = 0  # >=0 means GPU ID, -1 means CPU

    agent_class = AgentPPO

    env_class = gym.make
    env_args = {'num_envs': 1,
                'env_name': 'LunarLanderContinuous-v2',
                'max_step': 1000,
                'state_dim': 8,
                'action_dim': 2,
                'if_discrete': False,
                'target_return': 200,

                'id': 'LunarLanderContinuous-v2'}

    # actor_path = './LunarLanderContinuous-v2_PPO_1/actor.pt'
    eval_times = 4
    net_dim = 2 ** 7

    '''init'''
    args = Config(agent_class=agent_class, env_class=env_class, env_args=env_args)
    env = build_env(env_class=args.env_class, env_args=args.env_args)
    act = agent_class(net_dim, env.state_dim, env.action_dim, gpu_id=gpu_id, args=args).act
    # act.load_state_dict(th.load(actor_path, map_location=lambda storage, loc: storage))

    '''evaluate'''
    r_s_ary = [get_rewards_and_steps(env, act) for _ in range(eval_times)]
    r_s_ary = np.array(r_s_ary, dtype=np.float32)
    r_avg, s_avg = r_s_ary.mean(axis=0)  # average of episode return and episode step

    print(f'|r_avg {r_avg}  s_avg {s_avg}', flush=True)
    return r_avg, s_avg


def demo_evaluate_actors(dir_path: str, gpu_id: int, agent, env_args: dict, eval_times=2, net_dim=128):
    import gym
    from elegantrl.train.config import build_env
    # dir_path = './LunarLanderContinuous-v2_PPO_1'
    # gpu_id = 0
    # agent_class = AgentPPO
    # net_dim = 2 ** 7

    env_class = gym.make
    # env_args = {'num_envs': 1,
    #             'env_name': 'LunarLanderContinuous-v2',
    #             'max_step': 1000,
    #             'state_dim': 8,
    #             'action_dim': 2,
    #             'if_discrete': False,
    #             'target_return': 200,
    #             'eval_times': 2 ** 4,
    #
    #             'id': 'LunarLanderContinuous-v2'}
    # eval_times = 2 ** 1

    '''init'''
    env = build_env(env_class=env_class, env_args=env_args)
    act = agent(net_dim, env.state_dim, env.action_dim, gpu_id=gpu_id).act

    '''evaluate'''
    step_epi_r_s_ary = []

    act_names = [name for name in os.listdir(dir_path) if len(name) == 19]
    for act_name in act_names:
        act_path = f"{dir_path}/{act_name}"

        act.load_state_dict(th.load(act_path, map_location=lambda storage, loc: storage))
        r_s_ary = [get_rewards_and_steps(env, act) for _ in range(eval_times)]
        r_s_ary = np.array(r_s_ary, dtype=np.float32)
        r_avg, s_avg = r_s_ary.mean(axis=0)  # average of episode return and episode step

        step = int(act_name[6:15])

        step_epi_r_s_ary.append((step, r_avg, s_avg))

    step_epi_r_s_ary = np.array(step_epi_r_s_ary, dtype=np.float32)

    '''sort by step'''
    step_epi_r_s_ary = step_epi_r_s_ary[step_epi_r_s_ary[:, 0].argsort()]
    return step_epi_r_s_ary


def demo_load_pendulum_and_render():
    import torch
    from elegantrl.agents.AgentPPO import AgentPPO
    from elegantrl.train.config import Config, build_env

    gpu_id = 0  # >=0 means GPU ID, -1 means CPU

    agent_class = AgentPPO

    from elegantrl.envs.CustomGymEnv import PendulumEnv
    env_class = PendulumEnv
    env_args = {'num_envs': 1,
                'env_name': 'Pendulum-v1',
                'state_dim': 3,
                'action_dim': 1,
                'if_discrete': False, }

    actor_path = './Pendulum-v1_PPO_0/actor.pt'
    net_dim = 2 ** 7

    '''init'''
    env = build_env(env_class=env_class, env_args=env_args)
    args = Config(agent_class=agent_class, env_class=env_class, env_args=env_args)
    act = agent_class(net_dim, env.state_dim, env.action_dim, gpu_id=gpu_id, args=args).act
    act.load_state_dict(torch.load(actor_path, map_location=lambda storage, loc: storage))

    '''evaluate'''
    # eval_times = 2 ** 7
    # from elegantrl.envs.CustomGymEnv import PendulumEnv
    # eval_env = PendulumEnv()
    # from elegantrl.train.evaluator import get_cumulative_returns_and_step
    # r_s_ary = [get_cumulative_returns_and_step(eval_env, act) for _ in range(eval_times)]
    # r_s_ary = np.array(r_s_ary, dtype=np.float32)
    # r_avg, s_avg = r_s_ary.mean(axis=0)  # average of episode return and episode step
    #
    # print(f'|r_avg {r_avg}  s_avg {s_avg}', flush=True)

    '''render'''
    max_step = env.max_step
    if_discrete = env.if_discrete
    device = next(act.parameters()).device  # net.parameters() is a Python generator.

    state = env.reset()
    steps = None
    returns = 0.0  # sum of rewards in an episode
    for steps in range(max_step):
        s_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        a_tensor = act(s_tensor).argmax(dim=1) if if_discrete else act(s_tensor)
        action = a_tensor.detach().cpu().numpy()[0]  # not need detach(), because using th.no_grad() outside
        state, reward, done, _ = env.step(action * 2)  # for Pendulum specially
        returns += reward
        env.render()

        if done:
            break
    returns = getattr(env, 'cumulative_returns', returns)
    steps += 1

    print(f"\n| cumulative_returns {returns}"
          f"\n|      episode steps {steps}", flush=True)


def run():
    from elegantrl.agents.AgentPPO import AgentPPO
    flag_id = 1  # int(sys.argv[1])

    gpu_id = [2, 3][flag_id]
    agent = AgentPPO
    env_args = [
        {'num_envs': 1,
         'env_name': 'LunarLanderContinuous-v2',
         'max_step': 1000,
         'state_dim': 8,
         'action_dim': 2,
         'if_discrete': False,
         'target_return': 200,
         'eval_times': 2 ** 4,
         'id': 'LunarLanderContinuous-v2'},

        {'num_envs': 1,
         'env_name': 'BipedalWalker-v3',
         'max_step': 1600,
         'state_dim': 24,
         'action_dim': 4,
         'if_discrete': False,
         'target_return': 300,
         'eval_times': 2 ** 3,
         'id': 'BipedalWalker-v3', },
    ][flag_id]
    env_name = env_args['env_name']

    print('gpu_id', gpu_id, flush=True)
    print('env_name', env_name, flush=True)

    '''save step_epi_r_s_ary'''
    # cwd_path = '.'
    # dir_names = [name for name in os.listdir(cwd_path)
    #              if name.find(env_name) >= 0 and os.path.isdir(name)]
    # for dir_name in dir_names:
    #     dir_path = f"{cwd_path}/{dir_name}"
    #     step_epi_r_s_ary = demo_evaluate_actors(dir_path, gpu_id, agent, env_args)
    #     np.savetxt(f"{dir_path}-step_epi_r_s_ary.txt", step_epi_r_s_ary)

    '''load step_epi_r_s_ary'''
    step_epi_r_s_ary = []

    cwd_path = '.'
    ary_names = [name for name in os.listdir('.')
                 if name.find(env_name) >= 0 and name[-4:] == '.txt']
    for ary_name in ary_names:
        ary_path = f"{cwd_path}/{ary_name}"
        ary = np.loadtxt(ary_path)
        step_epi_r_s_ary.append(ary)
    step_epi_r_s_ary = np.vstack(step_epi_r_s_ary)
    step_epi_r_s_ary = step_epi_r_s_ary[step_epi_r_s_ary[:, 0].argsort()]
    print('step_epi_r_s_ary.shape', step_epi_r_s_ary.shape, flush=True)

    '''plot'''
    import matplotlib.pyplot as plt
    # plt.plot(step_epi_r_s_ary[:, 0], step_epi_r_s_ary[:, 1])

    plot_x_y_up_dw_step = []
    n = 8
    for i in range(0, len(step_epi_r_s_ary), n):
        y_ary = step_epi_r_s_ary[i:i + n, 1]
        if y_ary.shape[0] <= 1:
            continue

        y_avg = y_ary.mean()
        y_up = y_ary[y_ary > y_avg].mean()
        y_dw = y_ary[y_ary <= y_avg].mean()

        y_step = step_epi_r_s_ary[i:i + n, 2].mean()
        x_avg = step_epi_r_s_ary[i:i + n, 0].mean()
        plot_x_y_up_dw_step.append((x_avg, y_avg, y_up, y_dw, y_step))

    if_show_episode_step = True
    color0 = 'royalblue'
    color1 = 'lightcoral'
    # color2 = 'darkcyan'
    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    #           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    title = f"{env_name}_{agent.__name__}_ElegantRL"

    fig, ax = plt.subplots(1)

    plot_x = [item[0] for item in plot_x_y_up_dw_step]
    plot_y = [item[1] for item in plot_x_y_up_dw_step]
    plot_y_up = [item[2] for item in plot_x_y_up_dw_step]
    plot_y_dw = [item[3] for item in plot_x_y_up_dw_step]
    ax.plot(plot_x, plot_y, label='Episode Return', color=color0)
    ax.fill_between(plot_x, plot_y_up, plot_y_dw, facecolor=color0, alpha=0.3)
    ax.set_ylabel('Episode Return', color=color0)
    ax.tick_params(axis='y', labelcolor=color0)
    ax.grid(True)

    if if_show_episode_step:
        ax_twin = ax.twinx()
        plot_y_step = [item[4] for item in plot_x_y_up_dw_step]
        ax_twin.fill_between(plot_x, 0, plot_y_step, facecolor=color1, alpha=0.3)
        ax_twin.set_ylabel('Episode Step', color=color1)
        ax_twin.tick_params(axis='y', labelcolor=color1)
        ax_twin.set_ylim(0, np.max(plot_y_step) * 2)

    print('title', title, flush=True)
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    # demo_evaluate_actors()
    run()
