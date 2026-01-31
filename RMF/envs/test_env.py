
import numpy as np



class ZeroSumGameEnv:
    def __init__(self,arg):
        self.env_name = 'ZeroSumGameEnv'
        self.state_dim = 1
        self.action_dim = 1
        self.if_discrete = 0
        self.num_envs = 1

        self.numAgent = arg.numAgent
        self.eta = arg.eta
        self.R = arg.R
        self.cost = np.array(arg.cost)
        self.last_agent = 0   # 看上一回合是谁出的块
        self.block_idx = None  # 记录本回合是否有人出块
        self.totalPower = np.array(arg.totalPower)
        self.agents = [i for i in range(self.numAgent)]

        self.agent0range = arg.agent0range
        self.agent1range = arg.agent1range


    def reset(self):
        # 重置状态
        self.state = np.tile(np.array([1]),(self.numAgent,1))
        return self.state, dict()

    def step(self, action):
        # 从action中获取每个智能体的动作
        if isinstance(action, list):
            action = np.array(action)
        elif isinstance(action, np.ndarray):
            pass
        else:
            raise TypeError("Error: action should be list or np.ndarray")
        action = np.squeeze(action)
        action[0] = self.agent0range[0] + (action[0] + 1) * (self.agent0range[1] - self.agent0range[0]) / 2
        action[1] = self.agent1range[0] + (action[1] + 1) * (self.agent1range[1] - self.agent1range[0]) / 2

        action_agent1 = action[0]
        action_agent2 = action[1]
        # 假设每个智能体的得分与其动作直接相关，且动作之间互为零和
        reward_agent1 = -action_agent1 + action_agent2
        reward_agent2 = -reward_agent1  # 零和博弈
        reward = np.zeros(self.numAgent)
        reward[0] = reward_agent1
        reward[1] = reward_agent2
        # 更新状态，这里简单地使用动作直接更新状态
        self.state = np.tile(np.array([1]),(self.numAgent,1))
        # 判断是否结束，这里简单设置一个回合数限制（比如100步）
        done = False


        # 返回状态，奖励，是否结束
        return self.state, reward, done, {}


class PriceGameEnv:
    def __init__(self,arg):
        self.env_name = 'PriceGameEnv'
        self.state_dim = 1
        self.action_dim = 1
        self.if_discrete = 0
        self.num_envs = 1

        self.numAgent = arg.numAgent
        self.eta = arg.eta
        self.R = arg.R
        self.cost = np.array(arg.cost)
        self.last_agent = 0   # 看上一回合是谁出的块
        self.block_idx = None  # 记录本回合是否有人出块
        self.totalPower = np.array(arg.totalPower)
        self.agents = [i for i in range(self.numAgent)]

        self.agent0range = arg.agent0range
        self.agent1range = arg.agent1range


    def reset(self):
        # 重置状态
        self.state = np.tile(np.array([1]),(self.numAgent,1))
        return self.state, dict()

    def step(self, action):
        # 从action中获取每个智能体的动作
        if isinstance(action, list):
            action = np.array(action)
        elif isinstance(action, np.ndarray):
            pass
        else:
            raise TypeError("Error: action should be list or np.ndarray")
        action = np.squeeze(action)
        action[0] = self.agent0range[0] + (action[0] + 1) * (self.agent0range[1] - self.agent0range[0]) / 2
        action[1] = self.agent1range[0] + (action[1] + 1) * (self.agent1range[1] - self.agent1range[0]) / 2

        action_agent1 = action[0]
        action_agent2 = action[1]
        # 假设每个智能体的得分与其动作直接相关，且动作之间互为零和
        if action_agent1 + action_agent2 == 0:
            reward = np.zeros(self.numAgent)
        else:
            reward_agent1 = self.R*(action_agent1/(action_agent1 + action_agent2))-self.cost[0]*action_agent1
            reward_agent2 = self.R*(action_agent2/(action_agent1 + action_agent2))-self.cost[1]*action_agent2
            reward = np.zeros(self.numAgent)
            reward[0] = reward_agent1
            reward[1] = reward_agent2
        # 更新状态，这里简单地使用动作直接更新状态
        self.state = np.tile(np.array([1]),(self.numAgent,1))
        done = False

        return self.state, reward, done, {}




class SimpleEnv:
    def __init__(self,arg):
        self.env_name = 'SimpleEnv'
        self.state_dim = 1
        self.action_dim = 1
        self.if_discrete = 0
        self.num_envs = 1

        self.numAgent = arg.numAgent
        self.eta = arg.eta
        self.R = arg.R
        self.cost = np.array(arg.cost)
        self.last_agent = 0   # 看上一回合是谁出的块
        self.block_idx = None  # 记录本回合是否有人出块
        self.totalPower = np.array(arg.totalPower)
        self.agents = [i for i in range(self.numAgent)]

        self.agent0range = arg.agent0range
        self.agent1range = arg.agent1range


    def reset(self):
        # 重置状态
        self.state = np.tile(np.array([1]),(self.numAgent,1))
        return self.state, dict()

    def step(self, action):
        # 从action中获取每个智能体的动作
        if isinstance(action, list):
            action = np.array(action)
        elif isinstance(action, np.ndarray):
            pass
        else:
            raise TypeError("Error: action should be list or np.ndarray")
        action = np.squeeze(action)

        # 假设每个智能体的得分与其动作直接相关，且动作之间互为零和
        reward_agent1 = -np.abs(action[0]-0.25)
        reward_agent2 = -np.abs(action[1]-0.25)
        reward = np.zeros(self.numAgent)
        reward[0] = reward_agent1
        reward[1] = reward_agent2
        # 更新状态，这里简单地使用动作直接更新状态
        self.state = np.tile(np.array([1]),(self.numAgent,1))
        done = False

        return self.state, reward, done, {}

