import numpy as np
import torch
import math
from numpy.ma.core import argmin


class BC_env:
    def __init__(self, arg):
        self.env_name = 'BC_env'
        self.state_dim = 5
        self.action_dim = 1
        self.if_discrete = 0
        self.num_envs = 1

        self.num_agents = arg.numAgent        # 智能体数量
        self.f_max = arg.f_max                  # 最大算力
        self.lambda_para_possion = arg.lam      # 任务到达服从泊松分布
        self.data_task = arg.data_task          # 单个任务的数据量
        self.r_task = arg.r_task                # 单个计算任务获得的奖励
        self.r_block = arg.r_block              # 出块奖励
        self.rep_task = arg.rep_task            # 单个计算任务获得的基础信誉
        self.price = arg.price                  # 单位算力单位时间的成本
        self.D = arg.D                          # 出块难度集合
        self.delta = arg.delta                  # 信誉阈值集合
        self.z = arg.z                          # 单次哈希运算需要的计算机周期数
        self.c = arg.c                          # 单位bit任务需要的计算机周期数
        self.B = arg.B                          # 单个区块能承载的最大交易数目
        self.fee = arg.fee                      # 单个交易的交易费
        self.range = arg.range                  # 动作输出范围

        # 动作空间设定为[0.002, 0.998]
        # self.action_space = Box(low=0.002, high=0.998, shape=(1,))

        # 状态空间设定为5维
        # self.observation_space = Box(low=0, high=np.inf, shape=(5,))

        # 初始化状态
        # self.states = np.zeros((self.num_agents, 5))  # 全部智能体的状态表示

    def sample_task_arrival(self, mode):
        """
        mode:
            "poisson"
            "gaussian_tail"
            "uniform_wide"
            "mixture"
        """
        n = self.states.shape[0]
        lam = self.lambda_para_possion
        std = np.sqrt(lam)

        if mode == "poisson":
            x = np.random.poisson(lam, size=n)

        elif mode == "gaussian_tail":
            # -------- 关键：保留左尾，不 clip 到 0 ----------
            x = np.random.normal(lam, std, size=n)

            # 只做“极弱”的下界处理，避免负值爆炸
            # 注意：不是 clip(x,0)
            neg_idx = x < 0
            if np.any(neg_idx):
                # 对负值重新采样（而不是裁剪）
                x[neg_idx] = np.random.normal(lam, std, size=neg_idx.sum())

            # 保留离散性，但不 round
            x = np.floor(x).astype(int)

        elif mode == "uniform_wide":
            # -------- 关键：必须足够宽 ----------
            low = int(lam - 6 * std)  # ≈ 810
            high = int(lam + 4 * std)  # ≈ 1125
            x = np.random.randint(low, high + 1, size=n)

        elif mode == "mixture":
            x = np.zeros(n, dtype=int)
            choice = np.random.choice(3, size=n, p=[1 / 3, 1 / 3, 1 / 3])

            # Poisson
            idx = choice == 0
            if np.any(idx):
                x[idx] = np.random.poisson(lam, size=idx.sum())

            # Gaussian with tail
            idx = choice == 1
            if np.any(idx):
                g = np.random.normal(lam, std, size=idx.sum())
                neg = g < 0
                if np.any(neg):
                    g[neg] = np.random.normal(lam, std, size=neg.sum())
                x[idx] = np.floor(g).astype(int)

            # Wide uniform
            idx = choice == 2
            if np.any(idx):
                low = int(lam - 6 * std)
                high = int(lam + 4 * std)
                x[idx] = np.random.randint(low, high + 1, size=idx.sum())

        else:
            raise ValueError("Unknown arrival mode")

        return x

    def reset(self):
        self.states = np.zeros((self.num_agents, 5))
        self.states[:, 0] = np.random.poisson(lam=self.lambda_para_possion, size=self.states.shape[0])  # 任务到达量
        # self.states[:, 1] = np.random.randint(0, 101, size=self.states.shape[0])  # 当前任务队列
        # self.states[:, 2] = np.random.randint(0, 101, size=self.states.shape[0])  # 交易队列
        # self.states[:, 3] = np.random.randint(0, 40, size=self.states.shape[0])   # 信誉
        for i in range(self.num_agents):
            if self.states[i, 3] < self.delta[0]:
                self.states[i, 4] = self.D[0]
            elif self.states[i, 3] < self.delta[1]:
                self.states[i, 4] = self.D[1]
            elif self.states[i, 3] < self.delta[2]:
                self.states[i, 4] = self.D[2]
            elif self.states[i, 3] < self.delta[3]:
                self.states[i, 4] = self.D[3]
            elif self.states[i, 3] < self.delta[4]:
                self.states[i, 4] = self.D[4]
            # elif self.states[i, 3] < self.delta[5]:
            #     self.states[i, 4] = self.D[5]
            else:
                self.states[i, 4] = self.D[5]
        return self.states, dict()

    def step(self, actions):
        if isinstance(actions, list):
            actions = np.array(actions)
        elif isinstance(actions, np.ndarray):
            pass
        else:
            raise TypeError("Error: action should be list or np.ndarray")
        actions = np.squeeze(actions)
        for i in range(self.num_agents):
            actions[i] = self.range[0] + (actions[i] + 1) * (self.range[1]-self.range[0])/2
        actions_temp = np.zeros((self.num_agents, 2))
        actions_temp[:, 0] = actions
        actions_temp[:, 1] = 1 - actions
        actions = actions_temp

        # 出块时间计算
        q = 2 ** (-self.states[:, 4].reshape(-1, 1))
        Theta = (actions[:, 0].reshape(-1, 1) * q) / self.z
        block_time = np.random.exponential(1 / Theta, size=Theta.shape)
        selected_agent_index = np.argmin(block_time)
        t_block = block_time[selected_agent_index]
        # sum_theta = np.sum(Theta)
        # t_block = 1 / sum_theta
        # prob_block = Theta / sum_theta
        #
        # # 选择出块智能体
        # selected_agent_index = np.random.choice(len(prob_block), p=prob_block.flatten())

        # 计算每个智能体当前区块任务、计算任务等
        task = (self.states[:, 1].reshape(-1, 1) + self.states[:, 0].reshape(-1, 1))
        task_solve = np.minimum(actions[:, 1].reshape(-1, 1) * self.f_max * t_block / (self.c * self.data_task), task)
        task_next = np.maximum(task - task_solve, 0)

        # 更新信誉和出块难度
        rep_next = self.states[:, 3].reshape(-1, 1) + task_solve * self.rep_task
        rep_next[selected_agent_index] = 0  # 出块智能体信誉置0
        D_block = np.zeros((self.num_agents, 1))
        for i in range(self.num_agents):
            if rep_next[i] < self.delta[0]:
                D_block[i, 0] = self.D[0]
            elif rep_next[i] < self.delta[1]:
                D_block[i, 0] = self.D[1]
            elif rep_next[i] < self.delta[2]:
                D_block[i, 0] = self.D[2]
            elif rep_next[i] < self.delta[3]:
                D_block[i, 0] = self.D[3]
            elif rep_next[i] < self.delta[4]:
                D_block[i, 0] = self.D[4]
            # elif rep_next[i] < self.delta[5]:
            #     D_block[i, 0] = self.D[5]
            else:
                D_block[i, 0] = self.D[5]

        # 主交易队列更新
        tranc = self.states[:, 2].reshape(-1, 1) + task_solve  # ？？？task_solve should delete
        niu = min(tranc[selected_agent_index], self.B)
        tranc_next = np.zeros_like(tranc)
        tranc_next[selected_agent_index] = tranc[selected_agent_index] - niu

        # 依据超几何分布随机将其他未出块节点交易打包进入区块,更新各矿工的主交易队列
        tranc_other = np.delete(tranc, selected_agent_index, axis=0)

        hypergeo_population = []
        for i in range(self.num_agents - 1):
            hypergeo_population += [i] * int(tranc_other[i, 0])
        hypergeo_sample_size = int(min(self.B - niu, len(hypergeo_population)))
        hypergeo_sample = np.random.choice(hypergeo_population, size=hypergeo_sample_size, replace=False).astype(
            np.int64)  # replace=False 表示不放回抽样
        hypergeo_value_counts = np.bincount(hypergeo_sample)
        # 调整hypergeo_value_counts输出为固定维数
        output_size = self.num_agents - 1
        if len(hypergeo_value_counts) < output_size:
            fixed_hypergeo_value_counts = np.pad(hypergeo_value_counts, (0, output_size - len(hypergeo_value_counts)),
                                                 'constant')
        else:
            fixed_hypergeo_value_counts = hypergeo_value_counts[:output_size]
        niu_other = fixed_hypergeo_value_counts.reshape(-1, 1)
        tranc_other = np.maximum(tranc_other - niu_other, 0)
        # 更新各矿工的主交易队列
        if selected_agent_index == 0:
            tranc_next[selected_agent_index + 1:] = tranc_other
        elif 0 < selected_agent_index < tranc.shape[0]:
            tranc_next[:selected_agent_index] = tranc_other[:selected_agent_index]
            tranc_next[selected_agent_index + 1:] = tranc_other[selected_agent_index:]
        else:
            tranc_next[:selected_agent_index] = tranc_other

        # 本轮区块中上链任务的归属情况
        niu_next = np.zeros_like(tranc)
        if selected_agent_index == 0:
            niu_next[selected_agent_index + 1:] = niu_other
        elif 0 < selected_agent_index < tranc.shape[0]:
            niu_next[:selected_agent_index] = niu_other[:selected_agent_index]
            niu_next[selected_agent_index + 1:] = niu_other[selected_agent_index:]
        else:
            niu_next[:selected_agent_index] = niu_other
        niu_next[selected_agent_index] = niu

        # 计算奖励
        reward_task = niu_next * self.r_task
        reward_block = np.zeros_like(reward_task)
        reward_block[selected_agent_index] = self.r_block + self.fee * np.sum(niu)


        cost = self.f_max * self.price * t_block
        income = reward_task + reward_block
        reward = income - cost

        # 新任务到达
        # task_rev = np.random.poisson(lam=self.lambda_para_possion, size=(self.states.shape[0], 1))
        arrival = self.sample_task_arrival(mode="mixture")
        task_rev = arrival.reshape(-1, 1)

        # 更新下一个状态
        next_state = np.concatenate((task_rev, task_next, tranc_next, rep_next, D_block), axis=1)
        info = {}
        done = False
        return next_state, reward, done, info, cost, income, reward_task, reward_block

    def get_gmat8(self):
        # 邻接矩阵生成函数（平均场）
        Gmat = torch.zeros((self.num_agents, self.num_agents))
        n = int(math.sqrt(self.num_agents))
        for i in range(n):
            for j in range(n):
                current_index = i * n + j
                up_index = ((i - 1) % n) * n + j
                down_index = ((i + 1) % n) * n + j
                left_index = i * n + ((j - 1) % n)
                right_index = i * n + ((j + 1) % n)
                top_left_index = ((i - 1) % n) * n + ((j - 1) % n)
                bottom_left_index = ((i + 1) % n) * n + ((j - 1) % n)
                top_right_index = ((i - 1) % n) * n + ((j + 1) % n)
                bottom_right_index = ((i + 1) % n) * n + ((j + 1) % n)
                Gmat[current_index, up_index] = 1
                Gmat[current_index, down_index] = 1
                Gmat[current_index, left_index] = 1
                Gmat[current_index, right_index] = 1
                Gmat[current_index, top_left_index] = 1
                Gmat[current_index, bottom_left_index] = 1
                Gmat[current_index, top_right_index] = 1
                Gmat[current_index, bottom_right_index] = 1
        return Gmat

    def get_gmat4(self):
        # 邻接矩阵生成函数（4邻域）
        Gmat = torch.zeros((self.num_agents, self.num_agents))
        n = int(math.sqrt(self.num_agents))
        for i in range(n):
            for j in range(n):
                current_index = i * n + j
                up_index = ((i - 1) % n) * n + j
                down_index = ((i + 1) % n) * n + j
                left_index = i * n + ((j - 1) % n)
                right_index = i * n + ((j + 1) % n)
                Gmat[current_index, up_index] = 1
                Gmat[current_index, down_index] = 1
                Gmat[current_index, left_index] = 1
                Gmat[current_index, right_index] = 1
        return Gmat