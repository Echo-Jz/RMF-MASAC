# -*- coding: utf-8 -*-
import sys
import argparse

from RMF.train.MADDPG_runner import MADDPG_Trainer
from RMF.envs.BC_env import BC_env
def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for sharding")

    # 环境参数
    parser.add_argument("--numAgent", type=int, default=30, help="智能体个数，默认2")
    parser.add_argument("--f_max", type=int, default=10 ** 10, help="每个节点的总算力")
    parser.add_argument("--lam", type=int, default=1000, help="任务到达服从泊松分布")
    parser.add_argument("--data_task", type=int, default=10 ** 3, help="单个任务的数据量")
    parser.add_argument("--r_task", type=int, default=0.01, help="单个任务获得的奖励")
    parser.add_argument("--r_block", type=int, default=300, help="出块奖励")
    parser.add_argument("--rep_task", type=int, default=0.005, help="单个计算任务获得的基础信誉")
    parser.add_argument("--price", type=int, default=10 ** -20, help="单位时间单位算力的成本")
    parser.add_argument("--D", type=list, default=[28, 24, 20, 16, 12, 10], help="出块难度集合")
    parser.add_argument("--delta", type=list, default=[10, 20, 30, 40, 50], help="信誉阈值集合")
    parser.add_argument("--z", type=int, default=10 ** 4, help="单次哈希运算需要的计算机周期数")
    parser.add_argument("--c", type=int, default=5*10*16, help="单位bit任务需要的计算机周期数")
    parser.add_argument("--B", type=int, default=12000, help="单个区块能承载的最大交易数目")
    parser.add_argument("--fee", type=int, default=0.005, help="单个交易的交易费")
    parser.add_argument("--range", type=list, default=[0.0002, 0.9998], help="智能体0的算力取值范围")
    # parser.add_argument("--agent1range", type=list, default=[0, 1], help="智能体1的算力取值范围")
    """稀疏奖励"""
    parser.add_argument("--if_use_per_or_vtrace", action="store_true", default=False,help="是否使用优先经验回放（异策略，Prioritized Experience Replay，PER），或者是否使用V-trace + GAE（同策略）来应对稀疏奖励")

    """训练"""
    parser.add_argument("--random_seed", type=float, default=1, help="初始随机种子，便于复现")
    parser.add_argument("--horizon_len", type=int, default=200, help="即更新周期，越大更新越不频繁[默认100]，稀疏奖励需要大，记录数据越少，如果总步数比较少这个需要变小。每次更新网络前探索的步数，用于在特定步数后使用经验回放更新一次网络，异策略（DDPG）默认512，同策略（PPO）默认1000")
    parser.add_argument("--break_step", type=int, default=int(50000), help="退出训练的最大步数")
    parser.add_argument("--break_score", type=int, default=sys.maxsize, help="退出训练的最大奖励")
    parser.add_argument("--net_dims", nargs = "+", type = int, default = (256, 128), help="隐藏层神经元矩阵")
    parser.add_argument("--gpu_id", type=int, default=int(0), help="pytorch使用的GPU，-1是CPU")
    parser.add_argument("--gamma", type=float, default=0.97, help="折扣系数")#0.95
    parser.add_argument("--learning_rate", type=float, default=6e-5, help="学习率")#5e-5
    parser.add_argument("--clip_grad_norm", type=float, default=0.5, help="0.1 ~ 4.0,标准化后梯度裁剪值")
    parser.add_argument("--state_value_tau", type=float, default=0, help="标准化参数`std = (1-std)*std + tau*std`")
    parser.add_argument("--soft_update_tau", type=float, default=0.005, help="软更新参数")#0.01
    parser.add_argument("--reward_scale", type=float, default=2 ** -0,help="`reward_scale`是一个用于缩放奖励信号的超参数。它通常被设置为2的某个次方，并且与其他超参数不同，我们可以在训练之前根据环境的累积奖励范围直接选择一个值。这个值的目的是将累积奖励范围限制在较小的范围内，通常在-1000到1000之间，从而使神经网络更容易拟合。这对于深度强化学习算法很有帮助，特别是对于SAC算法效果最为显著。此外，建议在调整`reward_scale`时也要考虑Q值，以确保Q值的绝对值小于256，以便神经网络更好地拟合。")
    parser.add_argument("--repeat_times", type=int, default=1, help="每次更新次数=走的步数*repeat_times，默认异策略为1，同策略为8")
    parser.add_argument("--buffer_size", type=int, default=250000, help="默认经验回放大小1000000，太大，cuda内存不够。需减少。")#200000
    parser.add_argument("--explore_noise_std", type=float, default=0.05,help="探索噪声，默认0.05")

    """评估"""
    parser.add_argument("--if_render", action="store_true", default=False, help="是否可视化环境，需要先写环境可视化接口")
    parser.add_argument("--max_step", type=int, default=1000, help="评估时最大步数，即在特定回合后通过模拟该最大步数来计算累积奖励，通过累积奖励评估模型好坏")
    parser.add_argument("--if_keep_save", action="store_false", default=True, help="持续保存检查点（True）或者直到训练结束才保存（false）")
    parser.add_argument("--cwd", type=str, default=None, help="保存训练过程中产生的模型文件和日志的路径，默认当前工作路径")
    parser.add_argument("--save_gap", type=int, default=int(100), help="每多少步保存一次训练模型， {cwd}/actor_*.pth")
    parser.add_argument("--eval_times", type=int, default=int(100), help="这是评估的次数。在每一次评估中，将运行代理(agent)在环境中执行一系列动作，然后计算这些回合(episode)的累积奖励(cumulative return)的平均值。默认值为3，表示默认要进行3次评估。")
    parser.add_argument("--eval_per_step", type=int, default=int(100), help="这是指定多少训练步骤之后进行一次评估。在训练期间，每当训练步骤达到这个值的倍数时，就会触发一次评估。默认值为20000，表示每20000个训练步骤进行一次评估。")

    parser.add_argument("--if_remove", action="store_true", default=True, help="训练结束后是否删除cwd中的数据，，如果保存了，就可能在接下来的训练中自动加载上次的模型")
    parser.add_argument("--if_over_write", action="store_false", default=True, help="是否overwrite the best policy network. `self.cwd/actor.pth`")
    parser.add_argument("--if_save_buffer", action="store_false", default=True, help="是否在结束时保存经验回放数据")
    parser.add_argument("--if_tensorboard", action="store_false", default=True, help="是否保存tensorboard数据")

    parser.add_argument("--eval_env_class", type=str, default=None, help="这是用于评估的环境类。如果设置了这个参数，代理将在这个环境中进行评估。如果未设置，将使用与训练时相同的环境类。")
    parser.add_argument("--eval_env_args", type=str, default=None, help="这是评估环境的参数。如果设置了self.eval_env_class，那么这些参数将传递给评估环境的构造函数以初始化评估环境。如果未设置，将使用与训练时相同的环境参数。")

    return parser.parse_args()


if __name__ == '__main__':
    arglist = parse_args()
    MADDPG_Trainer(arglist, BC_env)
