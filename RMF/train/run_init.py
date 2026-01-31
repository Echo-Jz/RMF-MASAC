from .config import Config

class TrainerInit:
    def __init__(self,AgentClass,EnvClass,arglist):
        env_args = self.get_env_args(env=EnvClass(arglist), if_print=True)
        env_args['max_step'] = arglist.max_step
        env_args['arglist'] = arglist

        args = Config(agent_class=AgentClass, env_class=EnvClass, env_args=env_args)  # see `Config` for explanation
        args.net_dims = arglist.net_dims  # the middle layer dimension of MultiLayer Perceptron
        args.gpu_id = arglist.gpu_id  # the ID of single GPU, -1 means CPU
        args.gamma = arglist.gamma  # discount factor of future rewards
        args.learning_rate = arglist.learning_rate  # the learning rate for network updating
        args.clip_grad_norm = arglist.clip_grad_norm # 0.1 ~ 4.0, clip the gradient after normalization
        args.state_value_tau = arglist.state_value_tau  # the tau of normalize for value and state `std = (1-std)*std + tau*std`
        args.soft_update_tau = arglist.soft_update_tau  # 2 ** -8 ~= 5e-3. the tau of soft target update `net = (1-tau)*net + tau*net1`
        args.random_seed = arglist.random_seed # 神经网络随机种子，便于复现
        args.repeat_times = arglist.repeat_times # 每次更新次数=走的步数*repeat_times
        args.explore_noise_std = arglist.explore_noise_std    # 探索噪声，默认0.05

        args.horizon_len = arglist.horizon_len

        args.cwd = arglist.cwd  # current working directory to save model. None means set automatically
        args.if_remove = False  # remove the cwd folder? (True, False, None:ask me)
        args.break_step =  arglist.break_step # break training if 'total_step > break_step'
        args.break_score = arglist.break_score  # break training if `cumulative_rewards > break_score`
        args.if_keep_save = arglist.if_keep_save  # keeping save the checkpoint. False means save until stop training.
        args.if_over_write = arglist.if_over_write  # overwrite the best policy network. `self.cwd/actor.pth`
        args.if_save_buffer = arglist.if_save_buffer  # if save the replay buffer for continuous training after stop training
        args.if_render = arglist.if_render

        args.save_gap = arglist.save_gap  # save actor f"{cwd}/actor_*.pth" for learning curve.
        args.eval_times = arglist.eval_times  # number of times that get the average episodic cumulative return
        args.eval_per_step = arglist.eval_per_step  # evaluate the agent per training steps
        args.eval_env_class = arglist.eval_env_class  # eval_env = eval_env_class(*eval_env_args)
        args.eval_env_args = arglist.eval_env_args  # eval_env = eval_env_class(*eval_env_args)
        args.reward_scale = arglist.reward_scale
        args.arglist = arglist

        if args.if_off_policy:
            args.if_use_per = arglist.if_use_per_or_vtrace
            args.buffer_size = arglist.buffer_size
        else:
            args.if_use_vtrace = arglist.if_use_per_or_vtrace

        self.args = args

    def get_env_args(self,env, if_print: bool) -> dict:
        env_name = env.env_name
        state_dim = env.state_dim
        action_dim = env.action_dim
        if_discrete = env.if_discrete
        num_envs = env.num_envs
        env_args = {'env_name': env_name, 'state_dim': state_dim, 'action_dim': action_dim, 'if_discrete': if_discrete,'num_envs': num_envs}
        print(f"env_args = {repr(env_args)}") if if_print else None
        return env_args
