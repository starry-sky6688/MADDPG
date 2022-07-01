import torch
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
from maddpg.actor_critic import Critic


class MetaLearner:

    def __init__(self, args, sampler, gamma=0.95, outer_lr=0.5, inner_lr=0.5, tau=1.0, device='cpu'):
        self.task_sampler = sampler
        self.gamma = gamma
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.tau = tau
        # self.to(device)

        self.centralized_q = Critic(args=args)

        self.args = args
        self.train_step = 0
        self.total_training_step = 20000
        self.batch_size = 16
        self.num_tasks = 8

    def train(self):
        for i in range(self.total_training_step):
            tasks = self.task_sampler.sample(num_tasks=self.num_tasks)
            temp_q_functions = []
            for t in tasks:
                # load generalized centralized q function
                for a in t.agents:
                    pass
                    # a.policy.critic_target_network.load_state_dict(self.centralized_q.state_dict())
                # inner training
                r = t.run()
                temp_q_functions.append(t.agents[0].policy.critic_target_network.state_dict())
            # outer training

