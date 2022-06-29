import torch
import multiprocessing as mp

import ENV.make_env as me
from maml_rl.subproc_vec_env import SubprocVecEnv
from maml_rl.BatchEpisode import BatchEpisodes
from maddpg.actor_critic import Critic


def make_env(scenario_name, benchmark=False):
    def _make_env():
        return me.make_env(scenario_name=scenario_name, benchmark=benchmark)

    return _make_env


class Task:
    def __init__(self, scenario_name, batch_size, args, num_workers=mp.cpu_count(), benchmark=False):
        self.args = args
        self.scenario_name = scenario_name
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.queue = mp.Queue()
        # [lambda function]
        env_factorys = [make_env(scenario_name=scenario_name) for _ in range(num_workers)]
        # this is the main process manager, and it will be in charge of num_workers sub-processes interacting with
        # environment.
        self.envs = SubprocVecEnv(env_factorys, queue_=self.queue)
        self._env = me.make_env(scenario_name=scenario_name)

        self.policy = Critic(args=args)

    def sample(self, params=None, gamma=0.95, device='cpu'):
        episodes = BatchEpisodes(batch_size=self.batch_size, gamma=gamma, device=device)
        for i in range(self.batch_size):
            self.queue.put(i)
        for _ in range(self.num_workers):
            self.queue.put(None)

        observations, batch_ids = self.envs.reset()
        dones = [False]
        while (not all(dones)) or (not self.queue.empty()):  # if all done and queue is empty
            # for reinforcement learning, the forward process requires no-gradient
            with torch.no_grad():
                # convert observation to cuda
                # compute policy on cuda
                # convert action to cpu
                observations_tensor = torch.from_numpy(observations).to(device=device)
                # forward via policy network
                # policy network will return Categorical(logits=logits)
                actions_tensor = policy(observations_tensor, params=params).sample()
                actions = actions_tensor.cpu().numpy()

            new_observations, rewards, dones, new_batch_ids, _ = self.envs.step(actions)
            # here is observations NOT new_observations, batch_ids NOT new_batch_ids
            episodes.append(observations, actions, rewards, batch_ids)
            observations, batch_ids = new_observations, new_batch_ids

        return episodes

