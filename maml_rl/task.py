import torch
import multiprocessing as mp
import numpy as np
import common.utils as me
from maml_rl.subproc_vec_env import SubprocVecEnv
from maml_rl.BatchEpisode import BatchEpisodes
from agent import Agent
from common.replay_buffer import Buffer


def make_env(scenario_name, benchmark=False):
    def _make_env():
        return me.make_env(scenario_name=scenario_name, benchmark=benchmark)

    return _make_env


class Task:
    def __init__(self, scenario_name, num_agents, batch_size, args, num_workers=mp.cpu_count(), benchmark=False):
        self.bk_args = args
        self.scenario_name = scenario_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_agents = num_agents
        self.inner_times = 1000
        self.episode_limit = 100
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.queue = mp.Queue()
        # [lambda function]
        # env_factorys = [make_env(scenario_name=scenario_name, benchmark=benchmark) for _ in range(num_workers)]
        # this is the main process manager, and it will be in charge of num_workers sub-processes interacting with
        # environment.
        # self.envs = SubprocVecEnv(env_factorys, queue_=self.queue)
        args.scenario_name = self.scenario_name
        self.env, self.args = me.make_env(args=args)
        self.buffer = Buffer(args)

        self.agents = [Agent(agent_id=agent_id, args=args) for agent_id in range(self.args.n_agents)]

    '''
    # not used for now
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
                actions_tensor = torch.tensor([self.agents[i].policy.actor_target_network(observations[i]) for i
                                               in range(self.args.n_agents)])
                actions = actions_tensor.cpu().numpy()

            new_observations, rewards, dones, new_batch_ids, _ = self.envs.step(actions)
            # here is observations NOT new_observations, batch_ids NOT new_batch_ids
            episodes.append(observations, actions, rewards, batch_ids)
            observations, batch_ids = new_observations, new_batch_ids

        return episodes
    '''
    def run(self):
        returns = 0
        for time_step in range(self.inner_times):
            # reset the environment
            if time_step % self.episode_limit == 0:
                s = self.env.reset()
            u = []
            actions = []
            with torch.no_grad():
                for agent_id, agent in enumerate(self.agents):
                    action = agent.select_action(s[agent_id], self.noise, self.epsilon)
                    u.append(action)
                    actions.append(action)
            for i in range(self.args.n_agents, self.args.n_players):
                actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
            s_next, r, done, info = self.env.step(actions)
            self.buffer.store_episode(s[:self.args.n_agents], u, r[:self.args.n_agents], s_next[:self.args.n_agents])
            s = s_next
            if self.buffer.current_size >= self.args.batch_size:
                transitions = self.buffer.sample(self.args.batch_size)
                for agent in self.agents:
                    other_agents = self.agents.copy()
                    other_agents.remove(agent)
                    agent.learn(transitions, other_agents)
            self.noise = max(0.05, self.noise - 0.0000005)
            self.epsilon = max(0.05, self.epsilon - 0.0000005)
            if (time_step + 1) == self.inner_times:
                returns = self.evaluate()
        a_loss = []
        q_loss = []
        for agent in self.agents:
            a_loss.append(agent.policy.a_loss.detach().numpy())
            q_loss.append(agent.policy.q_loss.detach().numpy())
        return returns, np.mean(a_loss), np.mean(q_loss)

    def evaluate(self):
        returns = []
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            s = self.env.reset()
            rewards = 0
            for time_step in range(self.args.evaluate_episode_len):
                # self.env.render() 为了跑的快一点，先不要渲染了。
                actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action(s[agent_id], 0, 0)
                        actions.append(action)
                for i in range(self.args.n_agents, self.args.n_players):
                    actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
                s_next, r, done, info = self.env.step(actions)
                rewards += r[0]
                s = s_next
            returns.append(rewards)
            # print('Returns is', rewards)
        return sum(returns) / self.args.evaluate_episodes

    def reset(self):
        self.queue.empty()
        self.env.reset()
        self.buffer = Buffer(self.args)