from maml_rl.task import Task
import random
import time
import copy


class TaskSampler:
    def __init__(self, args, num_agents, batch_size):
        self.args = args
<<<<<<< HEAD
        self.scenarios_names = ["simple_speaker_listener", "simple_reference", "simple_spread"]
=======
<<<<<<< HEAD
        self.scenarios_names = ["simple_speaker_listener"]
        # self.scenarios_names = ["simple", "simple_adversary", "simple_crypto", "simple_push", "simple_reference",
        #                         "simple_speaker_listener", "simple_spread", "simple_tag", "simple_world_comm"]
=======
        # self.scenarios_names = ["simple_spread"]
        self.scenarios_names = ["simple_reference", "simple_speaker_listener", "simple_spread"]
>>>>>>> 4e07b38ea33c197bb5c87550325285cc5db6e430
>>>>>>> master
        self.num_agents = num_agents
        self.batch_size = batch_size

    def sample(self, num_tasks, input_shape):
        random.seed(int(time.time()))
        tasks = []
        for _ in range(num_tasks):
            scenario = random.choice(self.scenarios_names)
            args = copy.copy(self.args)
            args.scenario_name = scenario
            tasks.append(Task(scenario_name=scenario, num_agents=self.num_agents, batch_size=self.batch_size,
                              args=args, input_shape=input_shape))

        random.shuffle(tasks)
        return tasks
