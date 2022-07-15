from common.arguments import get_args
from maml_rl.metalearner import MetaLearner
from maml_rl.task_sampler import TaskSampler

def main():
    args = get_args()
    sampler = TaskSampler(args=args, num_agents=5, batch_size=1)
    leaner = MetaLearner(args=args, sampler=sampler)
    leaner.train()


if __name__ == '__main__':
    main()
