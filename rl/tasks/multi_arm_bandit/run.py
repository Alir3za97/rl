from jsonargparse import CLI

from rl.tasks.multi_arm_bandit.experiment import ExperimentManager

if __name__ == "__main__":
    CLI(ExperimentManager, as_positional=False)
