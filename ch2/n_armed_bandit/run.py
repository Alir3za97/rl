from jsonargparse import CLI

from n_armed_bandit.experiment import ExperimentManager


if __name__ == "__main__":
    CLI(ExperimentManager, as_positional=False)
