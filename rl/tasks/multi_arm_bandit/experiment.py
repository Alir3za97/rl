from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

from multi_arm_bandit.agent import Agent
from multi_arm_bandit.bandits import Bandit
from multi_arm_bandit.simulation import Simulation


class ExperimentManager:
    def __init__(
        self,
        agents: dict[str, Agent],
        bandit: Bandit,
        n_steps: int,
        n_runs: int = 1,
        plot_title: str | None = None,
        save_path: str | None = None,
    ) -> None:
        """Initialize the Experiment Manager.

        Args:
            agents: The agents to use in the experiment.
            bandit: The bandit to use in the experiment.
            n_steps: The number of steps to run the experiment.
            n_runs: The number of runs to run the experiment.
            plot_title: The title of the plot.
            save_path: The path to save the plot.

        """
        self.agents = agents
        self.bandit = bandit
        self.n_steps = n_steps
        self.n_runs = n_runs
        self.save_path = save_path
        self.plot_title = plot_title

    def run(self) -> None:
        results = defaultdict(list)
        for name, agent in (p_bar := tqdm(self.agents.items())):
            p_bar.set_description(f"Running {name}")
            simulation = Simulation(agent, self.bandit)

            for _ in tqdm(range(self.n_runs), leave=False, position=1):
                rewards = simulation.run(self.n_steps)
                results[name].append(rewards)

        self._plot_results(results)

    def _plot_results(self, results: dict[str, list[float]]) -> None:
        sns.set_theme(
            style="whitegrid",
            context="notebook",
            palette="deep",
            font="DejaVu Sans",
            font_scale=1.2,
        )
        plt.rcParams["font.family"] = "DejaVu Sans"  # Fallback for matplotlib
        plt.rcParams["axes.unicode_minus"] = False  # Fix minus sign rendering

        plt.figure(figsize=(12, 8))

        palette = sns.color_palette("deep", len(results))

        for i, (name, runs) in enumerate(results.items()):
            runs_array = np.asarray(runs)
            cumulative_rewards = np.cumsum(runs_array, axis=1)

            average_runs_reward = np.mean(cumulative_rewards, axis=0)

            normalized_average_runs_reward = (
                average_runs_reward / np.ones(len(average_runs_reward)).cumsum()
            )

            sns.lineplot(
                x=range(len(normalized_average_runs_reward)),
                y=normalized_average_runs_reward,
                label=name,
                color=palette[i],
                linewidth=2.5,
            )

        plt.title(self.plot_title, fontsize=24, pad=20)
        plt.xlabel("Steps", fontsize=18, labelpad=10)
        plt.ylabel("Average Reward", fontsize=18, labelpad=10)
        plt.tick_params(axis="both", which="major", labelsize=12)

        legend = plt.legend(
            fontsize=14,
            frameon=True,
            fancybox=True,
            framealpha=0.9,
            loc="best",
            edgecolor="lightgray",
        )
        legend.get_frame().set_linewidth(1.0)

        plt.tight_layout()

        plt.savefig(self.save_path, dpi=300, bbox_inches="tight")
