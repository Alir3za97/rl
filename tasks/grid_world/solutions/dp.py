import time
from typing import Any

import numpy as np

from grid_world.env import GridWorld
from grid_world.visualizer import ARROWS, GridWorldVisualizer


class GridWorldDP:
    def __init__(
        self,
        env: GridWorld,
        gamma: float = 0.9,
        max_iterations: int = 1000,
        theta: float = 1e-7,
        visualize: bool = True,
        vis_interval: int = 5,
        sleep_time: float = 0.2,
    ) -> None:
        """Initialize the GridWorldSolution.

        Args:
            env: The GridWorld environment.
            gamma: The discount factor.
            max_iterations: The maximum number of iterations.
            theta: The convergence threshold.
            visualize: Whether to visualize the solution.
            vis_interval: The interval to visualize the solution.
            sleep_time: The time to sleep between iterations.

        """
        self.env = env
        self.gamma = gamma
        self.max_iterations = max_iterations
        self.theta = theta
        self.visualize = visualize
        self.vis_interval = vis_interval
        self.sleep_time = sleep_time
        self.values = np.zeros((env.height, env.width))
        self.policy = self._init_policy()
        self.visualizer = GridWorldVisualizer(env)

    def run(
        self,
    ) -> tuple[dict, np.ndarray]:
        print("Choose algorithm:")
        print("1. Value Iteration")
        print("2. Policy Iteration")

        choice = input("Enter choice (1/2): ").strip()

        if choice == "2":
            print("Solving GridWorld using Policy Iteration...")
            self.policy_iteration()
        else:
            print("Solving GridWorld using Value Iteration...")
            self.value_iteration()

        self.visualize_policy()

        print("\nExecuting optimal policy:")
        user_input = input("Would you like to see the policy in action? (y/n): ").strip().lower()
        if user_input != "y":
            print("Skipping policy execution.")
            return self.policy, self.values
        self.execute_policy()
        return self.policy, self.values

    def value_iteration(self) -> np.ndarray:
        values = np.zeros((self.env.height, self.env.width))

        values[self.env.goal_position] = self.env.goal_reward

        min_val = values.min()
        max_val = values.max()

        for iteration in range(1, self.max_iterations + 1):
            delta = 0.0
            new_values = values.copy()

            for i in range(self.env.height):
                for j in range(self.env.width):
                    if not self._is_valid_state((i, j)):
                        continue

                    old_val = values[i, j]
                    new_values[i, j] = self._calculate_best_action_value((i, j), values)
                    delta = max(delta, abs(old_val - new_values[i, j]))

            values = new_values

            min_val = min(min_val, values.min())
            max_val = max(max_val, values.max())

            self._maybe_visualize(
                iteration=iteration,
                is_converged=(delta < self.theta),
                values=values,
                delta=delta,
                theta=self.theta,
            )

            if delta < self.theta:
                print(f"Value iteration converged in {iteration} iterations")
                break
        else:
            print(f"Warning: Value iteration didn't converge in {self.max_iterations} iterations")

        self.values = values
        self.policy = self._extract_policy()

        return values

    def policy_iteration(self) -> tuple[np.ndarray, dict]:
        policy = self._init_policy()
        values = np.zeros((self.env.height, self.env.width))

        values[self.env.goal_position] = self.env.goal_reward

        for iteration in range(1, self.max_iterations + 1):
            eval_iterations = 0
            delta = float("inf")
            while delta > self.theta:
                eval_iterations += 1
                delta = 0.0
                new_values = values.copy()

                for i in range(self.env.height):
                    for j in range(self.env.width):
                        state = (i, j)
                        if not self._is_valid_state(state):
                            continue

                        old_val = values[i, j]
                        action = policy[state]
                        new_values[i, j] = self._calculate_action_value(state, action, values)
                        delta = max(delta, abs(old_val - new_values[i, j]))

                values = new_values

            self._maybe_visualize(
                iteration=iteration,
                is_converged=False,
                values=values,
                policy=policy,
                eval_iterations=eval_iterations,
            )

            policy_stable = self._policy_improvement(policy, values)

            self._maybe_visualize(
                iteration=iteration,
                is_converged=policy_stable,
                values=values,
                policy=policy,
                policy_stable=policy_stable,
            )

            if policy_stable:
                print(f"Policy iteration converged in {iteration} iterations")
                break
        else:
            print(f"Warning: Policy iteration didn't converge in {self.max_iterations} iterations")

        self.values = values
        self.policy = policy

        return values, policy

    def _init_policy(self) -> dict[tuple[int, int], str]:
        return {
            (i, j): np.random.choice(self.env.allowed_actions)
            for i in range(self.env.height)
            for j in range(self.env.width)
            if self._is_valid_state((i, j))
        }

    def _is_valid_state(self, state: tuple[int, int]) -> bool:
        return state not in self.env.obstacle_positions and state != self.env.goal_position

    def _maybe_visualize(
        self, iteration: int, is_converged: bool = False, **vis_params: dict[str, Any]
    ) -> None:
        should_visualize = iteration % self.vis_interval == 0 or iteration == 1 or is_converged

        if self.visualize and should_visualize:
            self.visualizer.visualize_iteration(iteration=iteration, **vis_params)
            time.sleep(self.sleep_time)

    def _calculate_best_action_value(self, state: tuple[int, int], values: np.ndarray) -> float:
        return max(self._get_action_values(state, values).values())

    def _get_action_values(self, state: tuple[int, int], values: np.ndarray) -> dict[str, float]:
        return {
            action: self._calculate_action_value(state, action, values)
            for action in self.env.allowed_actions
        }

    def _calculate_action_value(
        self, state: tuple[int, int], action: str, values: np.ndarray
    ) -> float:
        original_pos = self.env.current_position
        self.env.current_position = state
        next_pos, reward, done = self.env.step(action)
        self.env.current_position = original_pos

        if done:
            return reward

        return reward + self.gamma * values[next_pos[0], next_pos[1]]

    def _extract_policy(self) -> dict[tuple[int, int], str]:
        policy = {}
        for i in range(self.env.height):
            for j in range(self.env.width):
                state = (i, j)
                if not self._is_valid_state(state):
                    continue

                action_values = self._get_action_values(state, self.values)
                policy[state] = max(action_values, key=action_values.get)
        return policy

    def _policy_improvement(self, policy: dict, values: np.ndarray) -> bool:
        policy_stable = True
        for state, old_action in policy.items():
            action_values = self._get_action_values(state, values)
            policy[state] = max(action_values, key=action_values.get)
            if policy[state] != old_action:
                policy_stable = False
        return policy_stable

    def visualize_policy(self) -> tuple[np.ndarray, np.ndarray]:
        self.visualizer.visualize_values_and_policy(self.values, self.policy)

        policy_grid = np.full((self.env.height, self.env.width), " ", dtype=object)
        value_grid = np.zeros_like(self.values)

        for i in range(self.env.height):
            for j in range(self.env.width):
                if (i, j) == self.env.goal_position:
                    policy_grid[i, j] = "G"
                    value_grid[i, j] = self.env.goal_reward
                elif (i, j) in self.env.obstacle_positions:
                    policy_grid[i, j] = "#"
                else:
                    policy_grid[i, j] = ARROWS.get(self.policy.get((i, j), ""), "?")
                    value_grid[i, j] = self.values[i, j]

        return policy_grid, value_grid

    def _print_grid(self, title: str, grid: np.ndarray, fmt: str = "") -> None:
        max_width = max(
            len(f"{cell:{fmt}}") if isinstance(cell, float) else len(str(cell))
            for row in grid
            for cell in row
        )
        cell_width = max(max_width, 4)
        border = "-" * (self.env.width * (cell_width + 2) + 1)

        print(f"\n{title}:")
        print(border)

        for row in grid:
            cells = []
            for cell in row:
                if isinstance(cell, float):
                    cells.append(f"{cell:>{cell_width}{fmt}}".replace("-", "−"))
                else:
                    cells.append(f"{cell!s:^{cell_width}}")
            print(f"| {' | '.join(cells)} |")
            print(border)

    def execute_policy(self, max_steps: int = 100, render: bool = True) -> tuple[float, int]:
        self.env.reset()
        total_reward = 0.0

        for step in range(1, max_steps + 1):
            if render:
                print(f"Step {step}: Current reward: {total_reward}")
                self.env.render(clear_screen=True)

            state = self.env.current_position
            action = self.policy.get(state)

            if not action:
                print(f"No policy for state {state}")
                break

            _, reward, done = self.env.step(action)
            total_reward += reward

            if done:
                print(f"Goal reached in {step} steps! Total reward: {total_reward}")
                break

        return total_reward, step
