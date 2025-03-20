import time
from typing import ClassVar


class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"


class GridWorld:
    ACTIONS = ("up", "down", "left", "right")

    ACTION_MAP: ClassVar[dict[str, tuple[int, int]]] = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }

    def __init__(
        self,
        width: int,
        height: int,
        start_position: tuple[int, int],
        goal_position: tuple[int, int],
        obstacle_positions: list[tuple[int, int]],
        blocked_reward: float = -1,
        goal_reward: float = 1,
        step_reward: float = 0,
    ) -> None:
        """Initialize the GridWorld environment.

        Args:
            width: The width of the grid.
            height: The height of the grid.
            start_position: The starting position of the agent.
            goal_position: The goal position of the agent.
            obstacle_positions: The positions of the obstacles.
            blocked_reward: The reward for blocked actions.
            goal_reward: The reward for reaching the goal.
            step_reward: The reward for each step.

        """
        self.width = width
        self.height = height
        self.start_position = start_position
        self.current_position = start_position
        self.goal_position = goal_position
        self.boundary_positions = set(
            [(-1, y) for y in range(self.height)]
            + [(self.width, y) for y in range(self.height)]
            + [(x, -1) for x in range(self.width)]
            + [(x, self.height) for x in range(self.width)]
        )

        self.obstacle_positions = set(obstacle_positions)
        self.blocked_reward = blocked_reward
        self.goal_reward = goal_reward
        self.step_reward = step_reward

    def reset(self) -> tuple[int, int]:
        self.current_position = self.start_position
        return self.current_position

    def step(self, action: str) -> tuple[tuple[int, int], float, bool]:
        dx, dy = self.ACTION_MAP[action]
        next_position = (self.current_position[0] + dx, self.current_position[1] + dy)

        if next_position in self.obstacle_positions or next_position in self.boundary_positions:
            return self.current_position, self.blocked_reward, False

        self.current_position = next_position

        if next_position == self.goal_position:
            return next_position, self.goal_reward, True

        return next_position, self.step_reward, False

    def render(self, clear_screen: bool = True, sleep_time: float = 0.1) -> None:
        if clear_screen:
            # ANSI escape code to clear screen and move cursor to home position
            print("\033[H\033[J", end="")

        if sleep_time:
            time.sleep(sleep_time)

        # Create grid representation
        grid = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                position = (i, j)
                if position == self.current_position:
                    cell = f"{Colors.BOLD}{Colors.BLUE}A{Colors.RESET}"
                elif position == self.goal_position:
                    cell = f"{Colors.BOLD}{Colors.GREEN}G{Colors.RESET}"
                elif position in self.obstacle_positions:
                    cell = f"{Colors.BOLD}{Colors.RED}#{Colors.RESET}"
                else:
                    cell = f"{Colors.RESET}·{Colors.RESET}"
                row.append(cell)
            grid.append(row)

        print(f"\n{Colors.BOLD}Grid World Environment{Colors.RESET}")
        print(f"{Colors.BOLD}Position: {self.current_position}{Colors.RESET}")

        print(f"{Colors.MAGENTA}+{'-' * (self.width * 2 + 1)}+{Colors.RESET}")

        for row in grid:
            print(
                f"{Colors.MAGENTA}|{Colors.RESET} "
                + " ".join(row)
                + f" {Colors.MAGENTA}|{Colors.RESET}"
            )

        print(f"{Colors.MAGENTA}+{'-' * (self.width * 2 + 1)}+{Colors.RESET}")

        print(f"\n{Colors.BOLD}Legend:{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BLUE}A{Colors.RESET} - Agent")
        print(f"{Colors.BOLD}{Colors.GREEN}G{Colors.RESET} - Goal")
        print(f"{Colors.BOLD}{Colors.RED}#{Colors.RESET} - Obstacle")
        print(f"{Colors.RESET}·{Colors.RESET} - Empty space")

    def close(self) -> None:
        pass
