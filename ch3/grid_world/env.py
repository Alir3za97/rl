from typing import Tuple, List
import time


class GridWorld:
    ACTIONS = ["up", "down", "left", "right"]
    ACTION_MAP = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }

    def __init__(
        self, width: int, height: int,
        start_position: Tuple[int, int], goal_position: Tuple[int, int],
        obstacle_positions: List[Tuple[int, int]],
        blocked_reward: float = -1,
        goal_reward: float = 1,
        step_reward: float = 0,
    ):
        self.width = width
        self.height = height
        self.start_position = start_position
        self.current_position = start_position
        self.goal_position = goal_position
        self.boundary_positions = set(
            [(-1, y) for y in range(self.height)] +
            [(self.width, y) for y in range(self.height)] +
            [(x, -1) for x in range(self.width)] +
            [(x, self.height) for x in range(self.width)]
        )

        self.obstacle_positions = set(obstacle_positions)
        self.blocked_reward = blocked_reward
        self.goal_reward = goal_reward
        self.step_reward = step_reward

    def reset(self):
        self.current_position = self.start_position
        return self.current_position

    def step(self, action: str) -> Tuple[Tuple[int, int], float, bool]:
        dx, dy = self.ACTION_MAP[action]
        next_position = (self.current_position[0] + dx, self.current_position[1] + dy)

        if next_position in self.obstacle_positions or next_position in self.boundary_positions:
            return self.current_position, self.blocked_reward, False

        self.current_position = next_position

        if next_position == self.goal_position:
            return next_position, self.goal_reward, True

        return next_position, self.step_reward, False

    def render(self, clear_screen=True, sleep_time=0.1):
        if clear_screen:
            # ANSI escape code to clear screen and move cursor to home position
            print("\033[H\033[J", end="")

        if sleep_time:
            time.sleep(sleep_time)

        RESET = "\033[0m"
        BOLD = "\033[1m"
        RED = "\033[31m"
        GREEN = "\033[32m"
        BLUE = "\033[34m"
        MAGENTA = "\033[35m"

        # Create grid representation
        grid = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                position = (i, j)
                if position == self.current_position:
                    cell = f"{BOLD}{BLUE}A{RESET}"
                elif position == self.goal_position:
                    cell = f"{BOLD}{GREEN}G{RESET}"
                elif position in self.obstacle_positions:
                    cell = f"{BOLD}{RED}#{RESET}"
                else:
                    cell = f"{RESET}·{RESET}"
                row.append(cell)
            grid.append(row)

        print(f"\n{BOLD}Grid World Environment{RESET}")
        print(f"{BOLD}Position: {self.current_position}{RESET}")

        print(f"{MAGENTA}+{'-' * (self.width * 2 + 1)}+{RESET}")

        for row in grid:
            print(f"{MAGENTA}|{RESET} " + " ".join(row) + f" {MAGENTA}|{RESET}")

        print(f"{MAGENTA}+{'-' * (self.width * 2 + 1)}+{RESET}")

        print(f"\n{BOLD}Legend:{RESET}")
        print(f"{BOLD}{BLUE}A{RESET} - Agent")
        print(f"{BOLD}{GREEN}G{RESET} - Goal")
        print(f"{BOLD}{RED}#{RESET} - Obstacle")
        print(f"{RESET}·{RESET} - Empty space")

    def close(self):
        pass
