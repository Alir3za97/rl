import time
from typing import ClassVar


class GridWorld:
    ACTIONS = ("up", "down", "left", "right")
    KING_ACTIONS = ("up", "down", "left", "right", "up-left", "up-right", "down-left", "down-right")

    ACTION_MAP: ClassVar[dict[str, tuple[int, int]]] = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }

    KING_ACTION_MAP: ClassVar[dict[str, tuple[int, int]]] = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
        "up-left": (-1, -1),
        "up-right": (-1, 1),
        "down-left": (1, -1),
        "down-right": (1, 1),
    }

    def __init__(
        self,
        width: int,
        height: int,
        start_position: tuple[int, int],
        goal_position: tuple[int, int],
        obstacle_positions: list[tuple[int, int]],
        teleports: dict[tuple[int, int], tuple[int, int]],
        allow_king_actions: bool = False,
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
            teleports: The positions of the teleports.
            allow_king_actions: Whether to allow king actions.
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
            [(-1, y) for y in range(-1, self.height + 1)]
            + [(self.width, y) for y in range(-1, self.height + 1)]
            + [(x, -1) for x in range(-1, self.width + 1)]
            + [(x, self.height) for x in range(-1, self.width + 1)]
        )

        self.obstacle_positions = set(obstacle_positions)
        self.teleports = teleports
        self.blocked_reward = blocked_reward
        self.goal_reward = goal_reward
        self.step_reward = step_reward

        self.allowed_actions = self.KING_ACTIONS if allow_king_actions else self.ACTIONS
        self.action_map = self.KING_ACTION_MAP if allow_king_actions else self.ACTION_MAP

        self.renderer = GridWorldTerminalRenderer(self)

    def reset(self) -> tuple[int, int]:
        self.current_position = self.start_position
        return self.current_position

    def step(self, action: str) -> tuple[tuple[int, int], float, bool]:
        if action not in self.allowed_actions:
            raise ValueError(f"Invalid action: {action}")

        dx, dy = self.action_map[action]
        next_position = (self.current_position[0] + dx, self.current_position[1] + dy)

        if next_position in self.obstacle_positions or next_position in self.boundary_positions:
            return self.current_position, self.blocked_reward, False

        if next_position in self.teleports:
            next_position = self.teleports[next_position]

        self.current_position = next_position

        if next_position == self.goal_position:
            return next_position, self.goal_reward, True

        return next_position, self.step_reward, False

    def render(self, clear_screen: bool = True, sleep_time: float = 0.1) -> None:
        self.renderer.render(clear_screen, sleep_time)

    def close(self) -> None:
        self.renderer.close()


class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"


class GridWorldTerminalRenderer:
    def __init__(self, env: GridWorld) -> None:
        """Initialize the GridWorld terminal renderer.

        Args:
            env: The GridWorld environment.

        """
        self.env = env

    def _get_teleport_colors(self) -> dict[tuple[int, int], str]:
        teleport_colors = {}
        teleport_count = 0

        color_codes = [
            196,  # bright red
            46,  # bright green
            21,  # blue
            226,  # yellow
            201,  # pink
            208,  # orange
            93,  # purple
            39,  # cyan
            154,  # lime
            129,  # magenta
        ]

        for source, dest in self.env.teleports.items():
            if source not in teleport_colors:
                color_idx = teleport_count % len(color_codes)
                color_code = color_codes[color_idx]
                teleport_colors[source] = f"\033[38;5;{color_code}m"
                teleport_colors[dest] = f"\033[38;5;{color_code}m"
                teleport_count += 1

        return teleport_colors

    def _get_cell_char(  # noqa: PLR0911
        self, position: tuple[int, int], teleport_colors: dict[tuple[int, int], str]
    ) -> str:
        if position == self.env.current_position:
            return f"{Colors.BOLD}{Colors.BLUE}A{Colors.RESET}"
        if position == self.env.goal_position:
            return f"{Colors.BOLD}{Colors.GREEN}G{Colors.RESET}"
        if position == self.env.start_position:
            return f"{Colors.BOLD}{Colors.MAGENTA}S{Colors.RESET}"
        if position in self.env.obstacle_positions:
            return f"{Colors.BOLD}{Colors.RED}#{Colors.RESET}"
        if position in self.env.teleports:
            return f"{Colors.BOLD}{teleport_colors[position]}0{Colors.RESET}"
        if position in self.env.teleports.values():
            return f"{Colors.BOLD}{teleport_colors[position]}o{Colors.RESET}"
        return f"{Colors.RESET}·{Colors.RESET}"

    def render(self, clear_screen: bool = True, sleep_time: float = 0.1) -> None:
        if clear_screen:
            # ANSI escape code to clear screen and move cursor to home position
            print("\033[H\033[J", end="")

        if sleep_time:
            time.sleep(sleep_time)

        # Create a mapping for teleport colors
        teleport_colors = self._get_teleport_colors()

        # Create grid representation
        grid = []
        for i in range(self.env.height):
            row = []
            for j in range(self.env.width):
                position = (i, j)
                cell = self._get_cell_char(position, teleport_colors)
                row.append(cell)
            grid.append(row)

        print(f"\n{Colors.BOLD}Grid World Environment{Colors.RESET}")
        print(f"{Colors.BOLD}Position: {self.env.current_position}{Colors.RESET}")

        print(f"{Colors.MAGENTA}+{'-' * (self.env.width * 2 + 1)}+{Colors.RESET}")

        for row in grid:
            print(
                f"{Colors.MAGENTA}|{Colors.RESET} "
                + " ".join(row)
                + f" {Colors.MAGENTA}|{Colors.RESET}"
            )

        print(f"{Colors.MAGENTA}+{'-' * (self.env.width * 2 + 1)}+{Colors.RESET}")

        print(f"\n{Colors.BOLD}Legend:{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BLUE}A{Colors.RESET} - Agent")
        print(f"{Colors.BOLD}{Colors.MAGENTA}S{Colors.RESET} - Start")
        print(f"{Colors.BOLD}{Colors.GREEN}G{Colors.RESET} - Goal")
        print(f"{Colors.BOLD}{Colors.RED}#{Colors.RESET} - Obstacle")

        # Display teleport legend
        if self.env.teleports:
            print(f"{Colors.BOLD}Colored '0'/'o'{Colors.RESET} - Teleport entrance/exit pairs")

        print(f"{Colors.RESET}·{Colors.RESET} - Empty space")
