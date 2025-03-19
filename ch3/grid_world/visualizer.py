# DISCLAIMER: This code is LLM generated. Don't ask me about it.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects

# Direction arrows for visualization
ARROWS = {"up": "↑", "down": "↓", "left": "←", "right": "→"}


class GridWorldVisualizer:
    """Matplotlib-based visualizer for grid world environments."""

    def __init__(self, env, figsize=(10, 8)):
        """Initialize visualizer with grid world environment."""
        self.env = env
        self.figsize = figsize
        # Create a custom colormap: red (negative) -> white (zero) -> green (positive)
        self.cmap = LinearSegmentedColormap.from_list(
            "value_cmap", [(0, "#ff5555"), (0.5, "#f8f8f2"), (1, "#50fa7b")]  # Dracula-inspired colors
        )
        # Direction vectors for policy arrows
        self.directions = {"up": (0, -0.4), "down": (0, 0.4), "left": (-0.4, 0), "right": (0.4, 0)}
        # Create persistent figure and axes
        self.fig = None
        self.ax = None
        # Track colorbar to prevent duplicates
        self.colorbar = None
        # Set dark theme colors
        self.bg_color = "#282a36"  # Dark background
        self.grid_color = "#44475a"  # Subtle grid lines
        self.text_color = "#f8f8f2"  # Light text
        self.accent_color = "#bd93f9"  # Purple accent
        self.wall_color = "#6272a4"  # Blue-ish gray
        self.goal_color = "#50fa7b"  # Green
        self.arrow_color = "#ff79c6"  # Pink
        self.arrow_edge_color = "#f8f8f2"  # Light edge

    def _ensure_figure_exists(self):
        """Make sure we have a figure to plot on."""
        if self.fig is None or not plt.fignum_exists(self.fig.number):
            plt.style.use("dark_background")
            self.fig, self.ax = plt.subplots(figsize=self.figsize)
            self.fig.patch.set_facecolor(self.bg_color)
            # Make the figure respond to window resize
            self.fig.canvas.mpl_connect("resize_event", lambda event: plt.tight_layout())

    def visualize_iteration(
        self,
        iteration,
        values,
        delta=None,
        theta=None,
        policy=None,
        policy_stable=None,
        eval_iterations=None,
        show=True,
        save_path=None,
        title=None,
    ):
        """Visualize a single iteration of value/policy iteration."""
        # Use persistent figure instead of creating new ones
        self._ensure_figure_exists()

        # Clear the figure completely
        self.fig.clf()

        # Recreate axes
        self.ax = self.fig.add_subplot(111)

        # Set up main title
        if title is not None:
            self.fig.suptitle(title, fontsize=14)
        else:
            # Original title logic
            if policy is None:
                title = f"Value Iteration - Iteration {iteration}"
                if delta is not None and theta is not None:
                    title += f"\nMax delta: {delta:.6f} (threshold: {theta})"
            else:
                title = f"Policy Iteration - Iteration {iteration}"
                if eval_iterations is not None:
                    title += f"\nPolicy evaluation: {eval_iterations} iterations"
                if policy_stable is not None:
                    title += f"\nPolicy {'stable' if policy_stable else 'updated'}"
            self.fig.suptitle(title, fontsize=14)

        # Plot the heatmap of values
        self._plot_grid(self.ax, values, policy)

        if show:
            # Use pause instead of show to update without blocking
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path)
            plt.draw()
            plt.pause(0.001)  # Small pause to update the display

        return self.fig, self.ax

    def visualize_values_and_policy(self, values, policy):
        """Visualize both values and policy arrows with matplotlib."""
        return self.visualize_iteration(
            iteration=0, values=values, policy=policy, title="Grid World Values and Policy", show=True
        )

    def visualize_values(self, values):
        """Visualize values with color coding using matplotlib."""
        return self.visualize_iteration(iteration=0, values=values, title="Grid World Values", show=True)

    def _plot_grid(self, ax, values, policy=None):
        """Plot the grid with values as a heatmap and optional policy arrows."""
        height, width = self.env.height, self.env.width

        # Set dark theme for the axes
        ax.set_facecolor(self.bg_color)

        # Setup heatmap and colorbar
        self._setup_heatmap(ax, values)

        # Add grid lines
        self._add_grid_lines(ax, width, height)

        # Add cell values and special cell styling
        self._add_cell_details(ax, values, policy)

        # Add policy arrows if provided
        if policy:
            self._add_policy_arrows(ax, policy)

        # Setup axes styling and labels
        self._setup_axes_styling(ax, width, height)

    def _setup_heatmap(self, ax, values):
        """Create the heatmap and set up colorbar."""
        # Normalize values for color mapping
        vmin, vmax = values.min(), values.max()
        # If values span negative to positive, center the colormap at zero
        if vmin < 0 < vmax:
            vabs = max(abs(vmin), abs(vmax))
            vmin, vmax = -vabs, vabs

        # Create the heatmap
        im = ax.imshow(values, cmap=self.cmap, origin="upper", vmin=vmin, vmax=vmax, alpha=0.7)

        # Add colorbar - safely handle existing colorbar
        try:
            # Only try to remove if it exists and belongs to current figure
            if self.colorbar is not None and self.colorbar.ax.figure == self.fig:
                self.colorbar.remove()
        except (AttributeError, KeyError):
            # Ignore any colorbar removal errors
            pass

        # Create fresh colorbar
        self.colorbar = plt.colorbar(im, ax=ax)
        self.colorbar.set_label("Value", color=self.text_color)
        self.colorbar.ax.yaxis.set_tick_params(color=self.text_color)
        plt.setp(plt.getp(self.colorbar.ax.axes, "yticklabels"), color=self.text_color)

    def _add_grid_lines(self, ax, width, height):
        """Add grid lines to the plot."""
        ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
        ax.grid(which="minor", color=self.grid_color, linestyle="-", linewidth=1.5)

    def _add_cell_details(self, ax, values, policy):
        """Add text and styling to grid cells."""
        height, width = self.env.height, self.env.width

        for i in range(height):
            for j in range(width):
                value = values[i, j]

                # Different styling for obstacles and goal
                if (i, j) in self.env.obstacle_positions:
                    self._style_obstacle_cell(ax, i, j)
                    continue

                if (i, j) == self.env.goal_position:
                    self._style_goal_cell(ax, i, j)

                # Add value text to all cells - with LOWER opacity where policy exists
                text_alpha = 0.4 if policy and policy.get((i, j)) in self.directions else 0.9
                self._add_value_text(ax, i, j, value, text_alpha)

    def _style_obstacle_cell(self, ax, i, j):
        """Style an obstacle cell."""
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, color=self.wall_color, alpha=0.8))
        ax.text(j, i, "WALL", ha="center", va="center", fontsize=8, color=self.text_color, weight="bold")

    def _style_goal_cell(self, ax, i, j):
        """Style the goal cell."""
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, color=self.goal_color, alpha=0.3))
        ax.text(j, i, "GOAL", ha="center", va="center", fontsize=8, color=self.text_color, weight="bold")

    def _add_value_text(self, ax, i, j, value, text_alpha):
        """Add value text to a cell."""
        text = ax.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center",
            color=self.text_color,
            fontsize=8,
            weight="bold",
            alpha=text_alpha,
            zorder=5,  # Lower zorder to appear below arrows
        )
        text.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground="black", alpha=text_alpha)])

    def _add_policy_arrows(self, ax, policy):
        """Add policy arrows to the grid."""
        height, width = self.env.height, self.env.width

        for i in range(height):
            for j in range(width):
                if (i, j) not in self.env.obstacle_positions and (i, j) != self.env.goal_position:
                    action = policy.get((i, j))
                    if action in self.directions:
                        dx, dy = self.directions[action]
                        ax.arrow(
                            j,
                            i,
                            dx,
                            dy,
                            head_width=0.3,
                            head_length=0.3,
                            fc=self.arrow_color,
                            ec=self.arrow_edge_color,
                            linewidth=2.5,
                            alpha=1.0,
                            zorder=10,  # Full opacity, higher zorder
                        )

    def _setup_axes_styling(self, ax, width, height):
        """Set up axis labels and styling."""
        ax.set_xticks(range(width))
        ax.set_yticks(range(height))
        ax.set_xticklabels(range(width), color=self.text_color)
        ax.set_yticklabels(range(height), color=self.text_color)
        ax.set_xlabel("Column", color=self.text_color)
        ax.set_ylabel("Row", color=self.text_color)
        ax.spines["bottom"].set_color(self.grid_color)
        ax.spines["top"].set_color(self.grid_color)
        ax.spines["left"].set_color(self.grid_color)
        ax.spines["right"].set_color(self.grid_color)
        ax.tick_params(axis="x", colors=self.text_color)
        ax.tick_params(axis="y", colors=self.text_color)

    def print_grid(self, grid, title="", fmt=""):
        """Print a grid as a matplotlib visualization."""
        self._ensure_figure_exists()

        # Clear completely to avoid colorbar issues
        self.fig.clf()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(self.bg_color)

        if title:
            self.ax.set_title(title, color=self.text_color)

        # Simple heatmap if grid contains numbers
        if np.issubdtype(grid.dtype, np.number):
            # Set up colormap to match main visualization
            vmin, vmax = grid.min(), grid.max()
            # If values span negative to positive, center the colormap at zero
            if vmin < 0 < vmax:
                vabs = max(abs(vmin), abs(vmax))
                vmin, vmax = -vabs, vabs

            im = self.ax.imshow(grid, cmap=self.cmap, vmin=vmin, vmax=vmax, alpha=0.7)

            # Handle colorbar safely
            try:
                if self.colorbar is not None and self.colorbar.ax.figure == self.fig:
                    self.colorbar.remove()
            except (AttributeError, KeyError):
                pass

            # Create fresh colorbar
            self.colorbar = plt.colorbar(im, ax=self.ax)
            self.colorbar.set_label("Value", color=self.text_color)
            self.colorbar.ax.yaxis.set_tick_params(color=self.text_color)
            plt.setp(plt.getp(self.colorbar.ax.axes, "yticklabels"), color=self.text_color)

            # Add text values with smaller font
            height, width = grid.shape
            for i in range(height):
                for j in range(width):
                    value = grid[i, j]
                    text = f"{value:{fmt}}" if fmt else f"{value}"
                    text_obj = self.ax.text(
                        j, i, text, ha="center", va="center", color=self.text_color, fontsize=8, weight="bold"
                    )
                    text_obj.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground="black")])
        else:
            # For non-numeric grids (e.g., policy arrows)
            self.ax.imshow(np.zeros_like(grid, dtype=float), cmap="viridis", alpha=0.1)
            height, width = grid.shape
            for i in range(height):
                for j in range(width):
                    self.ax.text(j, i, str(grid[i, j]), ha="center", va="center", fontsize=8, color=self.text_color)

        # Style grid lines and spines
        self.ax.grid(color=self.grid_color, linestyle="-", linewidth=1)
        self.ax.spines["bottom"].set_color(self.grid_color)
        self.ax.spines["top"].set_color(self.grid_color)
        self.ax.spines["left"].set_color(self.grid_color)
        self.ax.spines["right"].set_color(self.grid_color)

        self.ax.set_xticks(range(width))
        self.ax.set_yticks(range(height))
        self.ax.set_xticklabels(range(width), color=self.text_color)
        self.ax.set_yticklabels(range(height), color=self.text_color)

        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)  # Update without blocking
