import math

import pygame


class PoleBalancingUI:
    def __init__(self, dt: float) -> None:
        """Initialize the Pole Balancing UI.

        Args:
            dt: The timestep of the simulation.

        """
        pygame.init()
        self.screen = pygame.display.set_mode((1280, 720), pygame.SCALED | pygame.DOUBLEBUF)
        self.clock = pygame.time.Clock()
        self.scale = 200  # Scale for physics to pixels
        self.screen_width = 1280
        self.screen_height = 720
        self.cart_width = 80
        self.cart_height = 50
        self.pole_width = 10
        self.dt = dt
        self.camera_x = 0  # Camera position tracking
        self.camera_follow_speed = 0.05  # Smooth camera following

        # Create track markers
        self.track_markers = []
        marker_spacing = 200
        for i in range(-50, 50):
            self.track_markers.append(i * marker_spacing)

    def process_events(self) -> tuple[int | None, bool]:
        action = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None, True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_RIGHT]:
            action = 1  # Map right key to positive (rightward) force
        elif keys[pygame.K_LEFT]:
            action = 2  # Map left key to negative (leftward) force

        return action, False

    def render(self, state: tuple[float, float, float]) -> None:
        # Update camera position with smooth follow
        target_camera_x = state[0] * self.scale
        self.camera_x += (target_camera_x - self.camera_x) * self.camera_follow_speed

        # Smooth gradient background
        self.screen.fill((30, 30, 30))
        pygame.draw.rect(
            self.screen,
            (50, 50, 80),
            (0, self.screen_height // 2, self.screen_width, self.screen_height // 2),
        )

        # Draw track with position markers
        cart_y = self.screen_height // 2 + 20
        pygame.draw.line(
            self.screen,
            (200, 200, 200),
            (0, cart_y + 25),
            (self.screen_width, cart_y + 25),
            8,
        )

        # Draw track markers to indicate movement
        for marker_pos in self.track_markers:
            screen_x = marker_pos - self.camera_x + self.screen_width / 2
            # Only draw markers that are on screen
            if 0 <= screen_x <= self.screen_width:
                # Draw marker line
                pygame.draw.line(
                    self.screen,
                    (100, 100, 150),
                    (screen_x, cart_y + 15),
                    (screen_x, cart_y + 35),
                    3,
                )
                # Draw position label for major markers
                if marker_pos % 1000 == 0:
                    font = pygame.font.SysFont("Arial", 14)
                    pos_text = font.render(
                        text=f"{marker_pos / 100:.0f}m",
                        antialias=True,
                        color=(200, 200, 200),
                    )
                    self.screen.blit(pos_text, (screen_x - 15, cart_y + 40))

        # Cart position relative to camera
        cart_x = self.screen_width / 2

        # Draw pole
        pole_length = self.scale * 0.6
        pole_end_x = cart_x + pole_length * math.sin(-state[2])
        pole_end_y = cart_y - pole_length * math.cos(-state[2])
        pygame.draw.aaline(
            self.screen,
            (220, 80, 80),
            (cart_x, cart_y),
            (pole_end_x, pole_end_y),
            self.pole_width,
        )

        # Cart with depth and highlights
        pygame.draw.rect(
            self.screen,
            (80, 80, 200),
            (
                cart_x - self.cart_width / 2,
                cart_y - self.cart_height / 2,
                self.cart_width,
                self.cart_height,
            ),
            border_radius=8,
        )
        pygame.draw.line(
            self.screen,
            (150, 150, 255),
            (cart_x - self.cart_width / 2 + 5, cart_y - self.cart_height / 2 + 5),
            (cart_x + self.cart_width / 2 - 5, cart_y - self.cart_height / 2 + 5),
            3,
        )

        # Draw position info
        font = pygame.font.SysFont("Arial", 18)
        pos_info = font.render(
            text=f"Position: {state[0]:.2f}",
            antialias=True,
            color=(255, 255, 255),
        )
        self.screen.blit(pos_info, (10, 10))

        # Camera indicator
        camera_info = font.render(
            text=f"Camera at {self.camera_x / self.scale:.2f}",
            antialias=True,
            color=(200, 200, 200),
        )
        self.screen.blit(camera_info, (10, 40))

        # Real-time shadow - based on velocity instead of position
        shadow_size = min(40, abs(state[1] * 5))  # Based on cart velocity with cap
        shadow = pygame.Surface((self.cart_width + shadow_size, 20), pygame.SRCALPHA)
        shadow.fill((0, 0, 0, 50))
        self.screen.blit(shadow, (cart_x - (self.cart_width + shadow_size) / 2, cart_y + 25))

        pygame.display.flip()
        self.clock.tick(1 / self.dt)  # Sync with simulation timestep

    def close(self) -> None:
        pygame.quit()
