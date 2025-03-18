import math
import numpy as np


class PoleBalancingEnvironment:
    def __init__(
        self, gravity: float = 9.8, cart_mass: float = 10.0, pole_mass: float = 0.001,
        pole_length: float = 15, force_magnitude: float = 30.0, dt: float = 0.002
    ):
        # Physics parameters
        self.gravity = gravity
        self.cart_mass = cart_mass
        self.pole_mass = pole_mass
        self.pole_length = pole_length
        self.force_magnitude = force_magnitude
        self.dt = dt

        # State variables [x, x_dot, theta, theta_dot]
        self.state = None

        # Limits
        self.x_threshold = 2000
        self.theta_threshold = 90 * 2 * math.pi / 360

    def reset(self):
        self.state = np.array([
            0,  # x
            0,  # x_dot
            np.random.uniform(-0.05, 0.05),  # theta
            0   # theta_dot
        ])
        return self.state.copy()

    def step(self, action):
        # Convert action to force direction
        force = 0
        if action == 1:  # Right force only when action is 1
            force = self.force_magnitude
        elif action == 2:  # Left force only when action is 2
            force = -self.force_magnitude

        # Deconstruct state variables
        x, x_dot, theta, theta_dot = self.state
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        # System parameters
        total_mass = self.cart_mass + self.pole_mass
        theta_acc = (
            force * cos_theta
            + self.pole_mass * self.pole_length * theta_dot**2 * sin_theta * cos_theta
            + total_mass * self.gravity * sin_theta
        ) / (self.pole_length * (self.cart_mass + self.pole_mass * (sin_theta**2)))

        x_acc = (
            force
            + self.pole_mass * self.pole_length * (theta_dot**2 * sin_theta - theta_acc * cos_theta)
        ) / total_mass

        # Euler integration
        x = x + self.dt * x_dot
        x_dot = x_dot + self.dt * x_acc
        theta = theta + self.dt * theta_dot
        theta_dot = theta_dot + self.dt * theta_acc

        self.state = np.array([x, x_dot, theta, theta_dot])

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold
            or theta > self.theta_threshold
        )

        reward = 1.0 if not done else 0.0

        return self.state.copy(), reward, done, {}
