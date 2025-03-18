# Cart-Pole System Dynamics

## 1. System Description
The cart-pole system consists of:
- A cart of mass $M$ moving horizontally on a frictionless track.
- A pendulum (mass $m$) attached by a massless rod of length $L$ pivoting around a point on the cart.
- An external horizontal force $F_{\text{ext}}$ applied to the cart.

## 2. Coordinate System
We define two generalized coordinates:
- $x$: horizontal displacement of the cart.
- $\theta$: angular displacement of the pendulum from the vertical upright position ($\theta = 0$). Positive $\theta$ means the pole rotates clockwise.

## 3. Position & Velocity of the Pendulum Mass
The position of the pendulum mass (relative to the cart position) is:

$$
x_m = x + L\sin\theta, \quad y_m = L\cos\theta
$$

Taking derivatives to obtain velocities:

$$
\dot{x}_m = \dot{x} + L\dot{\theta}\cos\theta, \quad \dot{y}_m = -L\dot{\theta}\sin\theta
$$

## 4. Forces and Newton's Second Law
### 4.1. Horizontal Forces on the Cart
Horizontal forces include the external force $F_{\text{ext}}$ and the horizontal reaction force from the pendulum (tension $T$):

$$
M\ddot{x} = F_{\text{ext}} - T\cos\theta
$$

### 4.2. Horizontal Forces on the Pendulum Mass
The mass $m$ is horizontally accelerated by the tension $T$:

$$
m(\ddot{x} + L\ddot{\theta}\cos\theta - L\dot{\theta}^2\sin\theta) = T\cos\theta
$$

### 4.3. Vertical Forces on the Pendulum Mass
Vertically, the tension and gravity act on the mass. The vertical equation is:

$$
m(L\ddot{\theta}\sin\theta + L\dot{\theta}^2\cos\theta) = T\sin\theta - mg
$$

We usually don't explicitly need this vertical equation since it is implicitly included in the torque equation.

## 5. Rotational Dynamics (Torque Balance)
The pendulum rotates around the pivot. The rotational form of Newton's Second Law is:

$$
\sum \tau = I\ddot{\theta}
$$

Moment of inertia about the pivot is $I = mL^2$. The torques acting are:
- Gravitational torque: $-mgL\sin\theta$
- Inertial torque from cart acceleration: $mL\ddot{x}\cos\theta$

Thus, we have:

$$
mL^2\ddot{\theta} = -mgL\sin\theta + mL\ddot{x}\cos\theta
$$

Simplifying by dividing through by $mL$:

$$
L\ddot{\theta} + \ddot{x}\cos\theta + g\sin\theta = 0
$$

## 6. Complete Equations of Motion
Combining horizontal and rotational equations, we obtain two coupled equations:

$$
(M + m)\ddot{x} + mL\ddot{\theta}\cos\theta - mL\dot{\theta}^2\sin\theta = F_{\text{ext}}
$$

$$
L\ddot{\theta} + \ddot{x}\cos\theta + g\sin\theta = 0
$$

## 7. Solving for $ \ddot{x} $ and $ \ddot{\theta} $
To compute explicit solutions, we rearrange and solve the equations simultaneously:

Solve the second equation for $\ddot{x}$:

$$
\ddot{x} = -\frac{L\ddot{\theta} + g\sin\theta}{\cos\theta}
$$

Plugging this into the first equation to eliminate $\ddot{x}$:

$$
(M + m)\left(-\frac{L\ddot{\theta} + g\sin\theta}{\cos\theta}\right) + mL\ddot{\theta}\cos\theta - mL\dot{\theta}^2\sin\theta = F_{\text{ext}}
$$

Solve explicitly for $\ddot{\theta}$:

$$
\ddot{\theta} = \frac{-g\sin\theta(M + m) - \cos\theta\left(F_{\text{ext}} + mL\dot{\theta}^2\sin\theta\right)}{L\left(M + m\sin^2\theta\right)}
$$

Then substitute back into $\ddot{x}$:

$$
\ddot{x} = \frac{F_{\text{ext}} + mL(\dot{\theta}^2\sin\theta - \ddot{\theta}\cos\theta)}{M + m}
$$

## 8. Integrating the Dynamics
These equations give the second derivatives $\ddot{x}$ and $\ddot{\theta}$. To find the position $x$, angle $\theta$, and their velocities, we numerically integrate using methods like Euler's or Runge-Kutta:

- Initial conditions $x(0), \dot{x}(0), \theta(0), \dot{\theta}(0)$ are required.
- At each time step:
  - Compute $\ddot{x}(t)$ and $\ddot{\theta}(t)$ from the equations.
  - Integrate to find velocities $\dot{x}(t), \dot{\theta}(t)$.
  - Integrate velocities to update positions $x(t), \theta(t)$.

## 9. Final Summary
- **Cart position:** $x$, $\dot{x}$, $\ddot{x}$
- **Pole angle:** $\theta$, $\dot{\theta}$, $\ddot{\theta}$

This fully defines the dynamics of the cart-pole system and provides the basis for simulation or control implementations.
