from pole_balancing.environment.environment import PoleBalancingEnvironment
from pole_balancing.environment.pygame_ui import PoleBalancingUI


def run_demo():
    env = PoleBalancingEnvironment()
    ui = PoleBalancingUI(env.dt)
    state = env.reset()

    running = True
    while running:
        action, should_quit = ui.process_events()
        if should_quit:
            break

        state, reward, done, _ = env.step(action)
        ui.render(state)

        if done:
            state = env.reset()

    ui.close()


if __name__ == "__main__":
    run_demo()