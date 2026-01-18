import gfootball.env as football_env
import pygame

pygame.init()

env = football_env.create_environment(
    env_name="11_vs_11_stochastic",
    render=True,
    write_video=True,
    write_goal_dumps=True,
    write_full_episode_dumps=True,
    logdir="log"
)

env.reset()

done = False
max_steps = 50
t = 0

while (not done) and (t < max_steps):

    pygame.event.pump()  # handle pygame events to keep the window responsive

    action = env.action_space.sample()  # random play
    obs, reward, done, info = env.step(action)

    if t % 10 == 0:
        print("Step", t)
    t += 1

env.close()

print("Finished", t)
