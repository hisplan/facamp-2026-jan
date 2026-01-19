import sys

sys.path.append("..")
import utils
import gfootball.env as football_env
from gfootball.env import football_action_set
import pygame


def my_policy(obs):

    print(obs)

    # move right
    return 0


def run_scenario():

    env = football_env.create_environment(
        env_name="academy_run_to_score_with_keeper",
        representation="simple115v2",
        render=True,
        write_video=False,
        logdir="log",
        write_goal_dumps=True,
        write_full_episode_dumps=True,
    )

    obs = env.reset()

    done = False
    max_steps = 400
    t = 0

    while (not done) and (t < max_steps):

        pygame.event.pump()

        action = my_policy(obs)
        obs, reward, done, info = env.step(action)

        frame = env.render(mode="rgb_array")
        utils.save_frame(frame, t)

        if t % 10 == 0:
            print(f"Step: {t}/{max_steps}", "Reward:", reward, "Done:", done)
        t += 1

    print(f"Step: {t}/{max_steps}", "Reward:", reward, "Done:", done)

    frame = env.render(mode="rgb_array")
    utils.save_frame(frame, t)

    env.close()

    print("Finished", t)


if __name__ == "__main__":

    pygame.init()

    utils.cleanup()

    run_scenario()

    utils.make_video()
