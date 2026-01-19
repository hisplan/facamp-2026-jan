import sys

sys.path.append("..")
import gfootball.env as football_env
from stable_baselines3 import A2C
import utils


def main():

    env = football_env.create_environment(
        env_name="academy_empty_goal_close", render=False
    )

    obs = env.reset()

    model = A2C.load("model-a2c.zip")

    done = False
    max_steps = 500
    total_reward = 0
    t = 0

    while (not done) and (t < max_steps):

        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)

        frame = env.render(mode="rgb_array")
        utils.save_frame(frame, t)

        if t % 20 == 0:
            print(f"Step: {t}/{max_steps}", "Reward:", reward, "Done:", done)
        t += 1

        total_reward += reward

    print(f"Step: {t}/{max_steps}", "Reward:", reward, "Done:", done)
    print("Total reward:", total_reward)

    env.close()


if __name__ == "__main__":

    utils.cleanup()

    main()

    utils.make_video()
