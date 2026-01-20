import sys

sys.path.append("..")
import gfootball.env as football_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import utils


def main():

    env = football_env.create_environment(
        env_name="5_vs_5",
        representation="simple115v2",
        render=False,
        logdir="log",
        write_goal_dumps=True,
        write_full_episode_dumps=True,
    )

    # wrap environment for VecNormalize
    env = DummyVecEnv([lambda: env])
    env = VecNormalize.load("model-ppo-vecnorm.pkl", env)
    env.training = False  # do not update stats during testing
    env.norm_reward = False  # do not normalize rewards during testing

    obs = env.reset()

    model = PPO.load("model.zip")

    done = False
    max_steps = 500
    t = 0

    while (not done) and (t < max_steps):

        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)

        frame = env.venv.envs[0].render(mode="rgb_array")
        utils.save_frame(frame, t)

        if t % 20 == 0:
            print(f"Step: {t}/{max_steps}", "Reward:", reward, "Done:", done)
        t += 1

    print(info)

    env.close()


if __name__ == "__main__":

    utils.cleanup()

    main()

    utils.make_video()
