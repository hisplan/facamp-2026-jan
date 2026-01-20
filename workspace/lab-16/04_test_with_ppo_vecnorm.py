import sys

sys.path.append("..")
import gfootball.env as football_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import utils


def main():

    env = football_env.create_environment(
        env_name="academy_run_to_score_with_keeper",
        representation="simple115v2",
        render=False,
        rewards="scoring,checkpoints",
    )

    # wrap environment for VecNormalize
    env = DummyVecEnv([lambda: env])
    env = VecNormalize.load("model-ppo-vecnorm.pkl", env)
    env.training = False  # do not update stats during testing
    env.norm_reward = False  # do not normalize rewards during testing

    obs = env.reset()

    model = PPO.load("model-ppo-vecnorm.zip")

    done = False
    max_steps = 1500
    total_reward = 0
    t = 0

    while (not done) and (t < max_steps):

        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)

        # get raw frame from the underlying environment
        frame = env.venv.envs[0].render(mode="rgb_array")
        utils.save_frame(frame, t)

        if t % 20 == 0:
            print(f"Step: {t}/{max_steps}", "Reward:", reward, "Done:", done)
        t += 1

        total_reward += reward[0]

    frame = env.venv.envs[0].render(mode="rgb_array")
    utils.save_frame(frame, t)

    print(f"Step: {t}/{max_steps}", "Reward:", reward, "Done:", done)
    print("Total reward:", total_reward)

    env.close()


if __name__ == "__main__":

    utils.cleanup()

    main()

    utils.make_video()
