import sys

sys.path.append("..")
import gfootball.env as football_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import utils


def main():

    seed = 316
    utils.seed_everything(seed)

    # create base environment
    base_env = football_env.create_environment(
        env_name="academy_run_to_score_with_keeper",
        representation="simple115v2",
        render=False,
        rewards="scoring,checkpoints",
    )
    base_env.seed(seed)

    # wrap with Monitor to track episode statistics
    base_env = Monitor(base_env)

    # wrap in DummyVecEnv (required for VecNormalize)
    env = DummyVecEnv([lambda: base_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    obs = env.reset()

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard/")

    model.learn(total_timesteps=500_000, tb_log_name="run-with-vecnorm")

    model.save("model-ppo-vecnorm.zip")

    env.save("model-ppo-vecnorm.pkl")

    env.close()


if __name__ == "__main__":

    main()
