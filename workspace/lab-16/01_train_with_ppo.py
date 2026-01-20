import sys

sys.path.append("..")
import gfootball.env as football_env
from stable_baselines3 import PPO
import utils


def main():

    seed = 316
    utils.seed_everything(seed)

    env = football_env.create_environment(
        env_name="academy_run_to_score_with_keeper",
        representation="simple115v2",
        render=False,
        rewards="scoring,checkpoints",
    )
    env.seed(seed)

    obs = env.reset()

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard/")

    model.learn(total_timesteps=500_000, tb_log_name="run-without-vecnorm")

    model.save("model-ppo.zip")

    env.close()


if __name__ == "__main__":

    main()
