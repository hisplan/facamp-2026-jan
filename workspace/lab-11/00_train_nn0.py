import gfootball.env as football_env
from stable_baselines3 import PPO
import torch


def main():

    env = football_env.create_environment(
        env_name="academy_run_to_score",
        representation="simple115v2",
        rewards="scoring,checkpoints",
    )

    env.reset()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./tensorboard/",
    )

    model.learn(total_timesteps=200_000, tb_log_name="run-nn0")

    model.save("model-nn0.zip")

    env.close()


if __name__ == "__main__":

    main()
