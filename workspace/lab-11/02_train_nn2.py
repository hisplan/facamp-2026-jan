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

    # custom MLP policy
    # 2 layers, each with 64 nodes
    # RELU activation function
    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[64, 64])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./tensorboard/",
    )

    model.learn(total_timesteps=200_000, tb_log_name="run-nn2")

    model.save("model-nn2.zip")

    env.close()


if __name__ == "__main__":

    main()
