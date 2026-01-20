import gfootball.env as football_env
from stable_baselines3 import PPO


def main():
    env = football_env.create_environment(
        env_name="academy_run_to_score",
        representation="simple115v2",
        render=False,
        rewards="scoring",
    )

    obs = env.reset()

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard/")

    model.learn(total_timesteps=50_000, tb_log_name="my-first-monitor")

    model.save("model-ppo.zip")

    env.close()


if __name__ == "__main__":

    main()
