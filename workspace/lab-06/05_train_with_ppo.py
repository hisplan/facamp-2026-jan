import gfootball.env as football_env
from stable_baselines3 import PPO


def main():
    env = football_env.create_environment(
        env_name="academy_empty_goal_close", representation="simple115v2", render=False
    )

    obs = env.reset()

    model = PPO("MlpPolicy", env, verbose=1)

    model.learn(total_timesteps=10_000)

    model.save("model-ppo.zip")

    env.close()


if __name__ == "__main__":

    main()
