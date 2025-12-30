import gfootball.env as football_env
from stable_baselines3 import PPO


def main():
    env = football_env.create_environment(
        env_name="academy_empty_goal_close", render=False
    )

    obs = env.reset()

    model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, gamma=0.99)

    model.learn(total_timesteps=100)

    model.save("model-ppo.zip")

    env.close()


if __name__ == "__main__":

    main()
