import gfootball.env as football_env
from stable_baselines3 import A2C


def main():
    env = football_env.create_environment(
        env_name="academy_empty_goal_close", render=False
    )

    obs = env.reset()

    model = A2C("MlpPolicy", env, verbose=1)

    model.learn(total_timesteps=100)

    model.save("model-a2c.zip")

    env.close()


if __name__ == "__main__":

    main()
