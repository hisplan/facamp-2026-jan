import gfootball.env as football_env
from stable_baselines3 import PPO
from custom_reward import CustomReward
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
final_path = os.path.join(BASE_DIR, "model.zip")

env = football_env.create_environment(
    env_name="5_vs_5",
    render=False,
    write_video=False,
    representation="simple115v2",
    rewards="scoring",
)

env = CustomReward(env)

obs = env.reset()

model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=50_000)

model.save(final_path)

env.close()
