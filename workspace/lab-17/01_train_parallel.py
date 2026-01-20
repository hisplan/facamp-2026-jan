import gfootball.env as football_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor


def make_env():
    env = football_env.create_environment(
        env_name="5_vs_5",
        representation="simple115v2",
        rewards="scoring,checkpoints",
        render=False,
        logdir="log",
        write_goal_dumps=False,
        write_full_episode_dumps=False,
    )
    return Monitor(env)
    # return env


def main():

    n_envs = 16
    n_steps = 1024
    batch_size = n_envs * n_steps

    # each env runs in its own process
    env = make_vec_env(
        make_env,
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
        seed=316,
    )

    env = VecNormalize(
        env, norm_obs=True, norm_reward=False  # usually better for sparse goal rewards
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=316,
        n_steps=n_steps,
        batch_size=batch_size,
    )

    model.learn(total_timesteps=10_000_000)  # 4.3 hours

    model.save("model.zip")

    env.save("model-ppo-vecnorm.pkl")

    env.close()


if __name__ == "__main__":

    main()
