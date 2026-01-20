import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import gfootball.env as football_env
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import pandas as pd
import numpy as np


def create_env():

    env = football_env.create_environment(
        env_name="academy_run_to_score_with_keeper",
        render=False,
        write_video=False,
        representation="simple115v2",
        rewards="scoring,checkpoints",
    )

    return env


def objective(trial):
    # Optuna objective function for hyperparameter optimization.
    # this function will be called by Optuna to evaluate different hyperparameter configurations.

    # learning rate
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)

    # network architecture
    net_arch_type = trial.suggest_categorical("net_arch", ["small", "medium", "large"])
    if net_arch_type == "small":
        net_arch = [dict(pi=[64, 64], vf=[64, 64])]
    elif net_arch_type == "medium":
        net_arch = [dict(pi=[128, 128], vf=[128, 128])]
    else:  # large
        net_arch = [dict(pi=[256, 256], vf=[256, 256])]

    # PPO-specific hyperparameters
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    n_epochs = trial.suggest_int("n_epochs", 3, 30)

    # create environment
    env = create_env()

    # create model
    policy_kwargs = dict(net_arch=net_arch)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        policy_kwargs=policy_kwargs,
        verbose=0,
    )

    model.learn(total_timesteps=10_000)

    # final evaluation
    rewards = []
    for _ in range(10):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)

    mean_reward = np.mean(rewards)

    env.close()

    return mean_reward


def run_optimization():
    # create Optuna study
    study = optuna.create_study(
        study_name="football_ppo_optimization",
        direction="maximize",
        sampler=TPESampler(n_startup_trials=10),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )

    # number of trials to run
    n_trials = 3
    print(f"Starting Optuna optimization with {n_trials} trials...")

    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)

    # best trial
    trial = study.best_trial
    print("Mean Reward:", trial.value)
    print("Best Hyperparameters:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")

    return study


def learn(best_params):

    env = create_env()

    net_arch_type = best_params["net_arch"]
    if net_arch_type == "small":
        net_arch = [dict(pi=[64, 64], vf=[64, 64])]
    elif net_arch_type == "medium":
        net_arch = [dict(pi=[128, 128], vf=[128, 128])]
    else:
        net_arch = [dict(pi=[256, 256], vf=[256, 256])]

    policy_kwargs = dict(net_arch=net_arch)

    final_model = PPO(
        "MlpPolicy",
        env,
        learning_rate=best_params["learning_rate"],
        n_steps=best_params["n_steps"],
        batch_size=best_params["batch_size"],
        n_epochs=best_params["n_epochs"],
        policy_kwargs=policy_kwargs,
        verbose=1,
    )

    # train for full duration
    final_model.learn(total_timesteps=100_000)

    final_model.save("optuna_logs/best_model.zip")

    env.close()


def save_results(study):
    df = study.trials_dataframe()

    df.to_csv("optuna_logs/study_results.csv", index=False)


if __name__ == "__main__":

    study = run_optimization()

    save_results(study)

    learn(study.best_trial.params)
