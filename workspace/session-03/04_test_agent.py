import os
import shutil
import subprocess
import gfootball.env as football_env
from stable_baselines3 import PPO
from PIL import Image

OUT_DIR = "frames"


def cleanup():
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)


def make_video():

    output_video = "replay-001.mp4"
    counter = 0
    while os.path.exists(output_video):
        counter += 1
        output_video = f"replay-{counter:03d}.mp4"

    cmd = [
        "ffmpeg",
        "-y",
        "-r",
        "24",
        "-i",
        f"{OUT_DIR}/%05d.png",
        "-pix_fmt",
        "yuv420p",
        output_video,
    ]

    subprocess.run(cmd, check=True)

    print("Video saved to", output_video)


def main():

    env = football_env.create_environment(
        env_name="academy_empty_goal_close", render=False
    )

    obs = env.reset()

    model = PPO.load("model-ppo.zip")

    done = False
    max_steps = 500
    total_reward = 0
    t = 0

    while (not done) and (t < max_steps):

        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)

        frame = env.render(mode="rgb_array")
        if frame is not None:
            Image.fromarray(frame).save(f"{OUT_DIR}/{t:05d}.png")

        if t % 20 == 0:
            print(f"Step: {t}/{max_steps}", "Reward:", reward, "Done:", done)
        t += 1

        total_reward += reward

    print(f"Step: {t}/{max_steps}", "Reward:", reward, "Done:", done)

    env.close()

    print("Total reward:", total_reward)

    env.close()


if __name__ == "__main__":

    cleanup()

    main()

    make_video()
