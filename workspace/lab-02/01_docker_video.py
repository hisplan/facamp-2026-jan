import os
import numpy as np
from PIL import Image
import gfootball.env as football_env

OUT_DIR = "frames"

os.makedirs(OUT_DIR, exist_ok=False)

env = football_env.create_environment(env_name="11_vs_11_stochastic")

obs = env.reset()

done = False
max_steps = 100
t = 0

while (not done) and (t < max_steps):

    action = env.action_space.sample()  # random play
    obs, reward, done, info = env.step(action)

    frame_bgr = env.render(mode="rgb_array")
    if frame_bgr is not None:
        # BGR to RGB conversion
        frame_rgb = frame_bgr[..., ::-1]
        Image.fromarray(frame_rgb).save(f"{OUT_DIR}/{t:05d}.png")

    if t % 10 == 0:
        print("Step", t)
    t += 1

env.close()

print("Finished", t)
