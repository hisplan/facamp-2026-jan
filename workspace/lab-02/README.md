# Lab

Making Replay Video

## Method 1

This works both locally and in Docker environment.

```python
env = football_env.create_environment(env_name="11_vs_11_stochastic")

env.reset()

done = False
step = 0

while (not done):

    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

    frame_bgr = env.render(mode="rgb_array")
    if frame_bgr is not None:
        # BGR to RGB conversion
        frame_rgb = frame_bgr[..., ::-1]
        Image.fromarray(frame_rgb).save(f"frame-{step:05d}.png")

    step += 1

env.close()
```

```bash
ffmpeg -y -r 24 -i frame-%05d.png -pix_fmt yuv420p demo-random-play.mp4
```

## Method 2

⚠️ The following only works on local machine with pygame rendering support. It does not work in Docker environment.

Reference: https://github.com/google-research/football/blob/master/gfootball/doc/saving_replays.md

### Generating 3D Video

An .avi video can be created by setting the following when creating a football environment:

```
write_video=True,
write_goal_dumps=True,
write_full_episode_dumps=True,
logdir="log"
```

### From Dump to 3D Video

Replays a given trace dump using environment and generates an .avi video file.

#### Usage

```bash
python -m gfootball.replay --helpshort
Script allowing to replay a given trace file.
   Example usage:
   python replay.py --trace_file=/tmp/dumps/shutdown_20190521-165136974075.dump


flags:

/Users/chunj/miniconda3/envs/gfootball_fresh/lib/python3.10/site-packages/gfootball/replay.py:
  --fps: How many frames per second to render
    (default: '10')
    (an integer)
  --trace_file: Trace file to replay

Try --helpfull to get a list of all flags.
```

#### Example

```bash
conda activate gfootball
python -m gfootball.replay --trace_file ./log/episode_done_20260116-191623787447.dump
```

### From Dump to 2D Video

converts trace dump to a 2D representation video.

#### Usage

```bash
python -m gfootball.dump_to_video --helpshort
Script allowing to render a replay video from a game dump.
flags:

/Users/chunj/miniconda3/envs/gfootball_fresh/lib/python3.10/site-packages/gfootball/dump_to_video.py:
  --trace_file: Trace file to render

Try --helpfull to get a list of all flags.
```

#### Example

```bash
python -m gfootball.dump_to_video --trace_file ./log/episode_done_20260116-191623787447.dump
```

### From .avi to .mp4

```bash
ffmpeg -i input.avi -c:v copy output.mp4
```
