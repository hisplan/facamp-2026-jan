# Lab

Tensorboard Integration

Reference: https://stable-baselines3.readthedocs.io/en/sde/guide/tensorboard.html

## Setup

```bash
apt-get install screen
pip install tensorboard
```

## Run Tensorboard

```bash
screen -S tensorboard bash -l
tensorboard --logdir ./tensorboard/ --bind_all
```

Open http://localhost:6006/ in your browser.

Press CTRL+A then D to detach from the screen session.
