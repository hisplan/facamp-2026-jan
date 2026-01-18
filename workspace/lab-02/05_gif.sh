#!/bin/bash

ffmpeg -i anim-gif/sample.mp4 -vf "scale=480:-1:lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 anim-gif/sample.gif
