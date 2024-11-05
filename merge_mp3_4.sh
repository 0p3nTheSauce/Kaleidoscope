#!/bin/bash

if [ $# -lt 4 ]; then
  ffmpeg -i "$1" -i "$2" -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 "$3"
else
  echo "Usage: $0 <input_video.mp4> <input_audio.mp3> <output_with_audio.mp4>"
fi
