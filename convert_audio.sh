#!/bin/bash
for filename in audio/*.mp3; do
    ffmpeg -i "$filename" -ar 24000 "audio/converted/$(basename "$filename" .mp3).wav"
done
