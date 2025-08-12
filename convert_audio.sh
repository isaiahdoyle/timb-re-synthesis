#!/bin/bash

# Usage: `bash convert_audio.sh <inext> <outext> <outdir>`
#   -> will convert all files in working directory of type <inext> to type
#      <outext> and place results in <outdir>

for filename in ./*."$1"; do
    ffmpeg -i "$filename" -ar 24000 "$3/$(basename "$filename" .$1).$2"
done
