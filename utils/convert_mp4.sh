#!/bin/bash

# Recursively convert all .mp4 files under a directory to .mp3 alongside the originals.
set -euo pipefail

root_dir_arg="${1:-.}"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required but not found in PATH" >&2
  exit 1
fi

if [[ ! -d "$root_dir_arg" ]]; then
  echo "Directory not found: $root_dir_arg" >&2
  exit 1
fi

root_dir="$(cd "$root_dir_arg" && pwd -P)"

find "$root_dir" -type f -iname '*.mp4' -print0 |
  parallel -0 --bar --line-buffer '
    file="{}"
    mp3_file="${file%.*}.mp3"
    echo "Converting: $file -> $mp3_file" >&2
    ffmpeg -nostdin -y -i "$file" -vn -acodec libmp3lame -q:a 2 "$mp3_file"
  '
