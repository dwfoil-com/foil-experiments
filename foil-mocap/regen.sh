#!/usr/bin/env bash
# Regenerates the artifacts that were stripped from git to keep this repo
# small: the downloadable MediaPipe model, and the sample dots-trace video
# that MediaPipe produces from the tracked sample input.
#
# Usage: bash regen.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

MODEL_URL="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
MODEL_FILE="pose_landmarker_full.task"

if [ ! -f "$MODEL_FILE" ]; then
  echo "Downloading MediaPipe pose landmarker model (~9MB)..."
  curl -L -o "$MODEL_FILE" "$MODEL_URL"
fi

echo "Regenerating samples/output/slowmo_pump_dots_trace.mp4 ..."
python3 process_video.py samples/input/slowmo_pump.webm \
  --model "$MODEL_FILE" \
  --output samples/output

# process_video.py also writes a slowmo_pump_skeleton.mp4 byproduct — it
# isn't tracked in git (it's just a debug view), so leave/ignore it.
echo "Done."
