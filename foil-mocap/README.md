# Foil MoCap

Extract body pose and biomechanical features from pump foil videos using MediaPipe.

Given a video of someone pump foiling, this tool produces:
- **Skeleton overlay** on the original video
- **Dots + trace** visualization on black background (great for analyzing technique)
- **Raw pose data** (.npy) and **extracted features** (.json)

`samples/output/slowmo_pump_combined.mp4` (a side-by-side original/skeleton/dots
view) is still checked in as a reference, but the current `process_video.py`
no longer has the code path that produced it — treat it as a legacy asset,
not something `regen.sh` can reproduce.

## Sample Output

| Input | Dots + Trace |
|-------|-------------|
| `samples/input/slowmo_pump.webm` | `samples/output/slowmo_pump_dots_trace.mp4` |

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

The MediaPipe model (`pose_landmarker_full.task`, ~9MB) is **not** committed —
download it and regenerate the sample output with:

```bash
bash regen.sh
```

That fetches the model from Google's MediaPipe model store and re-runs
`process_video.py` on the tracked sample input.

## Usage

### Process a video

```bash
python process_video.py samples/input/slowmo_pump.webm --model pose_landmarker_full.task
```

Note: `--model` defaults to `pose_landmarker_full.task` sitting next to the
*input video*, not next to the script — pass it explicitly (as above) unless
you've copied the model into `samples/input/`.

This produces:
- `slowmo_pump_skeleton.mp4` - skeleton overlay
- `slowmo_pump_dots_trace.mp4` - dots and trajectory traces
- `slowmo_pump_body.npy` - raw landmark positions
- `slowmo_pump_features.json` - extracted biomechanical features

### Optional: Stabilize shaky video first

```bash
python stabilize_video.py input.mp4 -o stabilized.mp4
python process_video.py stabilized.mp4
```

Uses FFmpeg's vidstab filter to remove camera shake before processing.

## Extracted Features

The features JSON includes:
- Joint angles over time (hip, knee, ankle, shoulder)
- Vertical oscillation amplitude and frequency
- Center of mass trajectory
- Timing between upper and lower body movements

## How It Works

1. **MediaPipe Pose Landmarker** detects 33 body landmarks per frame
2. Landmarks are smoothed with a Savitzky-Golay filter
3. Joint angles and body segments are computed from landmark positions
4. Visualizations are rendered frame-by-frame with trajectory traces
5. Biomechanical features are extracted from the time series
