#!/usr/bin/env python3
"""Monitor Phase 2 completion and auto-run Phase 3/4/5 pipeline when both outputs are ready."""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_completion():
    """Check if both Phase 2 runs are complete."""
    hires_done = Path("results/cross_sections_hires/summary.json").exists()
    fullboard_done = Path("results/cross_sections_fullboard/summary.json").exists()

    # Count slices to show progress
    hires_count = len(list(Path("results/cross_sections_hires").glob("density_x*.npy")))
    fullboard_count = len(list(Path("results/cross_sections_fullboard").glob("density_x*.npy")))

    status = {
        "hires": {"done": hires_done, "count": hires_count, "total": 23},
        "fullboard": {"done": fullboard_done, "count": fullboard_count, "total": 56}
    }

    return status

def format_status(status):
    """Pretty-print status."""
    h = status["hires"]
    f = status["fullboard"]
    return f"Hires: {h['count']:2d}/{h['total']} | Fullboard: {f['count']:2d}/{f['total']} | Both done: {h['done'] and f['done']}"

def main():
    os.chdir(Path(__file__).parent)

    if len(sys.argv) > 1 and sys.argv[1] == "--poll":
        # Poll mode: check repeatedly until complete, then run Phase 3/4/5
        print(f"Monitoring Phase 2 completion...")
        interval = 30  # Check every 30 seconds
        max_wait = 120 * 60  # Wait up to 2 hours
        elapsed = 0

        while elapsed < max_wait:
            status = check_completion()
            msg = f"[{elapsed//60:3d}m] {format_status(status)}"
            print(msg, flush=True)

            if status["hires"]["done"] and status["fullboard"]["done"]:
                print("\n✓ Both Phase 2 runs complete! Running Phase 3/4/5 pipeline...", flush=True)
                print("=" * 70, flush=True)
                result = subprocess.run(["bash", "run_phase3_4_5.sh"], cwd=Path(__file__).parent)
                return result.returncode

            time.sleep(interval)
            elapsed += interval

        print("✗ Timeout waiting for Phase 2 completion")
        return 1
    else:
        # Check once and report status
        status = check_completion()
        print(format_status(status))
        if status["hires"]["done"] and status["fullboard"]["done"]:
            print("\nReady to run Phase 3/4/5. Execute:")
            print("  ./run_phase3_4_5.sh")
            return 0
        else:
            return 1

if __name__ == "__main__":
    sys.exit(main())
