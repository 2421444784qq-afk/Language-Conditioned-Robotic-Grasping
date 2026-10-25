"""Run one language-routed grasp demo in the two-object MuJoCo scene."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the tested perception/routing/control implementation.
from scripts.evaluate_random_positions import run_trial

PROMPTS = ("left", "right", "purple", "blue")
LAYOUTS = {
    "default": (-0.10, 0.10),
    "swapped": (0.10, -0.10),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", choices=PROMPTS, default="left")
    parser.add_argument("--layout", choices=LAYOUTS, default="default")
    parser.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args()

    purple_y, blue_y = LAYOUTS[args.layout]
    result = run_trial(1, purple_y, blue_y, args.prompt, args.seed)
    print("prompt:", args.prompt)
    print("layout:", args.layout, "purple_y=", purple_y, "blue_y=", blue_y)
    print("selected:", result.get("selected"), "expected:", result.get("expected"))
    print("status:", result.get("status"), "lift_m:", result.get("lift_m"))
    if result.get("status") != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
