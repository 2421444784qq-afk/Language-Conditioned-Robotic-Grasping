"""Record a real MuJoCo language-routed grasp as an animated GIF.

This is a visualization wrapper around the same perception/routing and staged
expert-control logic used by the public evaluation. It does not change the
reported evaluation scripts or their success criterion.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Put this repository first even when ROS has already added its own `scripts`
# package to sys.path. This avoids importing /opt/ros/.../scripts by mistake.
root_str = str(ROOT)
if root_str in sys.path:
    sys.path.remove(root_str)
sys.path.insert(0, root_str)

from envs.custom_pick_place_env import PandaPickPlaceGymEnv

# Load the sibling evaluation module by file path. ROS 2 Humble installs its own
# top-level package named `scripts`, so importing `scripts.evaluate_...` can
# accidentally resolve to /opt/ros/.../scripts instead of this repository.
import importlib.util
_eval_path = ROOT / "scripts" / "evaluate_random_positions.py"
_spec = importlib.util.spec_from_file_location("vla_evaluate_random_positions", _eval_path)
_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval)
detect = _eval.detect
route = _eval.route

XML = ROOT / "envs/assets/scene_two_graspable_blocks.xml"
PROMPTS = ("left", "right", "purple", "blue")
LAYOUTS = {"default": (-0.10, 0.10), "swapped": (0.10, -0.10)}
MAX_DELTA = 0.002
XY_TOL = 0.004
Z_TOL = 0.004
GRASP_OFFSET_Z = 0.025
LIFT_HEIGHT = 0.06
OPEN_COMMAND = -1.0
CLOSE_COMMAND = 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", choices=PROMPTS, default="left")
    ap.add_argument("--layout", choices=LAYOUTS, default="default")
    ap.add_argument("--seed", type=int, default=20261025)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--show", action="store_true", help="Show the front-camera frames in an OpenCV window.")
    args = ap.parse_args()

    out = args.output or (ROOT / "media" / f"demo_{args.prompt}_{args.layout}.gif")
    out.parent.mkdir(parents=True, exist_ok=True)
    py, by = LAYOUTS[args.layout]
    env = PandaPickPlaceGymEnv(xml_path=XML, image_obs=True, render_mode="rgb_array", reward_type="dense", random_block_position=False, random_target_position=False, seed=args.seed)
    frames = []

    def capture(repeat=1):
        frame = np.asarray(env._compute_observation()["pixels"]["front"]).copy()
        for _ in range(repeat):
            frames.append(frame)
        if args.show:
            # OpenCV in some robotics/WSL environments is built without HighGUI.
            # Recording does not depend on GUI display, so degrade gracefully.
            try:
                cv2.imshow("Language-Conditioned Robotic Grasping", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
                time.sleep(1.0 / max(args.fps, 1))
            except cv2.error as exc:
                print("WARNING: OpenCV GUI unavailable; continuing GIF recording without live window.")
                print("GUI error:", str(exc).splitlines()[0])
                args.show = False

    try:
        env.reset(seed=args.seed)
        m, d = env._model, env._data
        d.jnt("block").qpos[:3] = (0.5, py, env._block_z)
        d.jnt("blue_block").qpos[:3] = (0.5, by, env._block_z)
        mujoco.mj_forward(m, d)
        capture(args.fps)

        purple, blue = detect(env)
        selected = route(args.prompt, purple, blue)
        if not purple or not blue or selected is None:
            raise RuntimeError("Object detection/routing failed")
        print("prompt:", args.prompt, "layout:", args.layout)
        print("detected purple:", purple, "blue:", blue, "selected:", selected)

        body_name = "block" if selected == "purple" else "blue_block"
        body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
        pinch_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "pinch")
        obj = lambda: d.xpos[body_id].copy()
        eef = lambda: d.site_xpos[pinch_id].copy()

        def move_to(target, gcmd, max_steps):
            target = np.asarray(target, float)
            for step in range(max_steps):
                err = target - d.mocap_pos[0].copy()
                if np.linalg.norm(err[:2]) < XY_TOL and abs(err[2]) < Z_TOL:
                    ae = target - eef()
                    if np.linalg.norm(ae[:2]) < 0.01 and abs(ae[2]) < 0.01:
                        capture(3)
                        return True
                a = np.zeros(7, np.float32)
                a[:3] = np.clip(err, -MAX_DELTA, MAX_DELTA)
                a[6] = gcmd
                _, _, term, trunc, _ = env.step(a)
                if step % 2 == 0:
                    capture()
                if term or trunc:
                    return False
            return False

        initial = obj().copy()
        if not move_to([initial[0], initial[1], d.mocap_pos[0][2]], OPEN_COMMAND, 220):
            raise RuntimeError("ALIGN_FAIL")
        if not move_to([initial[0], initial[1], initial[2] + GRASP_OFFSET_Z], OPEN_COMMAND, 150):
            raise RuntimeError("DESCEND_FAIL")
        for _ in range(3):
            a = np.zeros(7, np.float32); a[6] = CLOSE_COMMAND
            env.step(a); capture(2)
        lift_target = eef().copy(); lift_target[2] += LIFT_HEIGHT
        if not move_to(lift_target, 0.0, 150):
            raise RuntimeError("LIFT_MOVE_FAIL")
        capture(args.fps)

        lift = float(obj()[2] - initial[2])
        ok = lift > 0.03
        print("lift_m:", round(lift, 5), "status:", "PASS" if ok else "NO_LIFT")
        imageio.mimsave(out, frames, duration=1.0 / max(args.fps, 1), loop=0)
        print("GIF:", out, "frames:", len(frames))
        if not ok:
            raise SystemExit(1)
    finally:
        if args.show:
            cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    main()
