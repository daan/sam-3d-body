"""
Compare the retargeted MHR stick figure against the ZED BODY_38 stick figure
in one rerun view (toggle each via entity visibility in the rerun UI).

MHR joints come from FK on the route-B retargeted global rotations
(retarget_mhr.npz) using the rig offsets; ZED joints are keypoints_3d_world.
Both are placed in the shared Z-up world; the MHR root is shifted onto the
ZED pelvis so the two skeletons overlay for direct pose comparison.

If the two stick figures match but the skinned mesh looked deformed, the
retarget is correct at the skeleton level and the deformation is twist/skinning
(twist was left at rest by design).

Usage (sam_3d_body env):
    python visualize_retarget_compare.py --in-dir ~/data/retarget_p2 \
        [--save ~/data/retarget_p2/retarget_compare.rrd]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rerun as rr

import retarget_zed_to_mhr as RT
from visualize_zed_bodies import BODY_38_BONES

RIG_DEFAULT = ("/home/daniel/code/dev-pyergonomics/src/pyergonomics/"
               "ui/view3d/mhr/content/mhr_skeleton.json")


def mhr_fk_positions(G: np.ndarray, offsets: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """Joint positions (native cm) from global rotations + rig offsets."""
    P = np.zeros((len(parents), 3))
    for j in range(len(parents)):
        p = int(parents[j])
        P[j] = offsets[j] if p < 0 else P[p] + G[p] @ offsets[j]
    return P


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--rig", type=Path, default=Path(RIG_DEFAULT))
    ap.add_argument("--save", type=Path, default=None)
    args = ap.parse_args()

    z = np.load(args.in_dir / "zed_bodies.npz", allow_pickle=False)
    fi = z["frame_indices"]; kpw = z["keypoints_3d_world"]; conf = z["keypoint_confidence"]
    row_of = {int(f): i for i, f in enumerate(fi)}

    r = np.load(args.in_dir / "retarget_mhr.npz", allow_pickle=False)
    G_all = r["pred_global_rots"].astype(np.float64)   # (F,127,3,3) rig-native
    frames = [int(f) for f in r["frame"]]
    rig = json.loads(args.rig.read_text())
    offsets = np.asarray(rig["offsets"], dtype=np.float64)
    parents = np.asarray(rig["parents"], dtype=np.int64)
    mhr_bones = [(int(p), j) for j, p in enumerate(parents) if p >= 0]

    rr.init("zed-vs-mhr-retarget", spawn=args.save is None)
    if args.save:
        rr.save(str(args.save))
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    # floor plate
    rr.log("/world/floor", rr.Boxes3D(centers=[[0, 0, 0]], half_sizes=[[3, 3, 0.002]],
                                      colors=[[60, 90, 60]]), static=True)

    PELVIS = RT.ZED["PELVIS"]
    F = RT.NATIVE_TO_WORLD
    for fr in frames:
        if fr not in row_of:
            continue
        rr.set_time("frame", sequence=fr)

        # ZED skeleton (world, m), blue.
        k = kpw[row_of[fr]]
        valid = conf[row_of[fr]] > 0.1
        rr.log("/world/zed",
               rr.LineStrips3D([[k[a], k[b]] for a, b in BODY_38_BONES
                                if valid[a] and valid[b]],
                               colors=[[80, 160, 255]], radii=0.008))
        rr.log("/world/zed/joints",
               rr.Points3D(k[valid], colors=[[80, 160, 255]], radii=0.018))

        # Retargeted MHR skeleton: FK -> world, shift root onto ZED pelvis. Red.
        G = G_all[frames.index(fr)]
        P = mhr_fk_positions(G, offsets, parents)          # native cm
        Pw = (F @ P.T).T * 0.01                            # -> world m, oriented
        Pw += (k[PELVIS] - Pw[RT.MHR["root"]])             # align root to ZED pelvis
        rr.log("/world/mhr",
               rr.LineStrips3D([[Pw[a], Pw[b]] for a, b in mhr_bones],
                               colors=[[255, 90, 90]], radii=0.006))
        rr.log("/world/mhr/joints",
               rr.Points3D(Pw, colors=[[255, 90, 90]], radii=0.012))

    if args.save:
        print(f"[ok] saved {args.save}")


if __name__ == "__main__":
    main()
