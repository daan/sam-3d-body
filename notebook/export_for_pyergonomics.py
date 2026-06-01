"""
Export SAM3D Body inference output to a single .npz consumable by an
importer in pyergonomics. Optionally applies the ZED-derived world
transform so 3D keypoints are in world coords (Z up, floor at z=0).

Schema of the output sam3d_export.npz:

    -- per-(frame, person) row, stacked along axis 0; N total rows --
    frame:               (N,) int64    — frame index from process_video_frames.py
    person:              (N,) int64    — per-frame index (0, 1, ... within a frame).
                                         NOT a persistent track ID; run
                                         track_persons.py first if you need that.
    bbox_xywh:           (N, 4) float32 — top-left + width/height in image px
    keypoints_2d:        (N, 70, 2) float32 — image pixel coords (MHR70 order)
    keypoints_3d:        (N, 70, 3) float32 — in WORLD coords if floor present,
                                              else in camera-RDF coords
    keypoint_confidence: (N, 70) float32 — all 1.0 (SAM3D doesn't expose per-kp conf)
    pred_cam_t:          (N, 3) float32 — SAM3D root translation in camera-RDF
    focal_length:        (N,) float32

    -- scalars / metadata --
    fps                : float
    image_width        : int
    image_height       : int
    num_frames         : int
    skeleton_name      : "mhr70"
    coordinate_system_3d : "world_z_up" or "camera_rdf"
    intrinsics_fx, intrinsics_fy, intrinsics_cx, intrinsics_cy : float

    -- present only if floor was applied --
    T_world_from_rdf   : (4, 4) float64
    floor_plane        : (4,) float64    — a,b,c,d (from zed_floor.json)

Usage:
    PYTHONPATH=. python notebook/export_for_pyergonomics.py \
        --frames-dir ~/data/vanRaam/zed1434recentered \
        --fps 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_world_transform(path: Path | None):
    """Return (T_world_from_rdf 4x4, floor_dict) or (None, None)."""
    if path is None or not path.exists():
        return None, None
    data = json.loads(path.read_text())
    T = np.asarray(data["T_world_from_rdf"], dtype=np.float64)
    return T, data


def apply_T(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    return points @ R.T + t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", type=Path, required=True,
                    help="output_dir from process_video_frames.py")
    ap.add_argument("--floor", type=Path, default=None,
                    help="zed_floor.json; defaults to <frames-dir>/zed_floor.json")
    ap.add_argument("--out", type=Path, default=None,
                    help="output .npz; defaults to <frames-dir>/sam3d_export.npz")
    ap.add_argument("--fps", type=float, default=None,
                    help="frames per second (recorded if given; some downstream "
                         "consumers need this)")
    args = ap.parse_args()

    floor_path = args.floor
    if floor_path is None:
        candidate = args.frames_dir / "zed_floor.json"
        if candidate.exists():
            floor_path = candidate
    T_world_from_rdf, floor_data = load_world_transform(floor_path)
    if T_world_from_rdf is not None:
        print(f"[export] applying world transform from {floor_path}")
    else:
        print("[export] no floor data — keypoints_3d stays in camera-RDF coords")

    npz_paths = sorted((args.frames_dir / "frames").glob("*.npz"))
    if not npz_paths:
        raise SystemExit(f"no per-frame npz files in {args.frames_dir / 'frames'}")

    frame_list: list[int] = []
    person_list: list[int] = []
    bbox_xywh: list[np.ndarray] = []
    kp2d_all: list[np.ndarray] = []
    kp3d_all: list[np.ndarray] = []
    cam_t_all: list[np.ndarray] = []
    focal_all: list[float] = []

    image_hw: tuple[int, int] | None = None

    for p in npz_paths:
        z = np.load(p, allow_pickle=True)
        fidx = int(z["frame_idx"])
        if image_hw is None:
            image_hw = (int(z["image_hw"][0]), int(z["image_hw"][1]))
        n = int(z["num_persons"])
        for i in range(n):
            keys = z[f"p{i}__keys"]
            person = {str(k): z[f"p{i}__{k}"] for k in keys}

            bb = np.asarray(person["bbox"], dtype=np.float32).ravel()[:4]
            x, y, x2, y2 = bb
            bbox_xywh.append(np.array([x, y, x2 - x, y2 - y], dtype=np.float32))

            kp2 = np.asarray(person["pred_keypoints_2d"], dtype=np.float32)
            kp2d_all.append(kp2)

            kp3_rdf = (np.asarray(person["pred_keypoints_3d"], dtype=np.float32)
                       + np.asarray(person["pred_cam_t"], dtype=np.float32).ravel())
            if T_world_from_rdf is not None:
                kp3 = apply_T(T_world_from_rdf, kp3_rdf).astype(np.float32)
            else:
                kp3 = kp3_rdf.astype(np.float32)
            kp3d_all.append(kp3)

            cam_t_all.append(np.asarray(person["pred_cam_t"], dtype=np.float32).ravel())
            focal_all.append(float(np.asarray(person["focal_length"]).ravel()[0]))

            frame_list.append(fidx)
            person_list.append(i)

    n_rows = len(frame_list)
    if n_rows == 0:
        raise SystemExit("no detections found in any frame")

    kp2d = np.stack(kp2d_all, axis=0)            # (N, 70, 2)
    kp3d = np.stack(kp3d_all, axis=0)            # (N, 70, 3)
    n_kp = kp3d.shape[1]
    confidence = np.ones((n_rows, n_kp), dtype=np.float32)

    H, W = image_hw if image_hw is not None else (0, 0)
    fx_uniq = float(np.unique(focal_all)[0]) if len(np.unique(focal_all)) == 1 else float(np.mean(focal_all))
    # We don't have cx/cy here unless from the floor file. Pull them if present.
    if floor_data is not None and "intrinsics" in floor_data:
        intr = floor_data["intrinsics"]
        fx_i = float(intr["fx"])
        fy_i = float(intr["fy"])
        cx_i = float(intr["cx"])
        cy_i = float(intr["cy"])
    else:
        fx_i = fx_uniq
        fy_i = fx_uniq
        cx_i = W / 2.0
        cy_i = H / 2.0

    arrays = {
        "frame":               np.asarray(frame_list, dtype=np.int64),
        "person":              np.asarray(person_list, dtype=np.int64),
        "bbox_xywh":           np.stack(bbox_xywh, axis=0).astype(np.float32),
        "keypoints_2d":        kp2d,
        "keypoints_3d":        kp3d,
        "keypoint_confidence": confidence,
        "pred_cam_t":          np.stack(cam_t_all, axis=0).astype(np.float32),
        "focal_length":        np.asarray(focal_all, dtype=np.float32),
        "fps":                 np.asarray(args.fps if args.fps else 0.0, dtype=np.float32),
        "image_width":         np.asarray(W, dtype=np.int64),
        "image_height":        np.asarray(H, dtype=np.int64),
        "num_frames":          np.asarray(int(np.max(frame_list)) + 1, dtype=np.int64),
        "skeleton_name":       np.asarray("mhr70"),
        "coordinate_system_3d": np.asarray(
            "world_z_up" if T_world_from_rdf is not None else "camera_rdf"),
        "intrinsics_fx":       np.asarray(fx_i, dtype=np.float32),
        "intrinsics_fy":       np.asarray(fy_i, dtype=np.float32),
        "intrinsics_cx":       np.asarray(cx_i, dtype=np.float32),
        "intrinsics_cy":       np.asarray(cy_i, dtype=np.float32),
    }

    if T_world_from_rdf is not None:
        arrays["T_world_from_rdf"] = T_world_from_rdf.astype(np.float64)
        if floor_data and "floor_plane" in floor_data:
            arrays["floor_plane"] = np.asarray(floor_data["floor_plane"], dtype=np.float64)

    out_path = args.out or (args.frames_dir / "sam3d_export.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)

    print(f"[ok] wrote {out_path}")
    print(f"    {n_rows} detections across {arrays['num_frames']} frames")
    print(f"    image: {W}x{H}, skeleton: mhr70 ({n_kp} keypoints)")
    print(f"    coords: {arrays['coordinate_system_3d']}")


if __name__ == "__main__":
    main()
