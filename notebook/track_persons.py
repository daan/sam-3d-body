"""
Post-process per-frame SAM3D Body outputs into per-person tracks.

Reads frames/NNNNNN.npz produced by process_video_frames.py and the original
video, runs a ReID-aware tracker (BoT-SORT) over the bboxes, then writes one
.npz per stable track_id containing time-stacked pose params, cam_t,
global_rot, etc.

Identity is recovered after gaps (someone leaving frame for >1s) via an
appearance-embedding gallery, configured by --track-buffer.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from boxmot import BotSort

# Fields stacked along the time axis. Must match what process_video_frames.py
# kept (PERSON_KEYS_TO_KEEP).
TIME_VARYING_KEYS = [
    "body_pose_params", "hand_pose_params", "expr_params",
    "global_rot", "pred_global_rots",
    "pred_cam_t", "focal_length",
    "pred_keypoints_2d", "pred_keypoints_3d", "pred_joint_coords",
    "bbox", "lhand_bbox", "rhand_bbox",
]

# Treated as identity-level — averaged across the track. Move into
# TIME_VARYING_KEYS if you want per-frame shape instead.
IDENTITY_KEYS = ["shape_params", "scale_params"]


def load_frame_npz(path):
    """Return (frame_idx, image_hw, [person_dict, ...]) for one frame."""
    z = np.load(path, allow_pickle=True)
    n = int(z["num_persons"])
    persons = []
    for i in range(n):
        keys = z[f"p{i}__keys"]
        persons.append({str(k): z[f"p{i}__{k}"] for k in keys})
    return int(z["frame_idx"]), tuple(z["image_hw"]), persons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--frames-dir", type=Path, required=True,
        help="output_dir from process_video_frames.py (contains frames/, meta.json)",
    )
    ap.add_argument(
        "--video", type=Path, required=True,
        help="same video used for inference — needed for ReID appearance crops",
    )
    ap.add_argument(
        "--reid-weights", type=Path,
        default=Path("osnet_x0_25_msmt17.pt"),
        help="ReID model weights (auto-downloaded by boxmot on first use)",
    )
    ap.add_argument(
        "--track-buffer", type=int, default=60,
        help="frames to keep a 'lost' track warm for re-ID. "
             "Set to fps * max_expected_gap_seconds.",
    )
    ap.add_argument("--match-thresh", type=float, default=0.8)
    ap.add_argument("--new-track-thresh", type=float, default=0.7)
    ap.add_argument("--track-high-thresh", type=float, default=0.6)
    ap.add_argument("--min-track-len", type=int, default=10)
    ap.add_argument("--device", type=str, default="0")
    args = ap.parse_args()

    frames_dir = args.frames_dir / "frames"
    out_dir = args.frames_dir / "tracks"
    out_dir.mkdir(exist_ok=True)

    tracker = BotSort(
        reid_weights=args.reid_weights,
        device=args.device,
        half=True,
        track_high_thresh=args.track_high_thresh,
        new_track_thresh=args.new_track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
        with_reid=True,
    )

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {args.video}")

    npz_paths = sorted(frames_dir.glob("*.npz"))
    npz_by_idx = {int(p.stem): p for p in npz_paths}
    if not npz_by_idx:
        raise RuntimeError(f"no per-frame .npz files in {frames_dir}")

    # Buffered per-track results, written at the end.
    per_track = defaultdict(list)  # track_id -> [(frame_idx, person_dict), ...]

    frame_idx = 0
    last_npz_idx = max(npz_by_idx)

    while True:
        ok, frame_bgr = cap.read()
        if not ok or frame_idx > last_npz_idx:
            break

        npz = npz_by_idx.get(frame_idx)
        if npz is None:
            # Frame missing from inference output — still advance the tracker
            # so its Kalman/age counters tick.
            tracker.update(np.empty((0, 6)), frame_bgr)
            frame_idx += 1
            continue

        _, _, persons = load_frame_npz(npz)
        if not persons:
            tracker.update(np.empty((0, 6)), frame_bgr)
            frame_idx += 1
            continue

        # BoxMOT expects [x1, y1, x2, y2, conf, cls]. SAM3D Body's bbox
        # may be 4-wide or 5-wide (with score) depending on detector — handle both.
        dets = []
        for p in persons:
            b = np.asarray(p["bbox"]).ravel()
            x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
            conf = float(b[4]) if b.size >= 5 else 1.0
            dets.append([x1, y1, x2, y2, conf, 0])
        dets = np.asarray(dets, dtype=np.float32)

        # Returns [x1, y1, x2, y2, track_id, conf, cls, det_idx]
        tracks = tracker.update(dets, frame_bgr)

        for row in tracks:
            det_idx = int(row[-1])
            track_id = int(row[4])
            if 0 <= det_idx < len(persons):
                per_track[track_id].append((frame_idx, persons[det_idx]))

        frame_idx += 1

    cap.release()

    print(f"\n[tracker] found {len(per_track)} raw tracks, "
          f"writing tracks with >= {args.min_track_len} frames\n")

    kept = 0
    for tid, entries in sorted(per_track.items()):
        if len(entries) < args.min_track_len:
            continue
        entries.sort(key=lambda e: e[0])
        frames = np.array([e[0] for e in entries], dtype=np.int64)

        stacked = {"frame_idx": frames, "track_id": np.int64(tid)}
        for k in TIME_VARYING_KEYS:
            vals = [np.asarray(e[1][k]) for e in entries if k in e[1]]
            if len(vals) == len(entries):
                stacked[k] = np.stack(vals, axis=0)
            elif vals:
                # Some frames missing this key — keep what we have plus an index.
                idx = np.array(
                    [i for i, e in enumerate(entries) if k in e[1]],
                    dtype=np.int64,
                )
                stacked[k] = np.stack(vals, axis=0)
                stacked[f"{k}_frame_mask"] = idx

        for k in IDENTITY_KEYS:
            vals = [np.asarray(e[1][k]) for e in entries if k in e[1]]
            if vals:
                stacked[k] = np.mean(np.stack(vals, axis=0), axis=0)

        np.savez_compressed(out_dir / f"track_{tid:04d}.npz", **stacked)
        gaps = int((np.diff(frames) > 1).sum())
        print(f"  track {tid:4d}: {len(entries):5d} frames "
              f"[{frames.min()}..{frames.max()}], gaps={gaps}")
        kept += 1

    print(f"\n[done] wrote {kept} tracks to {out_dir}")


if __name__ == "__main__":
    main()
