"""
Side-by-side visualization of SAM3D Body inference + tracker output.

Reads:
  - <frames-dir>/frames/NNNNNN.npz   from process_video_frames.py
  - <frames-dir>/tracks/track_*.npz  from track_persons.py
  - the source video

Writes an annotated mp4: left half = raw per-frame bboxes (no IDs),
right half = tracked bboxes (colored by stable track_id).

Runs in either env — only needs cv2 + numpy.
"""

import argparse
import colorsys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def color_for(track_id: int) -> tuple[int, int, int]:
    """Stable BGR color per track id."""
    h = (track_id * 0.61803398875) % 1.0   # golden-ratio hue
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def load_frame_npz(path):
    z = np.load(path, allow_pickle=True)
    n = int(z["num_persons"])
    persons = []
    for i in range(n):
        keys = z[f"p{i}__keys"]
        persons.append({str(k): z[f"p{i}__{k}"] for k in keys})
    return persons


def load_tracks(tracks_dir: Path):
    """Return {frame_idx: [(track_id, bbox_xyxy), ...]}."""
    by_frame = defaultdict(list)
    for p in sorted(tracks_dir.glob("track_*.npz")):
        z = np.load(p, allow_pickle=True)
        tid = int(z["track_id"])
        frames = z["frame_idx"]
        bboxes = z["bbox"]
        for f, bb in zip(frames, bboxes):
            b = np.asarray(bb).ravel()
            by_frame[int(f)].append((tid, b[:4]))
    return by_frame


def draw_box(img, x1, y1, x2, y2, color, label=None, thickness=2):
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(
            img,
            (int(x1), int(y1) - th - 6),
            (int(x1) + tw + 4, int(y1)),
            color,
            -1,
        )
        cv2.putText(
            img, label, (int(x1) + 2, int(y1) - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", type=Path, required=True,
                    help="output_dir from process_video_frames.py")
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="output mp4 path (default: <frames-dir>/viz.mp4)")
    ap.add_argument("--show-ids-on-raw", action="store_true",
                    help="also label raw boxes with their per-frame index")
    args = ap.parse_args()

    frames_dir = args.frames_dir / "frames"
    tracks_dir = args.frames_dir / "tracks"
    out_path = args.out or (args.frames_dir / "viz.mp4")

    npz_by_idx = {int(p.stem): p for p in frames_dir.glob("*.npz")}
    tracks_by_frame = load_tracks(tracks_dir) if tracks_dir.exists() else {}

    print(f"[viz] {len(npz_by_idx)} inference frames, "
          f"{len(tracks_by_frame)} frames with tracks")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W * 2, H))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        left = frame.copy()
        right = frame.copy()

        # Left: raw per-frame inference, all white boxes.
        npz = npz_by_idx.get(frame_idx)
        if npz is not None:
            for i, p in enumerate(load_frame_npz(npz)):
                b = np.asarray(p["bbox"]).ravel()
                label = f"#{i}" if args.show_ids_on_raw else None
                draw_box(left, *b[:4], (255, 255, 255), label=label)

        # Right: tracker output, colored by stable id.
        for tid, b in tracks_by_frame.get(frame_idx, []):
            draw_box(right, *b[:4], color_for(tid), label=f"id {tid}")

        cv2.putText(left, "raw per-frame", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(right, "tracked", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(left, f"frame {frame_idx}", (10, H - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

        writer.write(np.hstack([left, right]))
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"[viz] wrote {out_path}")


if __name__ == "__main__":
    main()
