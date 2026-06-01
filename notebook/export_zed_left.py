"""
Export the rectified LEFT-eye image of a ZED SVO as frame-synced PNGs.

This is the RGB source for the SAM3D/MHR pass. Frame numbering matches
extract_zed_floor.py exactly: both iterate grabs from set_svo_position(0)
and increment a frame counter every successful grab, so

    left_frames/NNNNNN.png  ==  grab N  ==  zed_bodies.npz row with frame N

That 1:1 correspondence is what lets us solve ZED↔MHR retargeting offsets at
the same instant (e.g. the held T-pose).

We export the *rectified* left view (sl.VIEW.LEFT): it matches the rectified
left intrinsics in zed_floor.json and the BODY_38 keypoints_2d, so SAM3D's 2D
output lines up with both. Depth mode is NONE (rectification is from
calibration, independent of depth) — image-only export is fast.

Usage:
    python export_zed_left.py \
        --svo /path/clip.svo2 \
        --out-dir ~/data/retarget_p2          # writes left_frames/NNNNNN.png

Then run SAM3D on the frames (note HD1080 off-center principal point):
    PYTHONPATH=. python process_video_frames.py \
        --frames-dir <out-dir>/left_frames \
        --output-dir <out-dir> \
        --intrinsics 1071.987 1071.987 983.639 561.045 \
        --recenter-principal-point
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pyzed.sl as sl
from tqdm import tqdm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--svo", type=Path, required=True, help="ZED SVO/SVO2 file")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="frames written to <out-dir>/left_frames/NNNNNN.<ext>")
    ap.add_argument("--subdir", default="left_frames",
                    help="subfolder name under --out-dir (default: left_frames)")
    ap.add_argument("--ext", choices=("png", "jpg"), default="png",
                    help="png (lossless, default) or jpg")
    ap.add_argument("--jpg-quality", type=int, default=95)
    ap.add_argument("--max-frames", type=int, default=None,
                    help="stop after this many grabs (default: whole SVO)")
    args = ap.parse_args()

    if not args.svo.is_file():
        raise FileNotFoundError(args.svo)

    init = sl.InitParameters()
    init.set_from_svo_file(str(args.svo))
    init.coordinate_units = sl.UNIT.METER
    # Image-only export: rectified LEFT view doesn't need depth.
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP

    zed = sl.Camera()
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"failed to open {args.svo}: {err}")

    try:
        info = zed.get_camera_information()
        lc = info.camera_configuration.calibration_parameters.left_cam
        w, h = int(lc.image_size.width), int(lc.image_size.height)
        n_total = zed.get_svo_number_of_frames()
        fps = float(info.camera_configuration.fps)
        print(f"[svo] {w}x{h} @ {fps:.2f}fps, {n_total} frames")

        frames_dir = args.out_dir / args.subdir
        frames_dir.mkdir(parents=True, exist_ok=True)

        max_frames = args.max_frames if args.max_frames is not None else n_total

        zed.set_svo_position(0)
        image = sl.Mat()
        frame_idx = 0
        written = 0
        with tqdm(total=max_frames, desc="export left", unit="frame") as pbar:
            while frame_idx < max_frames and zed.grab() == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU)
                bgra = image.get_data()              # (H, W, 4) BGRA
                bgr = np.ascontiguousarray(bgra[:, :, :3])
                out_path = frames_dir / f"{frame_idx:06d}.{args.ext}"
                if args.ext == "jpg":
                    cv2.imwrite(str(out_path), bgr,
                                [cv2.IMWRITE_JPEG_QUALITY, args.jpg_quality])
                else:
                    cv2.imwrite(str(out_path), bgr)
                written += 1
                frame_idx += 1
                pbar.update(1)
    finally:
        zed.close()

    print(f"[ok] wrote {written} frames to {frames_dir} "
          f"(00..{written - 1:06d}.{args.ext})")
    print(f"     frame N here == grab N == zed_bodies.npz frame N")


if __name__ == "__main__":
    main()
