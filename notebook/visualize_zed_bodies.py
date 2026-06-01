"""
Standalone Rerun viewer for ZED BODY_38 detections.

Reads zed_floor.json + zed_bodies.npz produced by extract_zed_floor.py.
Logs the floor plane, camera pose, per-frame 3D skeletons (color per
tracking ID) in world coords, and optionally a 2D view with the source
video and the ZED 2D keypoints overlaid.

Usage:
    python visualize_zed_bodies.py \
        --in-dir ~/data/vanRaam/zed1434recentered \
        --video ~/data/vanRaam/1434.mp4 \
        --save ~/data/vanRaam/zed1434recentered/viz_zed.rrd
"""

from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from tqdm import tqdm


# ZED SDK BODY_38 keypoint indices.
# 0 PELVIS  1 SPINE_1  2 SPINE_2  3 SPINE_3  4 NECK
# 5 NOSE    6 LEFT_EYE  7 RIGHT_EYE  8 LEFT_EAR  9 RIGHT_EAR
# 10 LEFT_CLAVICLE  11 RIGHT_CLAVICLE
# 12 LEFT_SHOULDER  13 RIGHT_SHOULDER
# 14 LEFT_ELBOW     15 RIGHT_ELBOW
# 16 LEFT_WRIST     17 RIGHT_WRIST
# 18 LEFT_HIP       19 RIGHT_HIP
# 20 LEFT_KNEE      21 RIGHT_KNEE
# 22 LEFT_ANKLE     23 RIGHT_ANKLE
# 24 LEFT_BIG_TOE   25 RIGHT_BIG_TOE
# 26 LEFT_SMALL_TOE 27 RIGHT_SMALL_TOE
# 28 LEFT_HEEL      29 RIGHT_HEEL
# 30 LEFT_HAND_THUMB_4    31 RIGHT_HAND_THUMB_4
# 32 LEFT_HAND_INDEX_1    33 RIGHT_HAND_INDEX_1
# 34 LEFT_HAND_MIDDLE_4   35 RIGHT_HAND_MIDDLE_4
# 36 LEFT_HAND_PINKY_1    37 RIGHT_HAND_PINKY_1
BODY_38_BONES: list[tuple[int, int]] = [
    # spine + head
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
    (5, 6), (5, 7), (6, 8), (7, 9),
    # left arm + hand
    (4, 10), (10, 12), (12, 14), (14, 16),
    (16, 30), (16, 32), (16, 34), (16, 36),
    # right arm + hand
    (4, 11), (11, 13), (13, 15), (15, 17),
    (17, 31), (17, 33), (17, 35), (17, 37),
    # left leg + foot
    (0, 18), (18, 20), (20, 22),
    (22, 24), (22, 26), (22, 28),
    # right leg + foot
    (0, 19), (19, 21), (21, 23),
    (23, 25), (23, 27), (23, 29),
]


def color_for(idx: int) -> tuple[int, int, int]:
    h = (idx * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def build_blueprint(has_video: bool) -> rrb.Blueprint:
    if has_video:
        return rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial2DView(name="Camera", origin="/world/zed_camera/image"),
                rrb.Spatial3DView(name="World", origin="/world"),
            ),
            collapse_panels=False,
        )
    return rrb.Blueprint(
        rrb.Spatial3DView(name="World", origin="/world"),
        collapse_panels=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, required=True,
                    help="dir containing zed_floor.json and zed_bodies.npz")
    ap.add_argument("--video", type=Path, default=None,
                    help="optional source video for the 2D view")
    ap.add_argument("--save", type=Path, default=None,
                    help="write to this .rrd instead of spawning the viewer")
    args = ap.parse_args()

    floor = json.loads((args.in_dir / "zed_floor.json").read_text())
    z = np.load(args.in_dir / "zed_bodies.npz", allow_pickle=False)

    fps = float(z["fps"])
    # Use hardcoded bones for stability; fall back to whatever is in the npz
    # only if it's already populated.
    bones_npz = z["bones"]
    bones = [(int(a), int(b)) for a, b in bones_npz] if len(bones_npz) else BODY_38_BONES
    print(f"[zed] {len(z['frame_indices'])} detections, fps={fps:.2f}, "
          f"{len(bones)} skeleton bones")

    rr.init("zed-bodies-viewer", spawn=args.save is None)
    if args.save:
        rr.save(str(args.save))

    rr.send_blueprint(build_blueprint(has_video=args.video is not None))

    # World convention: Z up, floor on XY plane.
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # AnnotationContext: one class per tracking ID with a stable color.
    NUM_CLASSES = 32
    NUM_KP = 38
    kp_annotations = [
        rr.AnnotationInfo(id=k, label=f"kp{k}") for k in range(NUM_KP)
    ]
    class_descriptions = [
        rr.ClassDescription(
            info=rr.AnnotationInfo(id=cls_id, label=f"body_{cls_id}",
                                   color=color_for(cls_id)),
            keypoint_annotations=kp_annotations,
            keypoint_connections=bones,
        )
        for cls_id in range(1, NUM_CLASSES + 1)
    ]
    rr.log("/", rr.AnnotationContext(class_descriptions), static=True)

    # Floor plate and scene bounds.
    cam_h = float(floor["translation"][2])
    box_half = max(3.0, cam_h + 1.0)
    rr.log(
        "/world/floor",
        rr.Boxes3D(
            centers=[[0.0, 0.0, 0.0]],
            half_sizes=[[box_half, box_half, 0.002]],
            colors=[[60, 90, 60]],
        ),
        static=True,
    )
    rr.log(
        "/world/scene_bounds",
        rr.Boxes3D(
            centers=[[0.0, 0.0, box_half / 2.0]],
            half_sizes=[[box_half, box_half, box_half / 2.0]],
            colors=[[80, 80, 80]],
        ),
        static=True,
    )

    # Camera placement. Use Transform3D + Pinhole, with the Pinhole's image
    # plane at /world/zed_camera/image so the 2D view binds correctly.
    R = np.asarray(floor["rotation"], dtype=np.float64)
    t = np.asarray(floor["translation"], dtype=np.float64)
    intr = floor["intrinsics"]
    rr.log(
        "/world/zed_camera",
        rr.Transform3D(translation=t.tolist(), mat3x3=R.tolist()),
        static=True,
    )
    # Note: Rerun's Pinhole archetype assumes the camera looks along its
    # *local* +Z. ZED RIGHT_HANDED_Z_UP has +Y as the looking direction, so
    # we need a small extra rotation that maps "image-Z" -> "ZED +Y".
    # Trick: place the Pinhole one level deeper with that extra Transform3D.
    # X stays X (right). Image +Y is "down" in image space, which for ZED
    # camera should be -Z (since +Z is up). Image +Z is "forward" = ZED +Y.
    # So R_imgFrame_to_zedCam =
    #   image_X -> +X
    #   image_Y -> -Z
    #   image_Z -> +Y
    # As a matrix (columns are image-axes expressed in ZED-cam frame):
    R_zedcam_from_pinhole = np.array([
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
        [0.0, -1.0, 0.0],
    ])
    rr.log(
        "/world/zed_camera/image",
        rr.Transform3D(mat3x3=R_zedcam_from_pinhole.tolist()),
        static=True,
    )
    rr.log(
        "/world/zed_camera/image",
        rr.Pinhole(
            focal_length=float(intr["fx"]),
            width=int(intr["image_width"]),
            height=int(intr["image_height"]),
        ),
        static=True,
    )

    # Per-frame logging.
    frame_indices = z["frame_indices"]
    body_ids = z["body_ids"]
    kp3d_world = z["keypoints_3d_world"]
    kp2d_all = z["keypoints_2d"]
    confidence = z["keypoint_confidence"]
    bboxes_2d = z["bboxes_2d"]

    order = np.argsort(frame_indices, kind="stable")
    frame_indices = frame_indices[order]
    body_ids = body_ids[order]
    kp3d_world = kp3d_world[order]
    kp2d_all = kp2d_all[order]
    confidence = confidence[order]
    bboxes_2d = bboxes_2d[order]

    unique_frames = np.unique(frame_indices)
    max_frame = int(unique_frames.max())

    cap = None
    if args.video is not None:
        cap = cv2.VideoCapture(str(args.video))
        if not cap.isOpened():
            print(f"[warn] cannot open {args.video}, 2D view will be empty")
            cap = None

    next_video_idx = 0
    for fidx in tqdm(range(max_frame + 1), desc="logging zed frames", unit="frame"):
        rr.set_time("frame", sequence=fidx)

        # Video frame in 2D view.
        if cap is not None:
            while next_video_idx <= fidx:
                ok, bgr = cap.read()
                if not ok:
                    cap.release(); cap = None; break
                if next_video_idx == fidx:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    ok2, buf = cv2.imencode(".jpg", bgr,
                                            [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok2:
                        rr.log(
                            "/world/zed_camera/image",
                            rr.EncodedImage(contents=buf.tobytes(),
                                            media_type="image/jpeg"),
                        )
                next_video_idx += 1

        # Clear previous per-person entities (some may disappear).
        rr.log("/world/zed_bodies", rr.Clear(recursive=True))
        rr.log("/world/zed_camera/image/zed_bodies", rr.Clear(recursive=True))

        mask = frame_indices == fidx
        rows = np.flatnonzero(mask)
        # If track IDs collide (e.g., all -1 in --no-tracking mode), fall back
        # to per-frame indexing so each body gets a distinct entity path.
        ids_in_frame = body_ids[rows]
        ids_collide = len(ids_in_frame) > 0 and len(np.unique(ids_in_frame)) != len(ids_in_frame)
        for k, row in enumerate(rows):
            bid_raw = int(body_ids[row])
            entity_idx = k if ids_collide or bid_raw < 0 else bid_raw
            cls_id = (entity_idx % NUM_CLASSES) + 1
            kp3 = kp3d_world[row]
            kp2 = kp2d_all[row]
            conf = confidence[row]
            valid = conf > 0.1
            if not valid.any():
                continue

            kp_ids = np.flatnonzero(valid).astype(np.int32)
            rr.log(
                f"/world/zed_bodies/body_{entity_idx:03d}",
                rr.Points3D(
                    positions=kp3[valid].astype(np.float32),
                    class_ids=cls_id,
                    keypoint_ids=kp_ids,
                    radii=0.02,
                    show_labels=False,
                ),
            )
            rr.log(
                f"/world/zed_camera/image/zed_bodies/body_{entity_idx:03d}",
                rr.Points2D(
                    positions=kp2[valid].astype(np.float32),
                    class_ids=cls_id,
                    keypoint_ids=kp_ids,
                    radii=3.0,
                    show_labels=False,
                ),
            )
            bb = bboxes_2d[row]
            rr.log(
                f"/world/zed_camera/image/zed_bodies/body_{entity_idx:03d}/bbox",
                rr.Boxes2D(
                    array=bb[None, :],
                    array_format=rr.Box2DFormat.XYXY,
                    class_ids=[cls_id],
                    show_labels=False,
                ),
            )

    if cap is not None:
        cap.release()

    print(f"[done] {len(unique_frames)} frames with detections")
    try:
        rr.flush(blocking=True)
    except TypeError:
        rr.flush()
    except AttributeError:
        pass


if __name__ == "__main__":
    main()
