"""
Extract the floor plane and (optionally) BODY_38 detections from a ZED SVO.

Mirrors the conventions in pyergonomics/importers/zed.py:
- ZED camera frame:  RIGHT_HANDED_Z_UP (+X right, +Y forward, +Z up)
- World frame:       Z-up, floor on the XY plane (z=0)
- p_world = R @ p_camera_zup + t,  with t = (0, 0, camera_height)

Also pre-computes a SAM3D-Body-ready transform: SAM3D's 3D output is in the
RDF camera frame (+X right, +Y down, +Z forward), so we bake in the fixed
RDF -> ZED_ZUP rotation and provide T_world_from_rdf (4x4) for downstream use.

Outputs:
  - zed_floor.json  : floor plane, world transform, intrinsics
  - zed_bodies.npz  : BODY_38 detections per frame (if --bodies)

This script must run in an environment that has the pyzed wheel installed —
typically a different env than sam_3d_body.

Usage:
    python extract_zed_floor.py \
        --svo /path/to/clip.svo2 \
        --out-dir /path/to/output_dir \
        --bodies --max-seconds 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyzed.sl as sl
from tqdm import tqdm


def _create_floor_transform(floor_plane_eq):
    """Return (R, t) such that p_world = R @ p_camera_zup + t.

    Convention: world Z = floor normal (up), world Y = camera forward
    projected onto the floor, world X = Y × Z. Camera origin is at
    (0, 0, d) in world.
    """
    a, b, c, d = floor_plane_eq
    n = np.array([a, b, c], dtype=np.float64)
    n = n / np.linalg.norm(n)

    z_world = n
    # In ZED RIGHT_HANDED_Z_UP, camera looks along +Y.
    cam_forward = np.array([0.0, 1.0, 0.0])
    y_world = cam_forward - np.dot(cam_forward, z_world) * z_world
    y_norm = np.linalg.norm(y_world)

    if y_norm < 1e-6:
        # Camera looking straight up/down: fall back to camera right.
        cam_right = np.array([1.0, 0.0, 0.0])
        x_world = cam_right - np.dot(cam_right, z_world) * z_world
        x_world = x_world / np.linalg.norm(x_world)
        y_world = np.cross(z_world, x_world)
    else:
        y_world = y_world / y_norm
        x_world = np.cross(y_world, z_world)

    R = np.vstack([x_world, y_world, z_world])
    t = np.array([0.0, 0.0, float(d)])
    return R, t


# Fixed change-of-basis: SAM3D Body (RDF) -> ZED camera (RIGHT_HANDED_Z_UP).
# RDF: +X right, +Y down,    +Z forward
# ZUP: +X right, +Y forward, +Z up
R_ZUP_FROM_RDF = np.array([
    [1.0,  0.0, 0.0],
    [0.0,  0.0, 1.0],
    [0.0, -1.0, 0.0],
], dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--svo", type=Path, required=True, help="ZED SVO/SVO2 file")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="output dir; writes zed_floor.json (and zed_bodies.npz if --bodies)")
    ap.add_argument("--warmup-frames", type=int, default=30,
                    help="frames to grab before find_floor_plane (helps stability)")
    ap.add_argument("--bodies", action="store_true",
                    help="also extract BODY_38 detections (slower)")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="when --bodies is set, stop after this many seconds of video")
    ap.add_argument("--detection-confidence", type=int, default=40,
                    help="confidence threshold for BODY_38 detection (0-100)")
    ap.add_argument("--no-tracking", action="store_true",
                    help="disable ZED body tracking (no temporal smoothing, "
                         "no persistent IDs, raw per-frame detections only)")
    args = ap.parse_args()

    if not args.svo.is_file():
        raise FileNotFoundError(args.svo)

    zed = sl.Camera()
    init = sl.InitParameters()
    init.set_from_svo_file(str(args.svo))
    init.coordinate_units = sl.UNIT.METER
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP

    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"failed to open {args.svo}: {err}")

    try:
        info = zed.get_camera_information()
        cal = info.camera_configuration.calibration_parameters
        lc = cal.left_cam
        intrinsics = {
            "fx": float(lc.fx), "fy": float(lc.fy),
            "cx": float(lc.cx), "cy": float(lc.cy),
            "image_width":  int(lc.image_size.width),
            "image_height": int(lc.image_size.height),
        }

        fps = float(info.camera_configuration.fps)
        n_frames_total = zed.get_svo_number_of_frames()

        ptp = sl.PositionalTrackingParameters()
        ptp.set_as_static = True
        ptp.set_floor_as_origin = True
        zed.enable_positional_tracking(ptp)

        for _ in range(args.warmup_frames):
            if zed.grab() != sl.ERROR_CODE.SUCCESS:
                break

        plane = sl.Plane()
        reset_tx = sl.Transform()
        ferr = zed.find_floor_plane(plane, reset_tx)
        if ferr != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"find_floor_plane failed: {ferr}")
        eq = [float(v) for v in plane.get_plane_equation()]

        R, t = _create_floor_transform(eq)

        # Combined transform usable directly on SAM3D Body 3D output.
        T_world_from_rdf = np.eye(4)
        T_world_from_rdf[:3, :3] = R @ R_ZUP_FROM_RDF
        T_world_from_rdf[:3,  3] = t

        out = {
            "coordinate_system": "RIGHT_HANDED_Z_UP",
            "floor_plane": eq,
            "rotation": R.tolist(),
            "translation": t.tolist(),
            "intrinsics": intrinsics,
            "T_world_from_rdf": T_world_from_rdf.tolist(),
            "fps": fps,
        }
        args.out_dir.mkdir(parents=True, exist_ok=True)
        floor_json_path = args.out_dir / "zed_floor.json"
        floor_json_path.write_text(json.dumps(out, indent=2))

        print(f"[ok] wrote {floor_json_path}")
        print(f"    floor: a={eq[0]:+.4f} b={eq[1]:+.4f} c={eq[2]:+.4f} d={eq[3]:+.4f}")
        print(f"    camera height above floor: {t[2]:.3f} m")
        print(f"    intrinsics: fx={intrinsics['fx']:.3f} fy={intrinsics['fy']:.3f} "
              f"cx={intrinsics['cx']:.3f} cy={intrinsics['cy']:.3f} "
              f"({intrinsics['image_width']}x{intrinsics['image_height']})  "
              f"fps={fps:.2f}")

        if args.bodies:
            _extract_bodies(zed, R, t, fps, n_frames_total, args.out_dir,
                            args.max_seconds, args.detection_confidence,
                            enable_tracking=not args.no_tracking)

    finally:
        zed.close()


def _extract_bodies(zed, R, t, fps, n_frames_total, out_dir,
                    max_seconds, detection_confidence, enable_tracking=True):
    """Iterate through the SVO from frame 0, run BODY_38 detection, transform
    keypoints to world coords, save to <out_dir>/zed_bodies.npz."""

    body_param = sl.BodyTrackingParameters()
    body_param.enable_tracking = bool(enable_tracking)
    body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
    # Required for per-joint orientations: without fitting, ZED returns
    # keypoints but leaves local_orientation_per_joint empty. We need the
    # fitted skeleton angles for MHR retargeting (route A).
    body_param.enable_body_fitting = True
    if not enable_tracking:
        print("[bodies] tracking disabled (raw per-frame detections, no IDs)")
    if not hasattr(sl.BODY_FORMAT, "BODY_38"):
        raise RuntimeError(
            "this pyzed build does not have sl.BODY_FORMAT.BODY_38; "
            "use BODY_34 instead by editing the script"
        )
    body_param.body_format = sl.BODY_FORMAT.BODY_38

    err = zed.enable_body_tracking(body_param)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"enable_body_tracking failed: {err}")

    body_runtime = sl.BodyTrackingRuntimeParameters()
    body_runtime.detection_confidence_threshold = detection_confidence

    # Try to read skeleton bones if pyzed exposes them.
    try:
        bones = [(int(a), int(b)) for a, b in sl.BODY_38_BONES.value]
    except AttributeError:
        bones = []

    max_frames = (int(round(max_seconds * fps))
                  if max_seconds is not None else n_frames_total)

    zed.set_svo_position(0)

    frame_indices: list[int] = []
    body_ids: list[int] = []
    kp3d_world: list[np.ndarray] = []
    kp2d: list[np.ndarray] = []
    confidence: list[np.ndarray] = []
    bboxes_2d: list[np.ndarray] = []
    # Per-joint orientations for retargeting onto the MHR rig (route A in
    # retarget_zed_to_mhr.py). `local_orientation_per_joint` is relative to
    # parent (frame-independent); `global_root_orientation` is the root quat
    # in the ZED camera frame — apply R afterwards to reach world. Quaternions
    # are xyzw (sl.Orientation order). Guarded: not all pyzed builds expose
    # these, in which case we save empty arrays and warn once.
    local_orient: list[np.ndarray] = []
    root_orient: list[np.ndarray] = []
    have_orient = None  # tri-state: None=unknown, then True/False after frame 1

    bodies = sl.Bodies()
    frame_idx = 0
    with tqdm(total=max_frames, desc="ZED bodies", unit="frame") as pbar:
        while frame_idx < max_frames and zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_bodies(bodies, body_runtime)
            for body in bodies.body_list:
                # Filter by tracking_state only when tracking is enabled —
                # with tracking off the state isn't set to OK for raw detections.
                if enable_tracking and body.tracking_state != sl.OBJECT_TRACKING_STATE.OK:
                    continue
                kp = np.asarray(body.keypoint, dtype=np.float32)        # (38, 3) cam ZUP
                kp_w = (R @ kp.T).T + t                                  # → world
                frame_indices.append(frame_idx)
                body_ids.append(int(body.id))
                kp3d_world.append(kp_w.astype(np.float32))
                kp2d.append(np.asarray(body.keypoint_2d, dtype=np.float32))
                confidence.append(np.asarray(body.keypoint_confidence, dtype=np.float32))
                bb = body.bounding_box_2d
                bboxes_2d.append(np.asarray(
                    [bb[0][0], bb[0][1], bb[2][0], bb[2][1]], dtype=np.float32))

                # Orientations (need enable_body_fitting; some builds also
                # lack the attribute entirely). Decide on the first body by
                # checking the array is actually populated, not just present.
                if have_orient is None:
                    loc0 = getattr(body, "local_orientation_per_joint", None)
                    have_orient = loc0 is not None and np.asarray(loc0).size > 0
                    if not have_orient:
                        print("[bodies] WARNING: no per-joint orientations "
                              "(empty local_orientation_per_joint). Ensure "
                              "enable_body_fitting=True and a fitting-capable "
                              "pyzed build; route A will have no data.")
                if have_orient:
                    loc = np.asarray(body.local_orientation_per_joint,
                                     dtype=np.float32)            # (38, 4) xyzw
                    root = np.asarray(body.global_root_orientation,
                                      dtype=np.float32)           # (4,) xyzw
                    local_orient.append(loc)
                    root_orient.append(root)
            frame_idx += 1
            pbar.update(1)

    if not frame_indices:
        print("[bodies] no detections found")
        return

    # Orientation arrays: aligned 1:1 with the detection rows when present,
    # else empty (downstream checks .size). xyzw quaternion order.
    if local_orient:
        local_arr = np.stack(local_orient, axis=0).astype(np.float32)   # (N,38,4)
        root_arr = np.stack(root_orient, axis=0).astype(np.float32)     # (N,4)
    else:
        local_arr = np.zeros((0, len(bones) or 38, 4), dtype=np.float32)
        root_arr = np.zeros((0, 4), dtype=np.float32)

    bodies_path = out_dir / "zed_bodies.npz"
    np.savez_compressed(
        bodies_path,
        frame_indices=np.asarray(frame_indices, dtype=np.int64),
        body_ids=np.asarray(body_ids, dtype=np.int64),
        keypoints_3d_world=np.stack(kp3d_world, axis=0),
        keypoints_2d=np.stack(kp2d, axis=0),
        keypoint_confidence=np.stack(confidence, axis=0),
        bboxes_2d=np.stack(bboxes_2d, axis=0),
        # Per-joint orientations for MHR retargeting (xyzw; local = relative to
        # parent in ZED frame, root = global root quat in ZED camera frame).
        local_orientation_per_joint=local_arr,
        global_root_orientation=root_arr,
        bones=np.asarray(bones, dtype=np.int32),
        skeleton_name=np.asarray("stereolabs_body38"),
        fps=np.asarray(fps, dtype=np.float32),
        max_frame_idx=np.asarray(frame_idx, dtype=np.int64),
    )
    print(f"[ok] wrote {bodies_path}: "
          f"{len(frame_indices)} detections across {frame_idx} frames "
          f"(~{frame_idx/fps:.1f}s)"
          + (f", with per-joint orientations" if local_orient
             else ", (no orientations — pyzed build lacks them)"))


if __name__ == "__main__":
    main()
