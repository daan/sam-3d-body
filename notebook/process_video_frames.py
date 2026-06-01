"""
Process video frames with SAM 3D Body.

- One .npz per frame under <output_dir>/frames/NNNNNN.npz
- Atomic writes so partial files never exist
- Skips already-completed frames => safe to kill and resume
- Appends to manifest.jsonl as each frame finishes
- Computes per-person quality metrics (reprojection error, bbox size, etc.)
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    print("[warn] CUDA_VISIBLE_DEVICES not set; using default GPU 0", file=sys.stderr)

import cv2  # noqa: E402
import torch  # noqa: E402

from utils import setup_sam_3d_body  # noqa: E402


# ---------------------------------------------------------------------------
# Fields to keep per person. `pred_vertices` and `mask` dropped to save space.
# `pred_pose_raw` is redundant with body_pose_params. `mhr_model_params`
# kept for now -- drop after inspecting whether it duplicates the individuals.
# ---------------------------------------------------------------------------

PERSON_KEYS_TO_KEEP = {
    # Detection
    "bbox", "lhand_bbox", "rhand_bbox",
    # 2D/3D joints
    "pred_keypoints_2d",   # (70, 2)
    "pred_keypoints_3d",   # (70, 3)
    "pred_joint_coords",   # (127, 3)  full MHR skeleton
    # MHR parameters (compact, re-driveable)
    "body_pose_params",    # (133,)
    "shape_params",        # (45,)
    "hand_pose_params",
    "scale_params",
    "expr_params",
    "global_rot",
    "pred_global_rots",
    "mhr_model_params",
    # Scene placement
    "focal_length",
    "pred_cam_t",
}

# Toggled by --save-vertices. ~50-80 KB compressed per person/frame.
VERTEX_KEY = "pred_vertices"


# ---------------------------------------------------------------------------
# Principal-point recentering by image padding
# ---------------------------------------------------------------------------

def compute_pad(cx: float, cy: float, w: int, h: int) -> tuple[int, int, int, int]:
    """Return (pad_left, pad_top, pad_right, pad_bottom) such that after padding,
    the principal point (cx, cy) lies at the center of the padded image."""
    if cx <= w / 2:
        pad_left = int(round(w - 2 * cx))
        pad_right = 0
    else:
        pad_left = 0
        pad_right = int(round(2 * cx - w))
    if cy <= h / 2:
        pad_top = int(round(h - 2 * cy))
        pad_bottom = 0
    else:
        pad_top = 0
        pad_bottom = int(round(2 * cy - h))
    return pad_left, pad_top, pad_right, pad_bottom


def pad_bgr(img_bgr: np.ndarray, pad_left: int, pad_top: int,
            pad_right: int, pad_bottom: int) -> np.ndarray:
    """Pad a BGR image with black on the requested sides."""
    if pad_left == pad_top == pad_right == pad_bottom == 0:
        return img_bgr
    return cv2.copyMakeBorder(
        img_bgr, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(0, 0, 0),
    )


def unpad_person_inplace(p: dict, pad_left: int, pad_top: int) -> None:
    """Subtract (pad_left, pad_top) from all 2D pixel fields so they are in
    original-image coords. Operates on a dict produced by flatten_person."""
    if pad_left == 0 and pad_top == 0:
        return
    if "pred_keypoints_2d" in p:
        kp = np.asarray(p["pred_keypoints_2d"], dtype=np.float32).copy()
        kp[..., 0] -= pad_left
        kp[..., 1] -= pad_top
        p["pred_keypoints_2d"] = kp
    for k in ("bbox", "lhand_bbox", "rhand_bbox"):
        v = p.get(k, None)
        if v is None:
            continue
        arr = np.asarray(v, dtype=np.float32).copy()
        if arr.size >= 4:
            arr[..., 0] -= pad_left  # x1
            arr[..., 1] -= pad_top   # y1
            arr[..., 2] -= pad_left  # x2
            arr[..., 3] -= pad_top   # y2
        p[k] = arr


# ---------------------------------------------------------------------------
# Derived per-person quality metrics
# ---------------------------------------------------------------------------

# Set once per job after inspecting one frame. See reprojection_error().
REPROJ_MODE = None  # "camera_frame" or "local_plus_cam_t"


def _project_camera_frame(kp3d, f, cx, cy):
    Z = kp3d[:, 2:3]
    Z = np.where(np.abs(Z) < 1e-6, 1e-6, Z)  # avoid div-by-zero
    return kp3d[:, :2] / Z * f + np.array([cx, cy])


def reprojection_error(person, image_hw):
    """Mean 2D distance (px) between predicted 2D joints and reprojected 3D joints."""
    global REPROJ_MODE
    try:
        kp3d = np.asarray(person["pred_keypoints_3d"], dtype=np.float64)
        kp2d = np.asarray(person["pred_keypoints_2d"], dtype=np.float64)
        f = float(np.asarray(person["focal_length"]).ravel()[0])
        cam_t = np.asarray(person["pred_cam_t"], dtype=np.float64).ravel()
        H, W = image_hw[:2]
        cx, cy = W / 2.0, H / 2.0

        if REPROJ_MODE is None:
            # First frame: decide which coordinate convention the model uses.
            err_cam = np.linalg.norm(
                _project_camera_frame(kp3d, f, cx, cy) - kp2d, axis=1
            ).mean()
            err_loc = np.linalg.norm(
                _project_camera_frame(kp3d + cam_t, f, cx, cy) - kp2d, axis=1
            ).mean()
            REPROJ_MODE = "camera_frame" if err_cam < err_loc else "local_plus_cam_t"
            print(f"[reproj] mode={REPROJ_MODE} "
                  f"(cam_frame={err_cam:.1f}px, local+t={err_loc:.1f}px)")

        xyz = kp3d if REPROJ_MODE == "camera_frame" else kp3d + cam_t
        proj = _project_camera_frame(xyz, f, cx, cy)
        return float(np.linalg.norm(proj - kp2d, axis=1).mean())
    except Exception:
        return float("nan")


def compute_derived(person, image_hw):
    """Cheap scalar quality signals, per person per frame."""
    H, W = image_hw[:2]
    derived = {}

    # Bbox geometry
    bbox = np.asarray(person.get("bbox", [0, 0, 0, 0]), dtype=np.float64).ravel()
    if bbox.size >= 4:
        x1, y1, x2, y2 = bbox[:4]
        bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
        derived["bbox_h_px"] = float(bh)
        derived["bbox_w_px"] = float(bw)
        derived["bbox_area_frac"] = float(bw * bh / (W * H))
        edge = 5
        derived["at_image_edge"] = bool(
            x1 < edge or y1 < edge or x2 > W - edge or y2 > H - edge
        )

    # Hand visibility
    for side in ("lhand_bbox", "rhand_bbox"):
        b = person.get(side, None)
        if b is None:
            derived[f"{side}_visible"] = False
        else:
            b = np.asarray(b).ravel()
            derived[f"{side}_visible"] = bool(b.size >= 4 and (b[2] - b[0]) > 0)

    # Reprojection
    derived["reproj_err_mean_px"] = reprojection_error(person, image_hw)

    return derived


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, dict):
        return {k: _to_numpy(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_numpy(v) for v in x]
    return x


def flatten_person(raw, image_hw):
    """Keep wanted fields, convert to numpy, attach derived metrics."""
    out = {}
    for k, v in raw.items():
        if k in PERSON_KEYS_TO_KEEP:
            out[k] = _to_numpy(v)
    derived = compute_derived(out, image_hw)
    for k, v in derived.items():
        out[f"q_{k}"] = v  # prefix so quality metrics are easy to spot
    return out


def save_frame_atomic(path, frame_idx, persons, image_shape):
    tmp = path.with_suffix(path.suffix + ".tmp")
    flat = {
        "frame_idx": np.int64(frame_idx),
        "image_hw": np.asarray(image_shape[:2], dtype=np.int64),
        "num_persons": np.int64(len(persons)),
    }
    for i, p in enumerate(persons):
        keys = []
        for k, v in p.items():
            try:
                arr = np.asarray(v)
            except Exception:
                arr = np.asarray(v, dtype=object)
            flat[f"p{i}__{k}"] = arr
            keys.append(k)
        flat[f"p{i}__keys"] = np.asarray(keys)
    # np.savez_compressed auto-appends ".npz" when given a string path,
    # which breaks atomic rename. Pass a file handle to prevent that.
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **flat)
    os.replace(tmp, path)


def append_manifest(manifest_path, record):
    with open(manifest_path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# Frame sources
# ---------------------------------------------------------------------------

def iter_frames_from_dir(frames_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in exts)
    for idx, p in enumerate(files):
        img = cv2.imread(str(p))
        if img is None:
            print(f"[warn] could not read {p}, skipping", file=sys.stderr)
            continue
        yield idx, str(p), img


def iter_frames_from_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield idx, f"{video_path.name}#{idx:06d}", frame
        idx += 1
    cap.release()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--frames-dir", type=Path, help="folder of frame images")
    src.add_argument("--video", type=Path, help="video file")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--hf-repo-id", default="facebook/sam-3d-body-dinov3",
    )
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--save-vertices", action="store_true",
        help="Also save pred_vertices per person per frame (~50-80 KB compressed). "
             "Required for mesh rendering in Rerun.",
    )
    parser.add_argument(
        "--intrinsics", type=float, nargs=4, default=None,
        metavar=("FX", "FY", "CX", "CY"),
        help="Calibrated camera intrinsics in pixels at the input frame resolution. "
             "When given, skips the FOV estimator (faster, more accurate than moge2). "
             "Get from ZED via pyzed.sl.Camera or ~/.zed/SN<serial>.conf.",
    )
    parser.add_argument(
        "--recenter-principal-point", action="store_true",
        help="Pad each frame so the principal point (cx, cy) lands at the padded "
             "image center. The model was trained assuming a centered principal "
             "point, so this is required for accurate 2D keypoints with off-center "
             "calibrated intrinsics. 3D output is unaffected. "
             "Saved bbox / pred_keypoints_2d are de-padded back to original coords.",
    )
    args = parser.parse_args()

    if args.save_vertices:
        PERSON_KEYS_TO_KEEP.add(VERTEX_KEY)

    if args.recenter_principal_point and args.intrinsics is None:
        parser.error("--recenter-principal-point requires --intrinsics")

    # Calibrated intrinsics: build a (1, 3, 3) torch tensor and skip moge2.
    cam_int = None
    pad_left = pad_top = 0
    if args.intrinsics is not None:
        fx, fy, cx, cy = args.intrinsics
        eff_cx, eff_cy = cx, cy
        if args.recenter_principal_point:
            # We don't know image width/height until we read a frame, so the
            # actual pad is computed there. Here we just stash the original cx/cy.
            print("[cam] --recenter-principal-point: padding will be applied per-frame")
        cam_int = torch.tensor(
            [[[fx, 0.0, eff_cx], [0.0, fy, eff_cy], [0.0, 0.0, 1.0]]],
            dtype=torch.float32,
        )
        print(f"[cam] calibrated intrinsics: fx={fx} fy={fy} cx={cx} cy={cy}")
        if not args.recenter_principal_point:
            print("[cam] WARNING: principal point not at image center will cause "
                  "2D keypoint offsets. Consider --recenter-principal-point.")

    out = args.output_dir
    frames_out = out / "frames"
    frames_out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.jsonl"
    meta_path = out / "meta.json"

    # Resume: skip frames that already have an .npz
    done = set()
    for p in frames_out.glob("*.npz"):
        try:
            done.add(int(p.stem))
        except ValueError:
            pass
    print(f"[resume] found {len(done)} existing frames in {frames_out}")

    print(f"[load] {args.hf_repo_id}")
    estimator = setup_sam_3d_body(
        hf_repo_id=args.hf_repo_id,
        fov_name="" if cam_int is not None else "moge2",
    )


    # Save mesh topology once (useful if you later regenerate vertices)
    faces = getattr(estimator, "faces", None)
    if faces is not None:
        np.save(out / "faces.npy", _to_numpy(faces))

    if args.frames_dir:
        frame_iter = iter_frames_from_dir(args.frames_dir)
        source_desc = str(args.frames_dir)
    else:
        frame_iter = iter_frames_from_video(args.video)
        source_desc = str(args.video)

    meta = {
        "source": source_desc,
        "hf_repo_id": args.hf_repo_id,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "faces_file": "faces.npy" if faces is not None else None,
        "kept_keys": sorted(PERSON_KEYS_TO_KEEP),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    t_start = time.time()
    processed = 0

    # Lazily computed on the first frame when --recenter-principal-point is on,
    # since we don't know the frame resolution until we read one.
    pads_computed = not args.recenter_principal_point
    pad_left = pad_top = pad_right = pad_bottom = 0

    for idx, path, img_bgr in frame_iter:
        frame_file = frames_out / f"{idx:06d}.npz"
        if idx in done or frame_file.exists():
            continue

        if not pads_computed:
            h, w = img_bgr.shape[:2]
            fx_i, fy_i, cx_i, cy_i = args.intrinsics
            pad_left, pad_top, pad_right, pad_bottom = compute_pad(cx_i, cy_i, w, h)
            new_cx, new_cy = cx_i + pad_left, cy_i + pad_top
            cam_int = torch.tensor(
                [[[fx_i, 0.0, new_cx], [0.0, fy_i, new_cy], [0.0, 0.0, 1.0]]],
                dtype=torch.float32,
            )
            new_w, new_h = w + pad_left + pad_right, h + pad_top + pad_bottom
            print(f"[cam] padding {w}x{h} -> {new_w}x{new_h}  "
                  f"(left={pad_left} top={pad_top} right={pad_right} bottom={pad_bottom})  "
                  f"new principal point: cx={new_cx} cy={new_cy}")
            pads_computed = True

        t_frame_start = time.time()
        try:
            if args.recenter_principal_point:
                # Pad in BGR, convert to RGB, run model with adjusted cam_int.
                padded_bgr = pad_bgr(img_bgr, pad_left, pad_top, pad_right, pad_bottom)
                padded_rgb = cv2.cvtColor(padded_bgr, cv2.COLOR_BGR2RGB)
                raw_outputs = estimator.process_one_image(padded_rgb, cam_int=cam_int)
                inference_shape = padded_bgr.shape
            elif args.frames_dir:
                raw_outputs = estimator.process_one_image(path, cam_int=cam_int)
                inference_shape = img_bgr.shape
            else:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                try:
                    raw_outputs = estimator.process_one_image(img_rgb, cam_int=cam_int)
                except TypeError:
                    tmp_img = out / "_tmp_frame.jpg"
                    cv2.imwrite(str(tmp_img), img_bgr)
                    raw_outputs = estimator.process_one_image(str(tmp_img), cam_int=cam_int)
                inference_shape = img_bgr.shape

            # flatten_person uses inference_shape for q_ metrics (reprojection,
            # bbox_area_frac, etc.) so they're self-consistent with the 2D the
            # model produced. We unpad afterwards so the SAVED 2D fields are in
            # original-image coords.
            persons = [
                flatten_person(p, inference_shape) for p in (raw_outputs or [])
            ]
            if args.recenter_principal_point:
                for p in persons:
                    unpad_person_inplace(p, pad_left, pad_top)

            save_frame_atomic(frame_file, idx, persons, img_bgr.shape)

            # Per-frame summary for the manifest -- easy to grep later
            dt_s = time.time() - t_frame_start
            summary = {
                "frame": idx,
                "src": path,
                "n_persons": len(persons),
                "ok": True,
                "t": time.time(),
                "dt_s": dt_s,
                "dt_per_person_s": dt_s / max(len(persons), 1),
            }
            if persons:
                summary["reproj_err_px"] = [
                    p.get("q_reproj_err_mean_px", None) for p in persons
                ]
                summary["bbox_h_px"] = [
                    p.get("q_bbox_h_px", None) for p in persons
                ]
            append_manifest(manifest_path, summary)
            processed += 1

            if processed % args.log_every == 0:
                rate = processed / (time.time() - t_start + 1e-6)
                dt_ms = dt_s * 1000
                per_person_ms = dt_ms / max(len(persons), 1)
                print(f"[{idx:06d}] persons={len(persons)}  "
                      f"total_done={len(done) + processed}  "
                      f"{dt_ms:.0f} ms/frame  "
                      f"{per_person_ms:.0f} ms/person  "
                      f"avg {rate:.2f} fps")

        except Exception as e:
            append_manifest(manifest_path, {
                "frame": idx,
                "src": path,
                "ok": False,
                "error": repr(e),
                "trace": traceback.format_exc(),
                "t": time.time(),
            })
            print(f"[error] frame {idx}: {e}", file=sys.stderr)
            torch.cuda.empty_cache()

    print(f"[done] processed {processed} new frames in "
          f"{time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()