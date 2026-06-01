"""
Rerun viewer for SAM3D Body per-frame output.

Reads <frames-dir>/frames/*.npz produced by process_video_frames.py and
opens an interactive viewer with:
  - 2D view: video frame + bboxes + 2D body skeleton
  - 3D view: body meshes (if --save-vertices was used) + 3D joint skeletons
  - timeline scrubber on frame_idx

Mesh visibility can be toggled in the viewer (right click the /world/person_*/mesh
entity → Hide).

Run from the sam_3d_body conda env:
    pip install rerun-sdk
    PYTHONPATH=. python notebook/visualize_rerun.py --frames-dir out/bending4 \
        --video ~/data/vanRaam/bending4.mp4
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

from sam_3d_body.metadata.mhr70 import pose_info as MHR70


def load_world_transform(path: Path | None) -> np.ndarray | None:
    """Load a 4x4 RDF->world transform from a zed_floor.json file.
    Returns None if path is None or the file doesn't exist."""
    if path is None or not path.exists():
        return None
    data = json.loads(path.read_text())
    if "T_world_from_rdf" not in data:
        print(f"[floor] {path} missing T_world_from_rdf; ignoring")
        return None
    T = np.asarray(data["T_world_from_rdf"], dtype=np.float64)
    if T.shape != (4, 4):
        print(f"[floor] T_world_from_rdf has wrong shape {T.shape}; ignoring")
        return None
    cs = data.get("coordinate_system", "?")
    if cs != "RIGHT_HANDED_Z_UP":
        print(f"[floor] WARNING: coordinate_system={cs!r}, expected RIGHT_HANDED_Z_UP")
    cam_height = float(np.asarray(data.get("translation", [0, 0, 0]))[2])
    print(f"[floor] loaded {path} (camera height above floor: {cam_height:.3f} m)")
    return T


def apply_T(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to points of shape (..., 3)."""
    R = T[:3, :3]
    t = T[:3, 3]
    return points @ R.T + t


def load_zed_bodies(zed_dir: Path | None) -> dict | None:
    """Load zed_bodies.npz and pre-index by frame for fast per-frame lookup.
    Returns None if the file isn't found."""
    if zed_dir is None:
        return None
    p = zed_dir / "zed_bodies.npz"
    if not p.exists():
        print(f"[zed] {p} not found; ZED overlay disabled")
        return None
    z = np.load(p, allow_pickle=False)
    fidx = z["frame_indices"]
    frame_to_rows: dict[int, list[int]] = {}
    for i, f in enumerate(fidx):
        frame_to_rows.setdefault(int(f), []).append(i)
    print(f"[zed] loaded {len(fidx)} detections across {len(frame_to_rows)} frames")
    return {
        "frame_to_rows": frame_to_rows,
        "body_ids":      z["body_ids"],
        "kp3d_world":    z["keypoints_3d_world"],
        "kp2d":          z["keypoints_2d"],
        "conf":          z["keypoint_confidence"],
        "bboxes":        z["bboxes_2d"],
    }


def _zed_color(cls_id: int) -> tuple[int, int, int]:
    """Cool-toned palette (blue/cyan/teal) for ZED bodies — distinct from
    SAM3D's red/grey."""
    h = 0.55 + (cls_id * 0.07) % 0.30   # hue in [0.55, 0.85] = cyan→blue→purple
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, 0.85, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


# ----- MHR70 helpers --------------------------------------------------------

NAME2ID = {info["name"]: info["id"] for info in MHR70["keypoint_info"].values()}
ID2NAME = {v: k for k, v in NAME2ID.items()}

# Body-only keypoint ids (MHR70 ordering puts body first, then hands).
# 0 nose, 1-2 eyes, 3-4 ears, 5-6 shoulders, 7-8 elbows, 9-10 hips,
# 11-12 knees, 13-14 ankles, 15-22 toes/heels.
BODY_KP_IDS = list(range(23))

# Edges from MHR70 skeleton_info, filtered to body-only endpoints.
BODY_EDGES: list[tuple[int, int]] = []
for link_info in MHR70["skeleton_info"].values():
    a_name, b_name = link_info["link"]
    a, b = NAME2ID[a_name], NAME2ID[b_name]
    if a in BODY_KP_IDS and b in BODY_KP_IDS:
        BODY_EDGES.append((a, b))


def color_for(idx: int) -> tuple[int, int, int]:
    """Stable RGB color from a non-negative integer (golden-ratio hue)."""
    h = (idx * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two XYXY bboxes."""
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def dedupe_persons(persons: list[dict], iou_thresh: float) -> list[dict]:
    """Greedy dedupe: sort by reprojection error ascending, drop later persons
    whose bbox overlaps a kept one by more than iou_thresh."""
    if not persons or iou_thresh >= 1.0:
        return persons

    def err(p: dict) -> float:
        v = p.get("q_reproj_err_mean_px", float("inf"))
        v = float(np.asarray(v).ravel()[0]) if v is not None else float("inf")
        return v if np.isfinite(v) else float("inf")

    ordered = sorted(persons, key=err)
    kept: list[dict] = []
    for cand in ordered:
        cand_bbox = np.asarray(cand["bbox"]).ravel()
        if any(bbox_iou(cand_bbox, np.asarray(k["bbox"]).ravel()) > iou_thresh
               for k in kept):
            continue
        kept.append(cand)
    return kept


def split_primary(persons: list[dict]) -> tuple[dict | None, list[dict]]:
    """Pick the person nearest the camera (smallest pred_cam_t Z in RDF frame)
    as 'primary'; the rest go to 'others'."""
    if not persons:
        return None, []
    def depth(p: dict) -> float:
        return float(np.asarray(p["pred_cam_t"]).ravel()[2])
    nearest = min(persons, key=depth)
    others = [p for p in persons if p is not nearest]
    return nearest, others


# Class ids used in AnnotationContext; control colors.
CLS_PRIMARY = 1
CLS_OTHER = 2

# ZED BODY_38 skeleton bone connections (hardcoded — see visualize_zed_bodies.py
# for the keypoint index → name mapping).
BODY_38_BONES: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
    (5, 6), (5, 7), (6, 8), (7, 9),
    (4, 10), (10, 12), (12, 14), (14, 16),
    (16, 30), (16, 32), (16, 34), (16, 36),
    (4, 11), (11, 13), (13, 15), (15, 17),
    (17, 31), (17, 33), (17, 35), (17, 37),
    (0, 18), (18, 20), (20, 22),
    (22, 24), (22, 26), (22, 28),
    (0, 19), (19, 21), (21, 23),
    (23, 25), (23, 27), (23, 29),
]

# Class ids 100..(100+ZED_NUM_CLASSES-1) reserved for ZED bodies (blue palette,
# distinguishable from SAM3D's red/grey).
ZED_CLS_BASE = 100
ZED_NUM_CLASSES = 32
ZED_NUM_KP = 38

# MHR70 foot keypoints. Mid-heel sits on the floor under the person.
LEFT_HEEL_ID = NAME2ID["left_heel"]
RIGHT_HEEL_ID = NAME2ID["right_heel"]


def foot_position_3d(p: dict, T: np.ndarray | None = None) -> np.ndarray:
    """Mid-heel position of `p` as a (3,) vector.
    In camera-RDF frame by default; in world frame if T (4x4 RDF->world) given."""
    kp3d = np.asarray(p["pred_keypoints_3d"])
    cam_t = np.asarray(p["pred_cam_t"]).ravel()
    mid_rdf = 0.5 * (kp3d[LEFT_HEEL_ID] + kp3d[RIGHT_HEEL_ID]) + cam_t
    if T is None:
        return mid_rdf
    return apply_T(T, mid_rdf[None, :])[0]


# ----- I/O ------------------------------------------------------------------

def load_frame_npz(path: Path) -> tuple[int, tuple[int, int], list[dict]]:
    z = np.load(path, allow_pickle=True)
    n = int(z["num_persons"])
    persons = []
    for i in range(n):
        keys = z[f"p{i}__keys"]
        persons.append({str(k): z[f"p{i}__{k}"] for k in keys})
    return int(z["frame_idx"]), tuple(z["image_hw"]), persons


# ----- Logging --------------------------------------------------------------

def setup_static(
    out_dir: Path,
    scene_box: tuple[float, float, float, float, float, float],
    T_world_from_rdf: np.ndarray | None,
) -> np.ndarray | None:
    """Log everything that doesn't change across frames. Returns faces array or None."""
    if T_world_from_rdf is None:
        # Old behavior: "world" is just the camera-RDF frame.
        rr.log("/", rr.ViewCoordinates.RDF, static=True)
    else:
        # True world frame, Z up, floor on the XY plane.
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # Annotation context: SAM3D classes (MHR70 body subset) + ZED classes
    # (BODY_38). Two skeleton conventions live side by side in one viewer.
    sam_cls_specs = [
        (CLS_PRIMARY, "sam3d_primary", (255, 80,  80)),   # red-ish
        (CLS_OTHER,   "sam3d_other",   (140, 140, 140)),  # grey
    ]
    sam_kp_annotations = [
        rr.AnnotationInfo(id=kid, label=ID2NAME[kid]) for kid in BODY_KP_IDS
    ]
    sam_descriptions = [
        rr.ClassDescription(
            info=rr.AnnotationInfo(id=cls_id, label=label, color=color),
            keypoint_annotations=sam_kp_annotations,
            keypoint_connections=BODY_EDGES,
        )
        for cls_id, label, color in sam_cls_specs
    ]
    zed_kp_annotations = [
        rr.AnnotationInfo(id=k, label=f"zed_kp{k}") for k in range(ZED_NUM_KP)
    ]
    zed_descriptions = [
        rr.ClassDescription(
            info=rr.AnnotationInfo(
                id=ZED_CLS_BASE + i,
                label=f"zed_body_{i}",
                color=_zed_color(i),
            ),
            keypoint_annotations=zed_kp_annotations,
            keypoint_connections=BODY_38_BONES,
        )
        for i in range(ZED_NUM_CLASSES)
    ]
    rr.log("/", rr.AnnotationContext(sam_descriptions + zed_descriptions),
           static=True)

    if T_world_from_rdf is None:
        # Static wireframe box that defines the 3D-view bounds in RDF coords.
        x_min, y_min, z_min, x_max, y_max, z_max = scene_box
        cx, cy, cz = (x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2
        hx, hy, hz = (x_max - x_min) / 2, (y_max - y_min) / 2, (z_max - z_min) / 2
        rr.log(
            "/world/scene_bounds",
            rr.Boxes3D(
                centers=[[cx, cy, cz]],
                half_sizes=[[hx, hy, hz]],
                colors=[[80, 80, 80]],
            ),
            static=True,
        )
    else:
        # World mode: anchor view to a 6x6x3 m box sitting on the floor in
        # front of the camera. The camera sits at (0, 0, cam_height) in world.
        cam_height = float(T_world_from_rdf[2, 3])
        box_half = max(3.0, cam_height + 1.0)
        rr.log(
            "/world/scene_bounds",
            rr.Boxes3D(
                centers=[[0.0, 0.0, box_half / 2.0]],
                half_sizes=[[box_half, box_half, box_half / 2.0]],
                colors=[[80, 80, 80]],
            ),
            static=True,
        )
        # Floor grid: a flat box on z=0 to make the ground visible.
        rr.log(
            "/world/floor",
            rr.Boxes3D(
                centers=[[0.0, 0.0, 0.0]],
                half_sizes=[[box_half, box_half, 0.002]],
                colors=[[60, 90, 60]],
            ),
            static=True,
        )
        # Camera frustum at the right pose in world. The Transform3D maps
        # SAM3D-RDF camera-local coords to world coords, so an Image / Pinhole
        # logged under /world/zed_camera/... would be auto-placed correctly.
        # (We're not logging the image there in this build; the 2D view still
        # uses /world/zed_camera/image. This entity is purely for the 3D frustum.)
        R = T_world_from_rdf[:3, :3]
        t = T_world_from_rdf[:3, 3]
        rr.log(
            "/world/zed_camera",
            rr.Transform3D(translation=t.tolist(), mat3x3=R.tolist()),
            static=True,
        )

    faces_path = out_dir / "faces.npy"
    if faces_path.exists():
        return np.load(faces_path).astype(np.int32)
    return None


def log_zed_frame(frame_idx: int, zed_data: dict | None) -> None:
    """Log ZED BODY_38 detections for this frame, if any."""
    rr.log("/world/zed_bodies", rr.Clear(recursive=True))
    rr.log("/world/zed_camera/image/zed_bodies", rr.Clear(recursive=True))
    if zed_data is None:
        return
    rows = zed_data["frame_to_rows"].get(int(frame_idx), [])
    if not rows:
        return
    body_ids = zed_data["body_ids"]
    kp3d_world = zed_data["kp3d_world"]
    kp2d = zed_data["kp2d"]
    conf = zed_data["conf"]
    bboxes = zed_data["bboxes"]

    ids_in_frame = body_ids[rows]
    ids_collide = len(np.unique(ids_in_frame)) != len(ids_in_frame)

    for k, row in enumerate(rows):
        bid_raw = int(body_ids[row])
        entity_idx = k if ids_collide or bid_raw < 0 else bid_raw
        cls_id = ZED_CLS_BASE + (entity_idx % ZED_NUM_CLASSES)
        c = conf[row]
        valid = c > 0.1
        if not valid.any():
            continue
        kp_ids = np.flatnonzero(valid).astype(np.int32)

        rr.log(
            f"/world/zed_bodies/body_{entity_idx:03d}",
            rr.Points3D(
                positions=kp3d_world[row][valid].astype(np.float32),
                class_ids=cls_id,
                keypoint_ids=kp_ids,
                radii=0.02,
                show_labels=False,
            ),
        )
        rr.log(
            f"/world/zed_camera/image/zed_bodies/body_{entity_idx:03d}",
            rr.Points2D(
                positions=kp2d[row][valid].astype(np.float32),
                class_ids=cls_id,
                keypoint_ids=kp_ids,
                radii=3.0,
                show_labels=False,
            ),
        )
        bb = bboxes[row]
        rr.log(
            f"/world/zed_camera/image/zed_bodies/body_{entity_idx:03d}/bbox",
            rr.Boxes2D(
                array=bb[None, :],
                array_format=rr.Box2DFormat.XYXY,
                class_ids=[cls_id],
                show_labels=False,
            ),
        )


def log_frame(
    frame_idx: int,
    image_hw: tuple[int, int],
    persons: list[dict],
    video_frame_rgb: np.ndarray | None,
    faces: np.ndarray | None,
    log_mesh: bool,
    T_world_from_rdf: np.ndarray | None = None,
    zed_data: dict | None = None,
) -> np.ndarray | None:
    """Log one frame. Returns the primary's foot position (3,) if a primary is
    present, else None. Position is in world coords if T_world_from_rdf given,
    otherwise camera-RDF coords."""
    rr.set_time("frame", sequence=frame_idx)

    # Camera pinhole. In world-frame mode, the camera frustum's WORLD pose is
    # set in setup_static via Transform3D on /world/zed_camera; this Pinhole
    # is logged on a sibling /world/camera purely so the 2D image view (which
    # uses /world/zed_camera/image as its origin) renders correctly.
    if persons:
        f = float(np.asarray(persons[0]["focal_length"]).ravel()[0])
        H, W = int(image_hw[0]), int(image_hw[1])
        rr.log(
            "/world/zed_camera",
            rr.Pinhole(focal_length=f, width=W, height=H,
                       camera_xyz=rr.ViewCoordinates.RDF),
        )

    if video_frame_rgb is not None:
        # Encode JPEG via cv2 (avoids a PIL fileno() incompatibility in some
        # Pillow/Rerun version combos).
        bgr = cv2.cvtColor(video_frame_rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            rr.log(
                "/world/zed_camera/image",
                rr.EncodedImage(contents=buf.tobytes(), media_type="image/jpeg"),
            )
        else:
            rr.log(
                "/world/zed_camera/image",
                rr.Image(video_frame_rgb, color_model=rr.ColorModel.RGB),
            )

    # Clear prior frame's per-person entities to handle varying person counts.
    rr.log("/world/zed_camera/image/persons", rr.Clear(recursive=True))
    rr.log("/world/persons", rr.Clear(recursive=True))

    primary, others = split_primary(persons)

    def log_person(entity_path: str, p: dict, cls_id: int) -> None:
        bbox = np.asarray(p["bbox"]).ravel()[:4]
        rr.log(
            f"/world/zed_camera/image/persons/{entity_path}/bbox",
            rr.Boxes2D(
                array=bbox[None, :],
                array_format=rr.Box2DFormat.XYXY,
                class_ids=[cls_id],
                show_labels=False,
            ),
        )

        kp2d = np.asarray(p["pred_keypoints_2d"])[BODY_KP_IDS]
        rr.log(
            f"/world/zed_camera/image/persons/{entity_path}/keypoints2d",
            rr.Points2D(
                positions=kp2d,
                class_ids=cls_id,
                keypoint_ids=BODY_KP_IDS,
                radii=3.0,
                show_labels=False,
            ),
        )

        cam_t = np.asarray(p["pred_cam_t"]).ravel()
        kp3d = np.asarray(p["pred_keypoints_3d"])[BODY_KP_IDS] + cam_t
        if T_world_from_rdf is not None:
            kp3d = apply_T(T_world_from_rdf, kp3d).astype(np.float32)
        rr.log(
            f"/world/persons/{entity_path}/joints3d",
            rr.Points3D(
                positions=kp3d,
                class_ids=cls_id,
                keypoint_ids=BODY_KP_IDS,
                radii=0.015,
                show_labels=False,
            ),
        )

        if log_mesh and faces is not None and "pred_vertices" in p:
            verts = np.asarray(p["pred_vertices"]).astype(np.float32) + cam_t
            if T_world_from_rdf is not None:
                verts = apply_T(T_world_from_rdf, verts).astype(np.float32)
            # Use the class color so primary and others remain visually distinct.
            color = (255, 80, 80) if cls_id == CLS_PRIMARY else (140, 140, 140)
            rr.log(
                f"/world/persons/{entity_path}/mesh",
                rr.Mesh3D(
                    vertex_positions=verts,
                    triangle_indices=faces,
                    albedo_factor=list(color),
                ),
            )

    if primary is not None:
        log_person("primary", primary, CLS_PRIMARY)
    for i, p in enumerate(others):
        log_person(f"others/p{i}", p, CLS_OTHER)

    log_zed_frame(frame_idx, zed_data)

    # Primary's foot position in 3D: re-logged per frame as a marker dot,
    # plus collected into a polyline at the end of the run.
    if primary is not None:
        foot = foot_position_3d(primary, T_world_from_rdf)
        rr.log(
            "/world/trajectory/current",
            rr.Points3D(positions=foot[None, :],
                        colors=[[255, 80, 80]],
                        radii=0.04),
        )
        return foot
    else:
        rr.log("/world/trajectory/current", rr.Clear(recursive=False))
        return None


def build_blueprint() -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(name="Camera", origin="/world/zed_camera/image"),
            rrb.Spatial3DView(name="3D", origin="/world"),
        ),
        collapse_panels=False,
    )




# ----- Main -----------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", type=Path, required=True,
                    help="output_dir from process_video_frames.py (contains frames/, faces.npy)")
    ap.add_argument("--video", type=Path, default=None,
                    help="source video; if given, frames are shown in the 2D view")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-mesh", action="store_true",
                    help="skip logging meshes even if pred_vertices is present (smaller recording)")
    ap.add_argument("--dedupe-iou", type=float, default=0.7,
                    help="drop overlapping detections (same person twice). "
                         "Among any pair with bbox IoU > this, keep the one with lower "
                         "q_reproj_err_mean_px. Set >=1.0 to disable.")
    ap.add_argument(
        "--scene-box", type=float, nargs=6,
        default=(-3.0, -2.0, 0.0, 3.0, 2.0, 8.0),
        metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
        help="static wireframe box that anchors the 3D view bounds "
             "(RDF: +X right, +Y down, +Z forward in meters). "
             "Default is a ~6×4×8 m volume in front of the camera.",
    )
    ap.add_argument("--save", type=Path, default=None,
                    help="if set, save to this .rrd file instead of spawning the viewer")
    ap.add_argument(
        "--floor", type=Path, default=None,
        help="path to zed_floor.json produced by extract_zed_floor.py. "
             "If omitted, looks for <frames-dir>/zed_floor.json. "
             "When present, 3D bodies + trajectory are shown in world coords "
             "(Z up, floor at z=0).",
    )
    ap.add_argument(
        "--zed-bodies", type=Path, default=None,
        help="path to zed_bodies.npz from extract_zed_floor.py. "
             "If omitted, looks for <frames-dir>/zed_bodies.npz. "
             "When present, ZED BODY_38 skeletons are overlaid in blue.",
    )
    args = ap.parse_args()

    # Auto-discover zed_floor.json next to the frames dir if --floor not set.
    floor_path = args.floor
    if floor_path is None:
        candidate = args.frames_dir / "zed_floor.json"
        if candidate.exists():
            floor_path = candidate
    T_world_from_rdf = load_world_transform(floor_path)

    # Auto-discover zed_bodies.npz; load via its directory.
    if args.zed_bodies is not None:
        zed_dir = args.zed_bodies.parent
    elif (args.frames_dir / "zed_bodies.npz").exists():
        zed_dir = args.frames_dir
    else:
        zed_dir = None
    zed_data = load_zed_bodies(zed_dir)

    rr.init("sam3d-body-viewer", spawn=args.save is None)
    if args.save:
        rr.save(str(args.save))

    rr.send_blueprint(build_blueprint())
    faces = setup_static(args.frames_dir, tuple(args.scene_box), T_world_from_rdf)
    if faces is None and not args.no_mesh:
        print("[viz] no faces.npy in frames dir → mesh will not be rendered")

    frames_dir = args.frames_dir / "frames"
    npz_paths = sorted(frames_dir.glob("*.npz"))
    if args.max_frames is not None:
        npz_paths = npz_paths[: args.max_frames]

    # Optional video reader, indexed by frame_idx.
    cap = None
    if args.video is not None:
        cap = cv2.VideoCapture(str(args.video))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {args.video}")

    next_video_idx = 0  # cap.read() returns frames in order
    n_in = n_out = 0
    trajectory: list[np.ndarray] = []  # 3D foot position per frame

    def log_trajectory_static() -> None:
        if len(trajectory) < 2:
            return
        rr.log(
            "/world/trajectory/path",
            rr.LineStrips3D(
                [np.stack(trajectory, axis=0).astype(np.float32)],
                colors=[[255, 80, 80]],
                radii=0.01,
            ),
            static=True,
        )

    for npz_path in tqdm(npz_paths, desc="logging frames", unit="frame"):
        frame_idx, image_hw, persons = load_frame_npz(npz_path)
        n_in += len(persons)
        persons = dedupe_persons(persons, args.dedupe_iou)
        n_out += len(persons)

        rgb = None
        if cap is not None:
            while next_video_idx <= frame_idx:
                ok, bgr = cap.read()
                if not ok:
                    cap = None
                    break
                if next_video_idx == frame_idx:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                next_video_idx += 1

        log_mesh = not args.no_mesh and faces is not None
        foot = log_frame(frame_idx, image_hw, persons, rgb, faces, log_mesh,
                         T_world_from_rdf=T_world_from_rdf,
                         zed_data=zed_data)
        if foot is not None:
            trajectory.append(foot)
            # Re-log progressively so the trajectory grows live in the viewer
            # and survives a Ctrl-C mid-run. Every 10 frames is plenty.
            if len(trajectory) % 10 == 0:
                log_trajectory_static()

    log_trajectory_static()  # final, complete path

    if cap is not None:
        cap.release()

    dropped = n_in - n_out
    pct = (100.0 * dropped / n_in) if n_in else 0.0
    print(f"[viz] logged {len(npz_paths)} frames, "
          f"{n_out}/{n_in} persons kept ({dropped} duplicates dropped, {pct:.1f}%)")
    print(f"[viz] primary trajectory: {len(trajectory)} points")

    # Make sure all data reaches the viewer before Python exits.
    try:
        rr.flush(blocking=True)
    except TypeError:  # older SDKs: flush() has no kwargs
        rr.flush()
    except AttributeError:
        pass


if __name__ == "__main__":
    main()
