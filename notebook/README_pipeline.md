# SAM3D Body + ZED Pipeline — Status & Import Contract

Summary of what's been built and learned. Read this before deciding how to
integrate into pyergonomics.

## What we have now

Three CLI scripts in `notebook/`, plus their corresponding visualizers.
Conceptually one pipeline:

```
ZED SVO  ──┬──→ extract_zed_floor.py ──→ zed_floor.json
           │                              zed_bodies.npz (optional, BODY_38)
           │
           └──→ ffmpeg ──→ frames/*.jpg
                              │
                              └──→ process_video_frames.py ──→ frames/*.npz
                                                                              │
                                                                              └→ export_for_pyergonomics.py
                                                                                       │
                                                                                       └──→ sam3d_export.npz
```

The `sam3d_export.npz` is the single file that pyergonomics needs. Everything
else is intermediate / debugging.

## Coordinate systems (this is the part that mattered most)

| Source                 | Convention           | Axes                            |
|------------------------|----------------------|---------------------------------|
| SAM3D Body 3D output   | OpenCV RDF           | +X right, +Y down, +Z forward   |
| ZED camera (`Z_UP`)    | RIGHT_HANDED_Z_UP    | +X right, +Y forward, +Z up     |
| World (target)         | Floor-aligned, Z up  | +X/+Y on the floor, +Z up       |

`zed_floor.json` carries a precomputed **4×4 `T_world_from_rdf`** that maps
SAM3D's RDF coordinates directly to world. `export_for_pyergonomics.py`
applies it before writing `keypoints_3d` to the export. So consumers of
`sam3d_export.npz` never see RDF — they just get world-frame Z-up metres
with floor at `z = 0`.

## Two non-obvious calibration findings

1. **The model was trained assuming a centered principal point.** Passing
   off-center calibrated intrinsics (`cx ≈ 583` for our ZED at HD720) made
   2D keypoints land ~57 px off the body. The fix is to **pad the input
   image** before inference (`--recenter-principal-point`) so the principal
   point ends up at the image center; intrinsics become centered too. 2D
   coords are de-padded back to the original resolution before saving, so
   the npz output and the source video stay aligned. 3D output is
   unaffected by padding.
2. **`USE_INTRIN_CENTER=True` doesn't actually fix it.** The model has a
   config flag with that name, but the network weights weren't trained for
   it — flipping the flag puts the camera head into an inconsistent
   regime. Padding is the correct workaround.

## ZED floor detection

`extract_zed_floor.py` opens the SVO with `RIGHT_HANDED_Z_UP +
set_floor_as_origin`, calls `find_floor_plane`, and computes a
`R, t, T_world_from_rdf` consistent with how pyergonomics's existing
`importers/zed.py` defines its world frame (Z up, floor at z=0, world Y =
camera forward projected onto the floor). The math is copied directly from
`zed.py:_create_floor_transform` — no convention mismatch.

For our test SVO: floor is at d=1.724 m below the camera — matches the
tripod height.

## Known limitations

- **No persistent person identity in SAM3D output.** Each frame's
  detections are independent. The `person` column in the export is a
  per-frame index (0, 1, ... within the frame), NOT a track ID. If
  pyergonomics needs stable IDs, either:
  - run `track_persons.py` first (boxmot, separate env — needs setup) and
    re-export, **or**
  - the importer can run norfair itself the way the existing
    `_sam3d_common.build_rows` does.
- **ZED's `body.keypoint_2d` had ~5–10 px noise** on our test footage.
  Partially attributable to a known left-eye camera defect in this
  recording. Worth re-checking on cleaner footage before drawing
  conclusions about ZED 2D accuracy.
- **SAM3D's primary-person trajectory swaps identity** when two people
  pass at similar depths. Real but expected — same root cause as bullet 1.

## Bare-minimum import contract

What pyergonomics needs to do to consume `sam3d_export.npz`:

```python
import numpy as np

z = np.load("sam3d_export.npz", allow_pickle=False)

# Per-(frame, person) rows, stacked. N total rows.
frame      = z["frame"]         # (N,) int64
person     = z["person"]        # (N,) int64  — per-frame index, NOT a track id
bbox       = z["bbox_xywh"]     # (N, 4) float32  — top-left + w/h, image px
kp2d       = z["keypoints_2d"]  # (N, 70, 2) float32  — image px, MHR70 order
kp3d_world = z["keypoints_3d"]  # (N, 70, 3) float32  — WORLD frame (Z up, m)
conf       = z["keypoint_confidence"]  # (N, 70) float32 — synthetic, all 1.0

# Scalars / metadata.
fps        = float(z["fps"])
W, H       = int(z["image_width"]), int(z["image_height"])
coord_sys  = str(z["coordinate_system_3d"])   # "world_z_up" or "camera_rdf"
fx, fy     = float(z["intrinsics_fx"]), float(z["intrinsics_fy"])
cx, cy     = float(z["intrinsics_cx"]), float(z["intrinsics_cy"])

# Extrinsics — present only when zed_floor.json was used during export.
if "T_world_from_rdf" in z.files:
    T = z["T_world_from_rdf"]    # (4, 4) float64 — SAM3D-RDF → world
    floor = z["floor_plane"]     # (4,) float64  — ax+by+cz+d=0 in ZED camera ZUP
```

### Things to verify in the importer

- `coordinate_system_3d == "world_z_up"`. If it's `camera_rdf`, no floor
  was applied during export — `keypoints_3d` are in SAM3D-RDF camera
  coords, not world. Either re-export with `--floor` or apply the
  fallback fixed rotation pyergonomics already uses.
- `image_width` and `image_height` are the **original** input resolution
  (1280×720 for HD720 ZED), **not** the padded internal resolution the
  inference saw. All 2D coords are in original-image pixels.
- `fps` is what you passed to `export_for_pyergonomics.py --fps N`; it's
  `0.0` if you didn't. ZED SVO is typically 30 fps; the JPGs extracted
  via ffmpeg preserve that.

### Convention summary (so the importer doesn't have to re-derive)

- `bbox_xywh` matches `from_zed`'s `x, y, w, h` convention.
- `keypoints_3d` in `world_z_up` mode matches `from_zed`'s
  `keypoints_3d` (same world frame: Z up, floor at z=0, world Y = camera
  forward projected onto the floor).
- `keypoints_2d` are in the original image pixel space — identical to
  what's drawn over the video in the source MP4.
- Skeleton is MHR70 (70 keypoints). For comparison with ZED BODY_34/38,
  see the existing `mhr_to_standard.py` for the 23-joint subset mapping.

## What's still to decide before pyergonomics integration

1. Track IDs: run boxmot in extract step, or let pyergonomics do norfair?
   (Both currently work; the existing `from_sam3d_video` does norfair on
   bboxes, which is fine.)
2. Should the floor transform be applied in the export, or stored
   alongside (raw RDF + transform) and applied in pyergonomics? Current
   choice: applied in export, but `T_world_from_rdf` is preserved in the
   file so it can be re-derived/undone.
3. Whether to extend `from_sam3d_video` to optionally take a
   `zed_floor_json` and read floor-aligned coords from the raw npz
   directly, or to ingest the pre-baked `sam3d_export.npz`.
