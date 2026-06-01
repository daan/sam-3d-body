"""
Retarget ZED BODY_38 motion onto the MHR rig (to drive the MHR avatar/mesh).

Goal: visualize a ZED-captured body with the MHR mesh. ZED gives a 38-joint
skeleton (positions + per-joint orientations); MHR is a 127-joint articulated
rig that drives a parametric mesh. Retargeting = (1) joint correspondence,
(2) per-joint rest-pose offset, (3) a shared world frame.

This module owns piece (1) — the authoritative correspondence table — and
scaffolds (2) with two routes:

  Route A — orientation copy + rest offset.
      Uses ZED's per-joint orientations directly. Solve a per-joint offset
      from one calibration pose (e.g. a held T-pose), then at runtime
      G_mhr_j = offset_j @ G_zed_j. Convention-sensitive; needs ZED
      orientations extracted (see extract_zed_floor.py --> add
      local_orientation_per_joint / global_root_orientation).

  Route B — bone-direction (look-at) from positions.
      Points each MHR bone along the world-space direction of the
      corresponding ZED bone (zed_child - zed_joint). Uses only ZED
      *positions* (reliable, already extracted). Rest offset falls out
      naturally; twist about each bone is underdetermined (needs a
      secondary-axis hint). Good for a first visual.

The OUTPUT of either route is a set of MHR global joint rotations
(127, 3, 3) with unmapped joints left at rest. That feeds straight into the
existing pyergonomics MHR avatar path (MhrPoseState -> Mhr_avatar_anim),
sourcing rotations from here instead of SAM3D's pred_global_rots.

Frame note: ZED data and MHR both land in the shared Z-up world frame the
floor calibration established (see zed_floor.json). ZED per-joint LOCAL
orientations are relative (frame-independent); only the ROOT global
orientation needs the floor rotation R applied — the same root-transform
identity used in ui/mhr_pose_state.py.

STATUS: correspondence table + ZED FK + offset solve are implemented and
unit-runnable. Route-A frame retarget is a thin wrapper (runnable once you
have ZED globals). Route-B and the MHR-local conversion against the real rig
rest are marked TODO — they need the MHR rig rest (mhr_skeleton.json
prerotations/offsets) wired in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# ZED BODY_38 keypoint indices (Stereolabs SDK).
# ---------------------------------------------------------------------------

ZED = {
    "PELVIS": 0, "SPINE_1": 1, "SPINE_2": 2, "SPINE_3": 3, "NECK": 4,
    "NOSE": 5, "LEFT_EYE": 6, "RIGHT_EYE": 7, "LEFT_EAR": 8, "RIGHT_EAR": 9,
    "LEFT_CLAVICLE": 10, "RIGHT_CLAVICLE": 11,
    "LEFT_SHOULDER": 12, "RIGHT_SHOULDER": 13,
    "LEFT_ELBOW": 14, "RIGHT_ELBOW": 15,
    "LEFT_WRIST": 16, "RIGHT_WRIST": 17,
    "LEFT_HIP": 18, "RIGHT_HIP": 19,
    "LEFT_KNEE": 20, "RIGHT_KNEE": 21,
    "LEFT_ANKLE": 22, "RIGHT_ANKLE": 23,
    "LEFT_BIG_TOE": 24, "RIGHT_BIG_TOE": 25,
    "LEFT_SMALL_TOE": 26, "RIGHT_SMALL_TOE": 27,
    "LEFT_HEEL": 28, "RIGHT_HEEL": 29,
    "LEFT_HAND_THUMB_4": 30, "RIGHT_HAND_THUMB_4": 31,
    "LEFT_HAND_INDEX_1": 32, "RIGHT_HAND_INDEX_1": 33,
    "LEFT_HAND_MIDDLE_4": 34, "RIGHT_HAND_MIDDLE_4": 35,
    "LEFT_HAND_PINKY_1": 36, "RIGHT_HAND_PINKY_1": 37,
}

# ZED BODY_38 parent hierarchy (child -> parent), derived from the bone list
# in visualize_zed_bodies.py. Used to compose local orientations into globals.
# PELVIS is the root (parent -1).
ZED_PARENT = {
    ZED["PELVIS"]: -1,
    ZED["SPINE_1"]: ZED["PELVIS"], ZED["SPINE_2"]: ZED["SPINE_1"],
    ZED["SPINE_3"]: ZED["SPINE_2"], ZED["NECK"]: ZED["SPINE_3"],
    ZED["NOSE"]: ZED["NECK"],
    ZED["LEFT_EYE"]: ZED["NOSE"], ZED["RIGHT_EYE"]: ZED["NOSE"],
    ZED["LEFT_EAR"]: ZED["LEFT_EYE"], ZED["RIGHT_EAR"]: ZED["RIGHT_EYE"],
    ZED["LEFT_CLAVICLE"]: ZED["NECK"], ZED["RIGHT_CLAVICLE"]: ZED["NECK"],
    ZED["LEFT_SHOULDER"]: ZED["LEFT_CLAVICLE"],
    ZED["RIGHT_SHOULDER"]: ZED["RIGHT_CLAVICLE"],
    ZED["LEFT_ELBOW"]: ZED["LEFT_SHOULDER"],
    ZED["RIGHT_ELBOW"]: ZED["RIGHT_SHOULDER"],
    ZED["LEFT_WRIST"]: ZED["LEFT_ELBOW"], ZED["RIGHT_WRIST"]: ZED["RIGHT_ELBOW"],
    ZED["LEFT_HIP"]: ZED["PELVIS"], ZED["RIGHT_HIP"]: ZED["PELVIS"],
    ZED["LEFT_KNEE"]: ZED["LEFT_HIP"], ZED["RIGHT_KNEE"]: ZED["RIGHT_HIP"],
    ZED["LEFT_ANKLE"]: ZED["LEFT_KNEE"], ZED["RIGHT_ANKLE"]: ZED["RIGHT_KNEE"],
    ZED["LEFT_BIG_TOE"]: ZED["LEFT_ANKLE"], ZED["RIGHT_BIG_TOE"]: ZED["RIGHT_ANKLE"],
    ZED["LEFT_SMALL_TOE"]: ZED["LEFT_ANKLE"], ZED["RIGHT_SMALL_TOE"]: ZED["RIGHT_ANKLE"],
    ZED["LEFT_HEEL"]: ZED["LEFT_ANKLE"], ZED["RIGHT_HEEL"]: ZED["RIGHT_ANKLE"],
    ZED["LEFT_HAND_THUMB_4"]: ZED["LEFT_WRIST"], ZED["RIGHT_HAND_THUMB_4"]: ZED["RIGHT_WRIST"],
    ZED["LEFT_HAND_INDEX_1"]: ZED["LEFT_WRIST"], ZED["RIGHT_HAND_INDEX_1"]: ZED["RIGHT_WRIST"],
    ZED["LEFT_HAND_MIDDLE_4"]: ZED["LEFT_WRIST"], ZED["RIGHT_HAND_MIDDLE_4"]: ZED["RIGHT_WRIST"],
    ZED["LEFT_HAND_PINKY_1"]: ZED["LEFT_WRIST"], ZED["RIGHT_HAND_PINKY_1"]: ZED["RIGHT_WRIST"],
}


# ---------------------------------------------------------------------------
# MHR rig joint ids (from ui/view3d/mhr/content/mhr_skeleton.json, 127 joints).
# Only the joints we actually retarget are named here.
# ---------------------------------------------------------------------------

MHR = {
    "root": 1,
    "c_spine0": 34, "c_spine1": 35, "c_spine2": 36, "c_spine3": 37,
    "c_neck": 110, "c_head": 113,
    "l_clavicle": 74, "l_uparm": 75, "l_lowarm": 76, "l_wrist": 78,
    "r_clavicle": 38, "r_uparm": 39, "r_lowarm": 40, "r_wrist": 42,
    "l_upleg": 2, "l_lowleg": 3, "l_foot": 4, "l_ball": 8,
    "r_upleg": 18, "r_lowleg": 19, "r_foot": 20, "r_ball": 24,
    # middle-finger base: a stable "hand direction" child for the wrist.
    "l_middle1": 88, "r_middle1": 52,
}

MHR_N_JOINTS = 127


# ---------------------------------------------------------------------------
# The correspondence table — the core artifact.
#
# Each row maps one MHR rig joint to the ZED keypoint that sits at it
# (`zed_at`, used for position + orientation copy) and the ZED keypoint that
# defines its bone direction (`zed_to`, the primary child, used by route B's
# look-at). `zed_to=None` => leaf joint (no bone direction).
#
# Coverage is the body chain (23 joints). MHR's extra DOF — fingers, jaw/eyes,
# subdivided spine twists, foot detail — have no ZED source and stay at rest.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JointMap:
    mhr: str            # MHR joint name
    zed_at: str         # ZED kp at this joint (position + orientation)
    zed_to: str | None  # ZED kp giving bone direction (route B); None=leaf
    note: str = ""

    @property
    def mhr_id(self) -> int:
        return MHR[self.mhr]

    @property
    def zed_at_id(self) -> int:
        return ZED[self.zed_at]

    @property
    def zed_to_id(self) -> int | None:
        return None if self.zed_to is None else ZED[self.zed_to]


CORRESPONDENCE: list[JointMap] = [
    # spine / head — MHR has 4 spine joints, ZED 3, so c_spine3 is approximate.
    JointMap("root",     "PELVIS",  "SPINE_1"),
    JointMap("c_spine0", "SPINE_1", "SPINE_2"),
    JointMap("c_spine1", "SPINE_2", "SPINE_3"),
    JointMap("c_spine2", "SPINE_3", "NECK"),
    JointMap("c_spine3", "SPINE_3", "NECK", "no direct ZED joint; shares SPINE_3->NECK"),
    JointMap("c_neck",   "NECK",    "NOSE"),
    JointMap("c_head",   "NOSE",    None, "head orientation approximate (nose/ears)"),
    # left arm
    JointMap("l_clavicle", "LEFT_CLAVICLE", "LEFT_SHOULDER"),
    JointMap("l_uparm",    "LEFT_SHOULDER", "LEFT_ELBOW"),
    JointMap("l_lowarm",   "LEFT_ELBOW",    "LEFT_WRIST"),
    JointMap("l_wrist",    "LEFT_WRIST",    "LEFT_HAND_MIDDLE_4", "twist underdetermined"),
    # right arm
    JointMap("r_clavicle", "RIGHT_CLAVICLE", "RIGHT_SHOULDER"),
    JointMap("r_uparm",    "RIGHT_SHOULDER", "RIGHT_ELBOW"),
    JointMap("r_lowarm",   "RIGHT_ELBOW",    "RIGHT_WRIST"),
    JointMap("r_wrist",    "RIGHT_WRIST",    "RIGHT_HAND_MIDDLE_4", "twist underdetermined"),
    # left leg
    JointMap("l_upleg",  "LEFT_HIP",     "LEFT_KNEE"),
    JointMap("l_lowleg", "LEFT_KNEE",    "LEFT_ANKLE"),
    JointMap("l_foot",   "LEFT_ANKLE",   "LEFT_BIG_TOE"),
    JointMap("l_ball",   "LEFT_BIG_TOE", None),
    # right leg
    JointMap("r_upleg",  "RIGHT_HIP",     "RIGHT_KNEE"),
    JointMap("r_lowleg", "RIGHT_KNEE",    "RIGHT_ANKLE"),
    JointMap("r_foot",   "RIGHT_ANKLE",   "RIGHT_BIG_TOE"),
    JointMap("r_ball",   "RIGHT_BIG_TOE", None),
]


# ---------------------------------------------------------------------------
# Quaternion / rotation helpers (xyzw order, matching ZED's sl.Orientation).
# ---------------------------------------------------------------------------

def quat_to_mat(q_xyzw: np.ndarray) -> np.ndarray:
    """(...,4) xyzw quaternion -> (...,3,3) rotation matrix."""
    q = np.asarray(q_xyzw, dtype=np.float64)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    n = x * x + y * y + z * z + w * w
    s = np.where(n > 0, 2.0 / n, 0.0)
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    m = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    m[..., 0, 0] = 1 - (yy + zz); m[..., 0, 1] = xy - wz;       m[..., 0, 2] = xz + wy
    m[..., 1, 0] = xy + wz;       m[..., 1, 1] = 1 - (xx + zz); m[..., 1, 2] = yz - wx
    m[..., 2, 0] = xz - wy;       m[..., 2, 1] = yz + wx;       m[..., 2, 2] = 1 - (xx + yy)
    return m


def zed_local_to_global(local_mats: np.ndarray) -> np.ndarray:
    """Compose ZED per-joint LOCAL rotations into GLOBAL rotations along the
    BODY_38 hierarchy. `local_mats` is (38, 3, 3) in ZED camera frame.

    Returns (38, 3, 3) global rotations (still in ZED camera frame; apply the
    floor R afterwards to reach world if needed)."""
    g = np.tile(np.eye(3), (local_mats.shape[0], 1, 1))
    # ZED_PARENT keys aren't guaranteed topologically ordered; resolve by
    # walking parents to root per joint (38 joints, cheap).
    for j in range(local_mats.shape[0]):
        chain = []
        k = j
        while k != -1 and k in ZED_PARENT:
            chain.append(k)
            k = ZED_PARENT[k]
        acc = np.eye(3)
        for node in reversed(chain):       # root-first
            acc = acc @ local_mats[node]
        g[j] = acc
    return g


# ---------------------------------------------------------------------------
# Route A — orientation copy + per-joint rest offset.
# ---------------------------------------------------------------------------

def solve_offsets(
    zed_global_calib: np.ndarray,   # (38, 3, 3) ZED global rots, WORLD frame
    mhr_global_calib: np.ndarray,   # (127, 3, 3) MHR global rots, WORLD frame
) -> dict[int, np.ndarray]:
    """Per-joint rest offset from ONE calibration pose (e.g. a held T-pose).

    For each mapped joint, offset_j = G_mhr_j @ G_zed_j^T, so that at runtime
    G_mhr_j = offset_j @ G_zed_j reproduces the MHR orientation when the ZED
    pose matches the calibration pose. Returns {mhr_id: offset (3,3)}.

    Both inputs MUST be in the same world frame and the SAME body pose. Use
    the held calibration frame from the representative capture.
    """
    offsets: dict[int, np.ndarray] = {}
    for jm in CORRESPONDENCE:
        gz = zed_global_calib[jm.zed_at_id]
        gm = mhr_global_calib[jm.mhr_id]
        offsets[jm.mhr_id] = gm @ gz.T
    return offsets


def retarget_orientation(
    zed_global_world: np.ndarray,       # (38, 3, 3) ZED global rots, WORLD
    offsets: dict[int, np.ndarray],
    rest_global: np.ndarray | None = None,  # (127,3,3) MHR rest globals
) -> np.ndarray:
    """Route A: produce (127, 3, 3) MHR global rotations for one frame.

    Mapped joints get offset_j @ G_zed_j; unmapped joints keep `rest_global`
    (identity if None). Convert to MHR LOCAL via the rig parents before
    feeding the avatar (see mhr_local_from_global, TODO)."""
    out = (rest_global.copy() if rest_global is not None
           else np.tile(np.eye(3), (MHR_N_JOINTS, 1, 1)))
    for jm in CORRESPONDENCE:
        out[jm.mhr_id] = offsets[jm.mhr_id] @ zed_global_world[jm.zed_at_id]
    return out


# ---------------------------------------------------------------------------
# Route B — bone-direction (look-at) from positions.
#
# Validated on ~/data/retarget_p2: ZED↔MHR bone directions agree to ~9° median
# (route A / full-orientation copy fails at ~78° because ZED twist is noisy).
# Twist is left at rest (the user's choice); add secondary-axis hints later if
# forearm pronation / spine axial rotation matters.
# ---------------------------------------------------------------------------

# Rig child of each mapped joint = the joint the bone points to (defines the
# bone we steer). Leaves (l_ball, r_ball, wrists, c_head) have no child here.
MHR_BONE_CHILD = {
    MHR["root"]: MHR["c_spine0"],
    MHR["c_spine0"]: MHR["c_spine1"], MHR["c_spine1"]: MHR["c_spine2"],
    MHR["c_spine2"]: MHR["c_spine3"], MHR["c_spine3"]: MHR["c_neck"],
    MHR["c_neck"]: MHR["c_head"],
    MHR["l_clavicle"]: MHR["l_uparm"], MHR["l_uparm"]: MHR["l_lowarm"],
    MHR["l_lowarm"]: MHR["l_wrist"], MHR["l_wrist"]: MHR["l_middle1"],
    MHR["r_clavicle"]: MHR["r_uparm"], MHR["r_uparm"]: MHR["r_lowarm"],
    MHR["r_lowarm"]: MHR["r_wrist"], MHR["r_wrist"]: MHR["r_middle1"],
    MHR["l_upleg"]: MHR["l_lowleg"], MHR["l_lowleg"]: MHR["l_foot"],
    MHR["l_foot"]: MHR["l_ball"],
    MHR["r_upleg"]: MHR["r_lowleg"], MHR["r_lowleg"]: MHR["r_foot"],
    MHR["r_foot"]: MHR["r_ball"],
}


# Structural MHR-rig-native -> pye z-up scene/world rotation: the avatar's
# Mhr_avatar_anim node cycling (Qt.quaternion(0.5,0.5,0.5,0.5)): native X->Y,
# Y->Z, Z->X. Since ZED world is also z-up, this is the rig-native -> ZED-world
# frame map F for route B. (Do NOT fit F from SAM3D pred_global_rots: SAM3D's
# MHR rig differs from pyergonomics' mhr_skeleton.json — bone dirs ~139° off,
# legs ~180° — so a SAM3D-fit frame is contaminated.)
NATIVE_TO_WORLD = np.array([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
], dtype=np.float64)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal (twist-free) rotation matrix mapping unit vector a -> unit b."""
    a, b = _unit(np.asarray(a, float)), _unit(np.asarray(b, float))
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -1 + 1e-8:                       # antiparallel: 180° about any ⟂ axis
        axis = _unit(np.cross(a, [1.0, 0, 0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = _unit(np.cross(a, [0, 1.0, 0]))
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def _orthoframe(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Orthonormal frame (columns) from primary axis a and secondary b."""
    a = _unit(a)
    b = _unit(b - np.dot(b, a) * a)
    c = np.cross(a, b)
    return np.stack([a, b, c], axis=1)


def frame_align(a1, a2, b1, b2) -> np.ndarray:
    """Rotation R with R·a1≈b1 and R·a2≈b2 (a's primary, b's secondary).
    Used to set a full joint orientation from two axes (e.g. the pelvis from
    spine-up + hip-line) instead of a single bone direction."""
    A = _orthoframe(a1, a2)
    B = _orthoframe(b1, b2)
    return B @ A.T


# Torso joints get a full 2-axis orientation (bone-to-child + a lateral
# reference) instead of single-bone look-at. Single-bone leaves twist about the
# bone at rest, which is fine for limbs but wrong for the pelvis/spine: their
# *facing* drives the legs and the shoulders, so an at-rest facing shows up as
# a ~90° rotation of everything hanging below them.
#   mhr_id -> ((zed_lat_a, zed_lat_b), (rig_lat_a, rig_lat_b))
# lateral = hip line for the pelvis, shoulder line for the spine.
TWO_AXIS = {
    MHR["root"]:     ((ZED["LEFT_HIP"], ZED["RIGHT_HIP"]),
                      (MHR["l_upleg"], MHR["r_upleg"])),
    MHR["c_spine0"]: ((ZED["LEFT_SHOULDER"], ZED["RIGHT_SHOULDER"]),
                      (MHR["l_uparm"], MHR["r_uparm"])),
    MHR["c_spine1"]: ((ZED["LEFT_SHOULDER"], ZED["RIGHT_SHOULDER"]),
                      (MHR["l_uparm"], MHR["r_uparm"])),
    MHR["c_spine2"]: ((ZED["LEFT_SHOULDER"], ZED["RIGHT_SHOULDER"]),
                      (MHR["l_uparm"], MHR["r_uparm"])),
    MHR["c_spine3"]: ((ZED["LEFT_SHOULDER"], ZED["RIGHT_SHOULDER"]),
                      (MHR["l_uparm"], MHR["r_uparm"])),
}


def kabsch(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Best-fit rotation R (3x3, det +1) minimizing ||R·src - dst|| over rows
    of unit vectors. Used to anchor the MHR-pose frame to the ZED world frame
    from one calibration pose."""
    H = src.T @ dst
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    return Vt.T @ D @ U.T


def load_mhr_rest(skeleton_json_path):
    """Load the MHR rig and compute its rest pose by forward kinematics from
    `prerotations_xyzw` + `offsets`. Returns dict with:
        parents       (127,)   int
        rest_global   (127,3,3) rest joint rotations (rig/native frame)
        rest_pos      (127,3)   rest joint positions
        rest_bone_dir {mhr_id: unit bone dir} for joints in MHR_BONE_CHILD
    """
    import json
    d = json.loads(open(skeleton_json_path).read())
    parents = np.asarray(d["parents"], dtype=np.int64)
    prerot = quat_to_mat(np.asarray(d["prerotations_xyzw"], dtype=np.float64))
    offsets = np.asarray(d["offsets"], dtype=np.float64)
    n = len(parents)
    G = np.tile(np.eye(3), (n, 1, 1))
    P = np.zeros((n, 3))
    for j in range(n):                      # parents are topologically ordered
        p = int(parents[j])
        if p < 0:
            G[j] = prerot[j]; P[j] = offsets[j]
        else:
            G[j] = G[p] @ prerot[j]; P[j] = P[p] + G[p] @ offsets[j]
    # bone_dir: rest bone direction in the GLOBAL rest frame.
    # bone_local: the same bone direction in the joint's LOCAL frame
    # (= G_rest[j]ᵀ · bone_dir). Derived from positions, not offset[child],
    # because some bones span an intermediate twist joint (e.g. l_lowarm ->
    # l_wrist via l_wrist_twist), so offset[child] is not the bone vector.
    # A joint's posed bone direction is then global_rot[j] @ bone_local[j].
    bone_dir = {jid: _unit(P[c] - P[jid]) for jid, c in MHR_BONE_CHILD.items()}
    bone_local = {jid: _unit(G[jid].T @ (P[c] - P[jid]))
                  for jid, c in MHR_BONE_CHILD.items()}
    return {"parents": parents, "rest_global": G, "rest_pos": P,
            "rest_bone_local": bone_local, "rest_bone_dir": bone_dir}


def solve_frame_align(zed_kp_world, mhr_global_calib, rest, offsets_child=None):
    """Kabsch-fit the rotation F mapping the MHR pose frame -> ZED world frame,
    from one calibration pose. Uses bone directions of the mapped joints:
        mhr_dir_j (pose frame) = mhr_global_calib[j] @ rig_offset_to_child
        zed_dir_j (world)      = kp[zed_to] - kp[zed_at]
        F ≈ argmin Σ ||F·mhr_dir - zed_dir||
    `mhr_global_calib` is pred_global_rots at the calibration frame.
    Returns (F (3,3), residual_deg)."""
    src, dst = [], []
    for jm in CORRESPONDENCE:
        c = MHR_BONE_CHILD.get(jm.mhr_id)
        if c is None or jm.zed_to is None:
            continue
        # posed bone dir in the pose frame = global_rot @ bone_local.
        md = _unit(mhr_global_calib[jm.mhr_id] @ rest["rest_bone_local"][jm.mhr_id])
        zd = _unit(zed_kp_world[jm.zed_to_id] - zed_kp_world[jm.zed_at_id])
        src.append(md); dst.append(zd)
    src, dst = np.asarray(src), np.asarray(dst)
    F = kabsch(src, dst)
    res = np.degrees(np.arccos(np.clip(np.sum((F @ src.T).T * dst, axis=1), -1, 1)))
    return F, float(res.mean())


def retarget_bonedir(zed_kp_world, rest, F=NATIVE_TO_WORLD):
    """Route B: produce (127,3,3) MHR global rotations (rig-native frame) for
    one frame, steering each mapped bone onto its ZED direction with twist at
    rest. `F` is the rig-native->ZED-world rotation; defaults to the avatar's
    structural cycling NATIVE_TO_WORLD (the principled value — confirm on the
    avatar). Output feeds the pyergonomics avatar as pred_global_rots.

    For each mapped joint j with a child: the target bone direction in the
    pose frame is Fᵀ·(zed_to - zed_at); rotate the joint's rest bone direction
    onto it (minimal rotation) and apply to the rest global. Output feeds the
    avatar as pred_global_rots (then mhr_local_from_global -> MhrPoseState)."""
    parents = rest["parents"]
    Gr = rest["rest_global"]
    Ft = F.T
    n = len(parents)

    # Rest LOCAL rotation of each joint (relative to its parent).
    Lrest = np.empty_like(Gr)
    for j in range(n):
        p = int(parents[j])
        Lrest[j] = Gr[j] if p < 0 else Gr[p].T @ Gr[j]

    corr = {jm.mhr_id: jm for jm in CORRESPONDENCE}
    G = Gr.copy()
    # Parent-first pass (rig joints are topologically ordered). Each joint is
    # based on its POSED parent times its rest-local rotation, so unmapped
    # joints (head, feet sub-joints, wrist-twist, fingers) ride along with the
    # chain instead of being stuck at absolute rest. Mapped bones are then
    # aimed; the torso gets a full 2-axis facing.
    for j in range(n):
        p = int(parents[j])
        G0 = Gr[j] if p < 0 else G[p] @ Lrest[j]
        jm = corr.get(j)
        if j in TWO_AXIS:
            # Torso: absolute facing from bone-up + lateral reference.
            up_w = zed_kp_world[jm.zed_to_id] - zed_kp_world[jm.zed_at_id]
            (za, zb), (ra, rb) = TWO_AXIS[j]
            lat_w = zed_kp_world[zb] - zed_kp_world[za]
            up_local = rest["rest_bone_local"][j]
            lat_local = _unit(Gr[j].T @ (rest["rest_pos"][rb] - rest["rest_pos"][ra]))
            G[j] = frame_align(up_local, lat_local, Ft @ up_w, Ft @ lat_w)
        elif jm is not None and j in MHR_BONE_CHILD and jm.zed_to is not None:
            # Limb/extremity bone: aim it at the ZED child within the posed
            # parent frame (minimal rotation → twist stays at rest, continuous).
            tgt = Ft @ _unit(zed_kp_world[jm.zed_to_id] - zed_kp_world[jm.zed_at_id])
            cur = G0 @ rest["rest_bone_local"][j]
            G[j] = rotation_between(cur, tgt) @ G0
        else:
            G[j] = G0                         # follow posed parent at rest-local
    return G


def mhr_local_from_global(global_mats: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """Convert (127,3,3) MHR GLOBAL rotations to LOCAL (relative to parent),
    matching MhrPoseState._globals_to_local_quats:
        local[j] = global[parent].T @ global[j]   (root: global[j]).
    Provided so the retarget output can be fed straight to the avatar.
    """
    local = np.empty_like(global_mats)
    for j in range(global_mats.shape[0]):
        p = int(parents[j])
        local[j] = global_mats[j] if p < 0 else global_mats[p].T @ global_mats[j]
    return local


__all__ = [
    "ZED", "ZED_PARENT", "MHR", "MHR_N_JOINTS", "CORRESPONDENCE", "JointMap",
    "MHR_BONE_CHILD",
    "quat_to_mat", "zed_local_to_global",
    "solve_offsets", "retarget_orientation",
    "rotation_between", "kabsch", "load_mhr_rest", "solve_frame_align",
    "retarget_bonedir", "mhr_local_from_global",
]
