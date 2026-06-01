"""
Skin the MHR mesh with the route-B retargeted rotations and render it in rerun
next to the ZED BODY_38 skeleton.

We do linear-blend skinning ourselves (offline) against the SAME rig the
retarget is built on (mhr_skeleton.json + mhr_mesh.npz from the pyergonomics
MHR content). This sidesteps the QML Mhr_avatar_anim rig, whose joint
convention differs from mhr_skeleton.json (so feeding it our rotations looked
wrong even though the FK skeleton is correct).

LBS: skin_M[j] = PosedGlobal[j] · RestGlobal[j]⁻¹ ; v' = Σ_j w · skin_M[j] · v_rest.

Usage (sam_3d_body env):
    python visualize_retarget_mesh.py --in-dir ~/data/retarget_p2 \
        [--save ~/data/retarget_p2/retarget_mesh.rrd] [--stride 1]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rerun as rr

import retarget_zed_to_mhr as RT
from visualize_zed_bodies import BODY_38_BONES

CONTENT = Path("/home/daniel/code/dev-pyergonomics/src/pyergonomics/ui/view3d/mhr/content")


def fk_positions(G, offsets, parents):
    P = np.zeros((len(parents), 3))
    for j in range(len(parents)):
        p = int(parents[j])
        P[j] = offsets[j] if p < 0 else P[p] + G[p] @ offsets[j]
    return P


def T44(R, t):
    T = np.tile(np.eye(4), (len(R), 1, 1))
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    return T


def inv_rigid(T):
    Ti = np.tile(np.eye(4), (len(T), 1, 1))
    Rt = np.transpose(T[:, :3, :3], (0, 2, 1))
    Ti[:, :3, :3] = Rt
    Ti[:, :3, 3] = -np.einsum("nij,nj->ni", Rt, T[:, :3, 3])
    return Ti


def face_normals(verts, faces):
    n = np.zeros_like(verts)
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    for c in range(3):
        np.add.at(n, faces[:, c], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True); ln[ln == 0] = 1
    return (n / ln).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--rig", type=Path, default=CONTENT / "mhr_skeleton.json")
    ap.add_argument("--mesh", type=Path, default=CONTENT / "mhr_mesh.npz")
    ap.add_argument("--save", type=Path, default=None)
    ap.add_argument("--stride", type=int, default=1, help="log every Nth frame")
    args = ap.parse_args()

    rig = json.loads(args.rig.read_text())
    offsets = np.asarray(rig["offsets"], dtype=np.float64)
    parents = np.asarray(rig["parents"], dtype=np.int64)
    rest = RT.load_mhr_rest(args.rig)
    rest_global, rest_pos = rest["rest_global"], rest["rest_pos"]
    inv_rest = inv_rigid(T44(rest_global, rest_pos))            # (127,4,4)

    m = np.load(args.mesh)
    Vrest = m["vertices"].astype(np.float64)                   # (V,3) native cm
    faces = m["faces"].astype(np.uint32)
    vi = m["vert_indices"].astype(np.int64)
    ji = m["skin_indices"].astype(np.int64)
    w = m["skin_weights"].astype(np.float64)
    Vh = np.concatenate([Vrest, np.ones((len(Vrest), 1))], axis=1)  # (V,4)
    wsum = np.zeros(len(Vrest)); np.add.at(wsum, vi, w); wsum[wsum == 0] = 1.0

    z = np.load(args.in_dir / "zed_bodies.npz", allow_pickle=False)
    fi = z["frame_indices"]; kpw = z["keypoints_3d_world"]; conf = z["keypoint_confidence"]
    row_of = {int(f): i for i, f in enumerate(fi)}
    r = np.load(args.in_dir / "retarget_mhr.npz", allow_pickle=False)
    G_all = r["pred_global_rots"].astype(np.float64)
    frames = [int(f) for f in r["frame"]]

    F = RT.NATIVE_TO_WORLD
    rid = RT.MHR["root"]; PELVIS = RT.ZED["PELVIS"]

    rr.init("zed-vs-mhr-mesh", spawn=args.save is None)
    if args.save:
        rr.save(str(args.save))
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("/world/floor", rr.Boxes3D(centers=[[0, 0, 0]], half_sizes=[[3, 3, 0.002]],
                                      colors=[[60, 90, 60]]), static=True)

    for n, fr in enumerate(frames):
        if n % args.stride or fr not in row_of:
            continue
        rr.set_time("frame", sequence=fr)
        G = G_all[n]
        P = fk_positions(G, offsets, parents)
        skin_M = np.einsum("nij,njk->nik", T44(G, P), inv_rest)      # (127,4,4)

        Me = skin_M[ji]                                             # (K,4,4)
        out = np.zeros((len(Vrest), 3))
        contrib = np.einsum("kij,kj->ki", Me, Vh[vi])[:, :3] * w[:, None]
        np.add.at(out, vi, contrib)
        Vp = out / wsum[:, None]                                    # posed verts (cm)

        Vw = (F @ Vp.T).T * 0.01                                    # -> world m
        root_w = (F @ P[rid]) * 0.01
        Vw += (kpw[row_of[fr]][PELVIS] - root_w)                    # align to ZED pelvis
        rr.log("/world/mhr_mesh",
               rr.Mesh3D(vertex_positions=Vw.astype(np.float32),
                         triangle_indices=faces,
                         vertex_normals=face_normals(Vw.astype(np.float32), faces),
                         albedo_factor=[200, 200, 210]))

        k = kpw[row_of[fr]]; valid = conf[row_of[fr]] > 0.1
        rr.log("/world/zed",
               rr.LineStrips3D([[k[a], k[b]] for a, b in BODY_38_BONES
                                if valid[a] and valid[b]],
                               colors=[[80, 160, 255]], radii=0.01))

    if args.save:
        print(f"[ok] saved {args.save}")


if __name__ == "__main__":
    main()
