"""
Render SMPL body meshes and overlay them on the original video.

Outputs (in visualizations/outputs/):
  smpl_overlay.mp4   – SMPL meshes alpha-composited over video
  smpl_only.mp4      – SMPL render on black background
  frames/            – individual overlay PNGs (optional, set SAVE_FRAMES=True)

Run with:
  conda run -n multi-tram python3 visualizations/render_smpl_overlay.py
"""

import os, io, pickle, json, warnings
import numpy as np
import cv2
import trimesh
import pyrender
from pathlib import Path
from scipy.spatial.transform import Rotation

os.environ["PYOPENGL_PLATFORM"] = "egl"
warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
HERE  = Path(__file__).parent
ROOT  = HERE.parent
RES   = ROOT / "results"
OUT   = HERE / "outputs"
OUT.mkdir(exist_ok=True)

SMPL_PKL   = Path("/home/divyanshu-singh/Divyanshu/graduate-school/Research/tram/data/smpl/SMPL_NEUTRAL.pkl")
VIDEO_PATH = ROOT / "data" / "video_1080p.mp4"

ALPHA      = 0.65          # mesh overlay opacity
FPS        = 24
SAVE_FRAMES = False        # set True to dump per-frame PNGs

# per-person colours (RGBA 0-255)
PERSON_COLORS = [
    (255, 105,  97, 255),   # coral
    ( 97, 168, 255, 255),   # sky blue
    (119, 221, 119, 255),   # mint
    (253, 199,  74, 255),   # gold
    (207, 141, 207, 255),   # lavender
    ( 77, 205, 196, 255),   # teal
    (255, 165,   0, 255),   # orange
    (200, 100, 200, 255),   # purple
]

# ── SMPL loader ───────────────────────────────────────────────────────────────

class _SMPLUnpickler(pickle.Unpickler):
    """Unpickle SMPL .pkl without chumpy installed."""
    class _Ch:
        def __init__(self, *a, **k): self._d = None
        def __setstate__(self, s):
            if isinstance(s, dict):
                self._d = np.array(s.get("x", s.get("r", [])))
            elif isinstance(s, (list, tuple)) and s:
                self._d = np.array(s[0]) if s[0] is not None else np.array([])
            else:
                self._d = np.array([])
        def __array__(self, dtype=None):
            return self._d.astype(dtype) if dtype else self._d

    def find_class(self, module, name):
        if "chumpy" in module:
            return self._Ch
        return super().find_class(module, name)


def load_smpl_model(pkl_path: Path) -> dict:
    with open(pkl_path, "rb") as f:
        raw = f.read()
    data = _SMPLUnpickler(io.BytesIO(raw), encoding="latin1").load()
    out = {}
    for k, v in data.items():
        if hasattr(v, "__array__"):
            out[k] = np.array(v, dtype=np.float32)
        elif hasattr(v, "toarray"):          # sparse matrix
            out[k] = v.toarray().astype(np.float32)
        else:
            out[k] = v
    return out


# ── SMPL forward pass ─────────────────────────────────────────────────────────

def smpl_forward(model: dict,
                 betas: np.ndarray,          # (10,)
                 global_orient: np.ndarray,  # (3,)  axis-angle
                 body_pose: np.ndarray,      # (23,3) axis-angle
                 transl: np.ndarray,         # (3,)
                 ) -> np.ndarray:            # (6890,3) vertices in camera space
    v_tmpl    = model["v_template"]          # (6890,3)
    shapedirs = model["shapedirs"]           # (6890,3,10)
    posedirs  = model["posedirs"]            # (6890,3,207)
    J_reg     = model["J_regressor"]        # (24,6890)
    weights   = model["weights"]            # (6890,24)
    parents   = model["kintree_table"][0].astype(np.int32)  # (24,)
    parents[0] = -1

    # 1. shape blend shapes
    v_shaped = v_tmpl + np.einsum("vij,j->vi", shapedirs, betas)   # (6890,3)

    # 2. rest-pose joints
    J = J_reg @ v_shaped   # (24,3)

    # 3. rotation matrices for all joints (global + body)
    all_aa = np.concatenate([global_orient[np.newaxis], body_pose], axis=0)  # (24,3)
    R = Rotation.from_rotvec(all_aa).as_matrix()                             # (24,3,3)

    # 4. pose blend shapes  (ignore global_orient for blend shapes → use R[1:])
    ident = np.eye(3, dtype=np.float32)
    pose_feat = (R[1:] - ident).reshape(-1)   # (207,)
    v_posed = v_shaped + np.einsum("vij,j->vi",
                                   posedirs.reshape(6890, 3, 207), pose_feat)

    # 5. global joint transforms (forward kinematics)
    G = np.zeros((24, 4, 4), dtype=np.float64)
    for j in range(24):
        T_loc = np.eye(4)
        T_loc[:3, :3] = R[j]
        T_loc[:3,  3] = J[j] - (J[parents[j]] if parents[j] >= 0 else 0)
        G[j] = G[parents[j]] @ T_loc if parents[j] >= 0 else T_loc

    # subtract rest-pose joint contribution: G_final[j] = G[j] @ [[I,-J[j]],[0,1]]
    # equivalent to: G[:3,3] -= G[:3,:3] @ J  (only rotation, not full transform)
    J_rest = np.einsum("jab,jb->ja", G[:, :3, :3], J)      # (24,3)
    G[:, :3, 3] -= J_rest                                   # relative transform

    # 6. LBS skinning
    T = np.einsum("jab,vj->vab", G, weights)               # (6890,4,4)
    v_hom = np.concatenate([v_posed, np.ones((6890, 1))], axis=1)
    verts = np.einsum("vab,vb->va", T, v_hom)[:, :3]

    # 7. apply root translation
    verts += transl

    return verts.astype(np.float32)


# ── load all SMPL params ───────────────────────────────────────────────────────

def load_person_params(pid: int) -> dict | None:
    p = RES / f"3_pose_estimation/person_{pid:03d}/smpl_params_camera.npz"
    if not p.exists():
        return None
    d = np.load(p)
    meta_path = RES / f"3_pose_estimation/person_{pid:03d}/metadata.json"
    meta = json.loads(meta_path.read_text())
    return {
        "poses":        d["poses"],          # (N,23,3)
        "betas":        d["betas"],          # (10,)
        "global_orient":d["global_orient"],  # (N,3)
        "transl":       d["transl"],         # (N,3)
        "frames":       meta["frames"],      # list of frame indices
    }


def load_all_tracks() -> dict:
    tracks = {}
    for i in range(1, 9):
        fp = RES / f"2_tracking/tracks/track_{i}_frames.npy"
        if fp.exists():
            tracks[i] = np.load(fp).tolist()
    return tracks


# ── pyrender scene builder ────────────────────────────────────────────────────

def make_camera(K: np.ndarray, H: int, W: int) -> pyrender.IntrinsicsCamera:
    return pyrender.IntrinsicsCamera(
        fx=float(K[0, 0]), fy=float(K[1, 1]),
        cx=float(K[0, 2]), cy=float(K[1, 2]),
        znear=0.1, zfar=500.0,
    )

# OpenCV → OpenGL: flip Y and Z so camera looks down +Z in world = -Z in GL
_CAM_POSE = np.diag([1.0, -1.0, -1.0, 1.0])


def render_frame(smpl_model, person_params, person_tracks,
                 frame_idx: int, K: np.ndarray, H: int, W: int,
                 renderer: pyrender.OffscreenRenderer):
    """Return (color_rgba uint8 H×W×4, labels list[(u,v,pid)]) for one frame."""
    scene = pyrender.Scene(ambient_light=[0.4, 0.4, 0.4], bg_color=[0, 0, 0, 0])

    # camera
    cam = make_camera(K, H, W)
    scene.add(cam, pose=_CAM_POSE)

    # directional light
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=_CAM_POSE)

    any_mesh = False
    labels = []
    faces = smpl_model["f"].astype(np.int32)
    for pid, params in person_params.items():
        if params is None:
            continue
        track_frames = person_tracks.get(pid, [])
        if frame_idx not in track_frames:
            continue
        local_idx = params["frames"].index(frame_idx)

        verts = smpl_forward(
            smpl_model,
            betas        = params["betas"],
            global_orient= params["global_orient"][local_idx],
            body_pose    = params["poses"][local_idx],
            transl       = params["transl"][local_idx],
        )

        # Project top-of-head vertex for label placement
        head_v = verts[412]  # vertex near top of head
        u = K[0, 0] * head_v[0] / head_v[2] + K[0, 2]
        v = K[1, 1] * head_v[1] / head_v[2] + K[1, 2]
        if 0 < u < W and 0 < v < H:
            labels.append((u, v, pid))

        mesh_tri = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        rgba = PERSON_COLORS[(pid - 1) % len(PERSON_COLORS)]
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[c / 255.0 for c in rgba],
            metallicFactor=0.0,
            roughnessFactor=0.6,
            alphaMode="BLEND",
        )
        scene.add(pyrender.Mesh.from_trimesh(mesh_tri, material=material, smooth=True))
        any_mesh = True

    if not any_mesh:
        return np.zeros((H, W, 4), dtype=np.uint8), []

    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    return color, labels   # (H, W, 4) uint8, list of label positions


def composite(video_frame: np.ndarray, render_rgba: np.ndarray, alpha: float,
              labels: list[tuple] | None = None) -> np.ndarray:
    """Alpha-composite SMPL render over BGR video frame, draw person ID labels."""
    mask  = render_rgba[:, :, 3:4].astype(np.float32) / 255.0  # (H,W,1)
    fg    = render_rgba[:, :, :3].astype(np.float32)[:, :, ::-1]  # RGB→BGR
    bg    = video_frame.astype(np.float32)
    out   = np.clip(bg * (1 - mask * alpha) + fg * (mask * alpha), 0, 255).astype(np.uint8)

    if labels:
        for (u, v, pid) in labels:
            c_rgb = PERSON_COLORS[(pid - 1) % len(PERSON_COLORS)][:3]
            c_bgr = (int(c_rgb[2]), int(c_rgb[1]), int(c_rgb[0]))
            txt = f"P{pid}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            tx, ty = int(u) - tw // 2, max(int(v) - 10, th + 4)
            cv2.rectangle(out, (tx - 3, ty - th - 3), (tx + tw + 3, ty + 3), (0, 0, 0), -1)
            cv2.putText(out, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c_bgr, 2, cv2.LINE_AA)
    return out


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("[1/5] Loading SMPL model …")
    smpl_model = load_smpl_model(SMPL_PKL)
    print(f"      v_template={smpl_model['v_template'].shape}, faces={smpl_model['f'].shape}")

    print("[2/5] Loading SMPL params for all persons …")
    person_params = {pid: load_person_params(pid) for pid in range(1, 9)}
    person_tracks = load_all_tracks()    # {pid: [frame_idx, ...]}

    print("[3/5] Loading camera intrinsics …")
    all_intrinsics = np.load(RES / "1_camera_estimation/cameras/intrinsics.npy")  # (100,3,3)
    # VIMO was given a single focal = mean(fx0,fy0) from frame 0. Use that same
    # focal for rendering so vertex positions project to the correct image locations.
    K0 = all_intrinsics[0]
    vimo_focal = float((K0[0, 0] + K0[1, 1]) / 2.0)
    vimo_K = np.array([[vimo_focal, 0, K0[0, 2]],
                        [0, vimo_focal, K0[1, 2]],
                        [0,          0,         1]], dtype=np.float32)
    print(f"      VIMO focal={vimo_focal:.1f}, center=({K0[0,2]:.0f},{K0[1,2]:.0f})")

    print("[4/5] Opening video …")
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"      {W}×{H}, {total} frames")

    renderer = pyrender.OffscreenRenderer(W, H)

    out_overlay = OUT / "smpl_overlay.mp4"
    out_smpl    = OUT / "smpl_only.mp4"
    fourcc      = cv2.VideoWriter_fourcc(*"mp4v")
    vw_overlay  = cv2.VideoWriter(str(out_overlay), fourcc, FPS, (W, H))
    vw_smpl     = cv2.VideoWriter(str(out_smpl),    fourcc, FPS, (W, H))

    if SAVE_FRAMES:
        (OUT / "frames").mkdir(exist_ok=True)

    print("[5/5] Rendering frames …")
    for fi in range(min(total, 100)):
        ret, frame_bgr = cap.read()
        if not ret:
            break

        render_rgba, labels = render_frame(smpl_model, person_params, person_tracks,
                                           fi, vimo_K, H, W, renderer)

        overlay = composite(frame_bgr, render_rgba, ALPHA, labels)
        smpl_bgr = cv2.cvtColor(render_rgba[:, :, :3], cv2.COLOR_RGB2BGR)

        vw_overlay.write(overlay)
        vw_smpl.write(smpl_bgr)

        if SAVE_FRAMES:
            cv2.imwrite(str(OUT / "frames" / f"frame_{fi:06d}.png"), overlay)

        if fi % 10 == 0:
            visible = sum(1 for pid, p in person_params.items()
                          if p and fi in person_tracks.get(pid, []))
            print(f"  frame {fi:3d}/100  |  {visible} people rendered")

    cap.release()
    renderer.delete()
    vw_overlay.release()
    vw_smpl.release()

    # Re-encode with ffmpeg for broad compatibility
    for src, dst_name in [(out_overlay, "smpl_overlay_h264.mp4"),
                          (out_smpl,    "smpl_only_h264.mp4")]:
        dst = OUT / dst_name
        os.system(f'ffmpeg -y -i "{src}" -c:v libx264 -crf 18 -pix_fmt yuv420p "{dst}" -loglevel error')
        if dst.exists():
            src.unlink()
            dst.rename(src)

    print(f"\nDone!")
    print(f"  overlay → {out_overlay}")
    print(f"  smpl    → {out_smpl}")


if __name__ == "__main__":
    main()
