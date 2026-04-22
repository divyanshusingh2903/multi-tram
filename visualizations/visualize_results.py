"""
Visualization of multi-TRAM pipeline results.
Generates figures for all 4 stages: camera estimation, tracking, pose, world-space.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
import pickle, json, os
from pathlib import Path

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

RES = Path(__file__).parent.parent / "results"
COLORS = plt.cm.tab10(np.linspace(0, 1, 10))

# ── helpers ──────────────────────────────────────────────────────────────────

def load_tracks():
    tracks = {}
    for i in range(1, 9):
        frames = np.load(RES / f"2_tracking/tracks/track_{i}_frames.npy")
        bboxes = np.load(RES / f"2_tracking/tracks/track_{i}_bboxes.npy")
        tracks[i] = {"frames": frames, "bboxes": bboxes}
    return tracks

def load_trajectories():
    trajs = {}
    for i in range(1, 9):
        p = RES / f"4_world_space/person_{i:03d}/trajectory_world.npy"
        trajs[i] = np.load(p)
    return trajs

def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")

# ── Figure 1: Bird's-eye 2D trajectory (X-Z plane) ──────────────────────────

def fig_birdseye(trajs):
    fig, ax = plt.subplots(figsize=(10, 8))
    for pid, traj in trajs.items():
        c = COLORS[pid - 1]
        x, z = traj[:, 0], traj[:, 2]
        # gradient line (time → alpha)
        pts = np.stack([x, z], axis=1).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        n = len(segs)
        alphas = np.linspace(0.3, 1.0, n)
        lc = LineCollection(segs, colors=[(*c[:3], a) for a in alphas], linewidths=2)
        ax.add_collection(lc)
        ax.scatter(x[0], z[0], color=c, marker="o", s=80, zorder=5, label=f"Person {pid}")
        ax.scatter(x[-1], z[-1], color=c, marker="*", s=120, zorder=5)

    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Z — depth (m)", fontsize=12)
    ax.set_title("Multi-Person Trajectories — Bird's-Eye View (X-Z plane)\n○ = start,  ★ = end", fontsize=13)
    ax.legend(loc="upper left", fontsize=9)
    ax.autoscale()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    save(fig, "fig1_birdseye_trajectories.png")

# ── Figure 2: 3-D trajectories ───────────────────────────────────────────────

def fig_3d(trajs):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    for pid, traj in trajs.items():
        c = COLORS[pid - 1]
        x, y, z = traj[:, 0], traj[:, 1], traj[:, 2]
        ax.plot(x, z, y, color=c, linewidth=1.5, label=f"Person {pid}", alpha=0.85)
        ax.scatter([x[0]], [z[0]], [y[0]], color=c, marker="o", s=60, zorder=5)
        ax.scatter([x[-1]], [z[-1]], [y[-1]], color=c, marker="*", s=100, zorder=5)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z — depth (m)")
    ax.set_zlabel("Y — height (m)")
    ax.set_title("Multi-Person 3-D Trajectories in World Space", fontsize=13)
    ax.legend(fontsize=9, loc="upper left")
    save(fig, "fig2_3d_trajectories.png")

# ── Figure 3: Tracking timeline (Gantt) ─────────────────────────────────────

def fig_timeline(tracks):
    fig, ax = plt.subplots(figsize=(12, 5))
    for pid, info in tracks.items():
        frames = info["frames"]
        c = COLORS[pid - 1]
        # Draw continuous segments
        starts = [frames[0]]
        for k in range(1, len(frames)):
            if frames[k] != frames[k - 1] + 1:
                ax.barh(pid, frames[k - 1] - starts[-1] + 1, left=starts[-1],
                        height=0.6, color=c, alpha=0.85)
                starts.append(frames[k])
        ax.barh(pid, frames[-1] - starts[-1] + 1, left=starts[-1],
                height=0.6, color=c, alpha=0.85)
        ax.text(frames[-1] + 0.5, pid, f" {len(frames)}f", va="center", fontsize=9, color=c)

    ax.set_yticks(list(tracks.keys()))
    ax.set_yticklabels([f"Person {p}" for p in tracks])
    ax.set_xlabel("Frame index")
    ax.set_title("Person Tracking Timeline (frame coverage)")
    ax.set_xlim(-1, 105)
    ax.grid(axis="x", alpha=0.3)
    save(fig, "fig3_tracking_timeline.png")

# ── Figure 4: BBox center trajectories (image space) ────────────────────────

def fig_bbox_image(tracks):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    W, H = 1920, 1080

    for ax_i, (ax, coord, label) in enumerate(zip(axes, [0, 1], ["X (pixel)", "Y (pixel)"])):
        for pid, info in tracks.items():
            c = COLORS[pid - 1]
            frames = info["frames"]
            bboxes = info["bboxes"]
            # cx or cy from (x, y, w, h)
            centers = bboxes[:, coord] + bboxes[:, coord + 2] / 2
            ax.plot(frames, centers, color=c, linewidth=1.5, label=f"Person {pid}", alpha=0.85)
        ax.set_xlabel("Frame")
        ax.set_ylabel(label)
        ax.set_title(f"BBox center {label} over time")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        if ax_i == 1:
            ax.invert_yaxis()

    fig.suptitle("Bounding Box Trajectories — Image Space", fontsize=13)
    save(fig, "fig4_bbox_image_trajectories.png")

# ── Figure 5: Sample depth map ───────────────────────────────────────────────

def fig_depth():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (ax, frame_idx) in enumerate(zip(axes, [0, 49, 99])):
        d = np.load(RES / f"1_camera_estimation/depth_maps/depth_{frame_idx:06d}.npy")
        im = ax.imshow(d, cmap="plasma", origin="upper")
        plt.colorbar(im, ax=ax, fraction=0.03, label="Depth (m)")
        ax.set_title(f"Depth map — frame {frame_idx}")
        ax.axis("off")
    fig.suptitle("Scene Depth Maps (VGGT — monocular estimation)", fontsize=13)
    save(fig, "fig5_depth_maps.png")

# ── Figure 6: Camera trajectory ──────────────────────────────────────────────

def fig_camera():
    poses = np.load(RES / "1_camera_estimation/cameras/poses.npy")  # (100, 4, 4)
    # Camera position = last column of pose (world coordinates of camera)
    cam_pos = poses[:, :3, 3]  # (100, 3)
    T = np.arange(len(cam_pos))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # XZ bird's eye
    ax = axes[0]
    sc = ax.scatter(cam_pos[:, 0], cam_pos[:, 2], c=T, cmap="viridis", s=30, zorder=3)
    ax.plot(cam_pos[:, 0], cam_pos[:, 2], "k-", alpha=0.3, linewidth=1)
    plt.colorbar(sc, ax=ax, label="Frame")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("Camera Trajectory — Bird's-Eye (X-Z)")
    ax.grid(alpha=0.3)

    # Height over time
    ax = axes[1]
    ax.plot(T, cam_pos[:, 1], color="steelblue", linewidth=2)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Camera Y-height (m)")
    ax.set_title("Camera Height Over Time")
    ax.grid(alpha=0.3)

    fig.suptitle("Camera Trajectory — VGGT Estimated", fontsize=13)
    save(fig, "fig6_camera_trajectory.png")

# ── Figure 7: Trajectory statistics summary ──────────────────────────────────

def fig_stats(trajs):
    pids = sorted(trajs)
    lengths = []
    for pid in pids:
        t = trajs[pid]
        diffs = np.diff(t, axis=0)
        lengths.append(np.sum(np.linalg.norm(diffs, axis=1)))

    mean_z = [trajs[p][:, 2].mean() for p in pids]
    mean_x = [trajs[p][:, 0].mean() for p in pids]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    bars = ax.bar([f"P{p}" for p in pids], lengths,
                  color=[COLORS[p - 1] for p in pids], edgecolor="k", linewidth=0.5)
    ax.set_ylabel("Total path length (m)")
    ax.set_title("Trajectory Length per Person")
    ax.grid(axis="y", alpha=0.3)
    for bar, l in zip(bars, lengths):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{l:.1f}", ha="center", fontsize=9)

    ax = axes[1]
    ax.bar([f"P{p}" for p in pids], mean_z,
           color=[COLORS[p - 1] for p in pids], edgecolor="k", linewidth=0.5)
    ax.set_ylabel("Mean depth Z (m)")
    ax.set_title("Mean Depth (Distance from Camera)")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    for pid in pids:
        t = trajs[pid]
        ax.plot(t[:, 2], label=f"P{pid}", color=COLORS[pid - 1], linewidth=1.5)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Z — depth (m)")
    ax.set_title("Depth Over Time (all persons)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("World-Space Trajectory Statistics", fontsize=13)
    save(fig, "fig7_trajectory_stats.png")

# ── Figure 8: Person masks sample ────────────────────────────────────────────

def fig_masks():
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    frame_idx = 0
    for i, ax in enumerate(axes.flat):
        pid = i + 1
        p = RES / f"2_tracking/masks/frame_{frame_idx:06d}_person_{pid:03d}.npy"
        if p.exists():
            mask = np.load(p)
            ax.imshow(mask, cmap="hot", vmin=0, vmax=1)
            ax.set_title(f"Person {pid} mask\n(frame {frame_idx})")
        else:
            ax.text(0.5, 0.5, "no mask", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Person {pid}")
        ax.axis("off")
    fig.suptitle("Instance Segmentation Masks — Frame 0", fontsize=13)
    save(fig, "fig8_masks_sample.png")

# ── Figure 9: Combined overview dashboard ────────────────────────────────────

def fig_dashboard(trajs, tracks):
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    # --- Bird's eye ---
    ax1 = fig.add_subplot(gs[0, :2])
    for pid, traj in trajs.items():
        c = COLORS[pid - 1]
        x, z = traj[:, 0], traj[:, 2]
        ax1.plot(x, z, color=c, linewidth=1.8, label=f"P{pid}")
        ax1.scatter(x[0], z[0], color=c, marker="o", s=60, zorder=5)
        ax1.scatter(x[-1], z[-1], color=c, marker="*", s=90, zorder=5)
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Z depth (m)")
    ax1.set_title("Trajectories — Bird's-Eye (X-Z)")
    ax1.legend(fontsize=8, ncol=4, loc="upper right")
    ax1.grid(alpha=0.3)
    ax1.set_aspect("equal")

    # --- 3D ---
    ax2 = fig.add_subplot(gs[0, 2], projection="3d")
    for pid, traj in trajs.items():
        c = COLORS[pid - 1]
        ax2.plot(traj[:, 0], traj[:, 2], traj[:, 1], color=c, linewidth=1.2, alpha=0.85)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Z")
    ax2.set_zlabel("Y")
    ax2.set_title("3-D Trajectories")
    ax2.tick_params(labelsize=7)

    # --- Timeline ---
    ax3 = fig.add_subplot(gs[1, :2])
    for pid, info in tracks.items():
        frames = info["frames"]
        c = COLORS[pid - 1]
        starts = [frames[0]]
        for k in range(1, len(frames)):
            if frames[k] != frames[k - 1] + 1:
                ax3.barh(pid, frames[k - 1] - starts[-1] + 1, left=starts[-1],
                         height=0.6, color=c, alpha=0.85)
                starts.append(frames[k])
        ax3.barh(pid, frames[-1] - starts[-1] + 1, left=starts[-1],
                 height=0.6, color=c, alpha=0.85)
    ax3.set_yticks(list(tracks.keys()))
    ax3.set_yticklabels([f"P{p}" for p in tracks])
    ax3.set_xlabel("Frame")
    ax3.set_title("Tracking Timeline")
    ax3.grid(axis="x", alpha=0.3)

    # --- Trajectory lengths ---
    ax4 = fig.add_subplot(gs[1, 2])
    pids = sorted(trajs)
    lengths = []
    for pid in pids:
        t = trajs[pid]
        lengths.append(np.sum(np.linalg.norm(np.diff(t, axis=0), axis=1)))
    bars = ax4.bar([f"P{p}" for p in pids], lengths,
                   color=[COLORS[p - 1] for p in pids], edgecolor="k", linewidth=0.5)
    ax4.set_ylabel("Path length (m)")
    ax4.set_title("Trajectory Lengths")
    ax4.grid(axis="y", alpha=0.3)
    for bar, l in zip(bars, lengths):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{l:.0f}m", ha="center", fontsize=7)

    # --- Depth map ---
    ax5 = fig.add_subplot(gs[2, 0])
    d = np.load(RES / "1_camera_estimation/depth_maps/depth_000000.npy")
    im = ax5.imshow(d, cmap="plasma")
    plt.colorbar(im, ax=ax5, fraction=0.04, label="m")
    ax5.set_title("Depth Map — Frame 0")
    ax5.axis("off")

    # --- Camera trajectory ---
    ax6 = fig.add_subplot(gs[2, 1])
    poses = np.load(RES / "1_camera_estimation/cameras/poses.npy")
    cam_pos = poses[:, :3, 3]
    sc = ax6.scatter(cam_pos[:, 0], cam_pos[:, 2], c=np.arange(100), cmap="viridis", s=20)
    ax6.plot(cam_pos[:, 0], cam_pos[:, 2], "k-", alpha=0.2)
    plt.colorbar(sc, ax=ax6, label="frame", fraction=0.04)
    ax6.set_xlabel("X (m)")
    ax6.set_ylabel("Z (m)")
    ax6.set_title("Camera Trajectory")
    ax6.grid(alpha=0.3)

    # --- Depth over time ---
    ax7 = fig.add_subplot(gs[2, 2])
    for pid in pids:
        ax7.plot(trajs[pid][:, 2], color=COLORS[pid - 1], linewidth=1.5, label=f"P{pid}")
    ax7.set_xlabel("Frame")
    ax7.set_ylabel("Depth Z (m)")
    ax7.set_title("Person Depth Over Time")
    ax7.legend(fontsize=7, ncol=2)
    ax7.grid(alpha=0.3)

    fig.suptitle("Multi-TRAM Results Dashboard — 8 People, 100 Frames", fontsize=15, fontweight="bold")
    save(fig, "fig0_dashboard.png")

# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data...")
    tracks = load_tracks()
    trajs = load_trajectories()

    print("Generating figures...")
    fig_birdseye(trajs)
    fig_3d(trajs)
    fig_timeline(tracks)
    fig_bbox_image(tracks)
    fig_depth()
    fig_camera()
    fig_stats(trajs)
    fig_masks()
    fig_dashboard(trajs, tracks)

    print(f"\nDone. All figures saved to ./{OUT}/")
    for f in sorted(OUT.glob("*.png")):
        print(f"  {f.name}")
