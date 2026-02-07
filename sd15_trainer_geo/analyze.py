# =============================================================================
# analyze.py — Geometry profiler + claim verification charts
#
# Two parts:
#   1. GeometryProfiler callback — attach to trainer.fit() to snapshot
#      per-layer entropy, volume, deformation, blend every N steps
#   2. analyze() — reads log_history + profiler data, produces charts
#
# Usage:
#   profiler = GeometryProfiler(pipe, every=50)
#   trainer.fit(dataset, callbacks=[profiler])
#   analyze(trainer, profiler, save_dir="/content/analysis")
# =============================================================================

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any


# =============================================================================
# Part 1: Geometry profiler callback
# =============================================================================

class GeometryProfiler:
    """
    Training callback that snapshots per-layer geometry stats.
    Captures entropy, volume², deformation scale, blend at configurable intervals.
    """

    def __init__(self, pipe, every: int = 50):
        """
        Args:
            pipe:   Pipeline with unet that has get_geometry_stats()
            every:  Snapshot interval (in training steps)
        """
        self.pipe = pipe
        self.every = every
        self.snapshots: List[Dict[str, Any]] = []

    def __call__(self, trainer, step: int, logs: Dict[str, float]):
        """Called by trainer at each log step."""
        if step % self.every != 0:
            return

        stats = self.pipe.unet.get_geometry_stats()
        if not stats:
            return

        snapshot = {"step": step, **stats}
        # Also grab the logged geo components
        for k, v in logs.items():
            if k.startswith("geo/"):
                snapshot[k] = v
        snapshot["loss"] = logs.get("loss", 0)
        snapshot["task_loss"] = logs.get("task_loss", 0)
        snapshot["geo_loss"] = logs.get("geo_loss", 0)

        self.snapshots.append(snapshot)

    def save(self, path: str):
        """Save snapshots to JSON."""
        with open(path, "w") as f:
            json.dump(self.snapshots, f, indent=2)
        print(f"Profiler: saved {len(self.snapshots)} snapshots → {path}")

    def load(self, path: str):
        """Load snapshots from JSON."""
        with open(path) as f:
            self.snapshots = json.load(f)
        print(f"Profiler: loaded {len(self.snapshots)} snapshots ← {path}")


# =============================================================================
# Part 2: Analysis + charts
# =============================================================================

def analyze(
    trainer=None,
    profiler: Optional[GeometryProfiler] = None,
    log_history: Optional[List[Dict]] = None,
    snapshots: Optional[List[Dict]] = None,
    save_dir: str = "/content/analysis",
    num_layers: int = 4,
    show: bool = True,
):
    """
    Generate analysis charts from training data.

    Can use either:
      - trainer + profiler objects (live, after training)
      - log_history + snapshots lists (loaded from JSON)

    Produces:
      1. Loss curves (total, task, geo)
      2. Per-layer entropy over training (sharpening claim)
      3. Entropy decay rate + inflection point detection
      4. Blend evolution
      5. Per-layer volume² stability
      6. Per-layer deformation scale
      7. Combined summary dashboard
    """
    os.makedirs(save_dir, exist_ok=True)

    # Resolve data sources
    logs = log_history or (trainer.log_history if trainer else [])
    snaps = snapshots or (profiler.snapshots if profiler else [])

    if not logs:
        print("No log_history available.")
        return
    if not snaps:
        print("No profiler snapshots available. Run with GeometryProfiler callback.")
        print("Generating loss-only charts from log_history...")

    # ── Extract arrays from logs ──
    steps_log = np.array([l["step"] for l in logs])
    loss_total = np.array([l.get("loss", 0) for l in logs])
    loss_task = np.array([l.get("task_loss", 0) for l in logs])
    loss_geo = np.array([l.get("geo_loss", 0) for l in logs])
    lr = np.array([l.get("lr", 0) for l in logs])

    # ── Extract arrays from snapshots ──
    if snaps:
        steps_snap = np.array([s["step"] for s in snaps])
        blend = np.array([s.get("blend", 0.5) for s in snaps])
        entropy = {
            i: np.array([s.get(f"layer_{i}/entropy", 0) for s in snaps])
            for i in range(num_layers)
        }
        vol_sq = {
            i: np.array([s.get(f"layer_{i}/vol_sq", 0) for s in snaps])
            for i in range(num_layers)
        }
        deform = {
            i: np.array([s.get(f"layer_{i}/deform_scale", 0) for s in snaps])
            for i in range(num_layers)
        }

    # Colors
    layer_colors = ["#e74c3c", "#f39c12", "#27ae60", "#3498db"]
    layer_names = [f"Layer {i}" for i in range(num_layers)]

    # ── Chart 1: Loss curves ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("Loss Curves", fontweight="bold", fontsize=13)

    axes[0].plot(steps_log, loss_total, color="#2c3e50", linewidth=0.8, alpha=0.6)
    axes[0].plot(steps_log, _smooth(loss_total, 10), color="#2c3e50", linewidth=2)
    axes[0].set_title("Total Loss")
    axes[0].set_xlabel("Step")
    axes[0].grid(alpha=0.3)

    axes[1].plot(steps_log, loss_task, color="#2980b9", linewidth=0.8, alpha=0.6)
    axes[1].plot(steps_log, _smooth(loss_task, 10), color="#2980b9", linewidth=2)
    axes[1].set_title("Task Loss (flow matching)")
    axes[1].set_xlabel("Step")
    axes[1].grid(alpha=0.3)

    axes[2].plot(steps_log, loss_geo, color="#c0392b", linewidth=0.8, alpha=0.6)
    axes[2].plot(steps_log, _smooth(loss_geo, 10), color="#c0392b", linewidth=2)
    axes[2].set_title("Geometric Loss")
    axes[2].set_xlabel("Step")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "01_loss_curves.png"), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    if not snaps:
        print(f"Charts saved → {save_dir}/")
        return {"save_dir": save_dir}

    # ── Chart 2: Per-layer entropy (sharpening claim) ──
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(num_layers):
        ax.plot(steps_snap, entropy[i], color=layer_colors[i], linewidth=0.8, alpha=0.4)
        ax.plot(steps_snap, _smooth(entropy[i], 5), color=layer_colors[i],
                linewidth=2.5, label=layer_names[i])
    ax.set_title("Attention Entropy per Layer (↓ = sharpening)", fontweight="bold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Entropy (nats)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "02_entropy_per_layer.png"), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    # ── Chart 3: Entropy ordering verification ──
    # Claim: H₁ > H₂ > H₃ > H₄ (monotonic decrease, emergent)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 3a: Final entropy bar chart
    final_entropy = [entropy[i][-1] for i in range(num_layers)]
    bars = axes[0].bar(layer_names, final_entropy, color=layer_colors)
    axes[0].set_title("Final Entropy Ordering", fontweight="bold")
    axes[0].set_ylabel("Entropy (nats)")
    monotonic = all(final_entropy[i] >= final_entropy[i + 1] for i in range(num_layers - 1))
    verdict = "✓ MONOTONIC" if monotonic else "✗ NOT MONOTONIC"
    axes[0].annotate(verdict, xy=(0.5, 0.95), xycoords="axes fraction",
                     ha="center", fontsize=12, fontweight="bold",
                     color="green" if monotonic else "red")
    axes[0].grid(alpha=0.3, axis="y")

    # 3b: Entropy gap between layers over time
    for i in range(num_layers - 1):
        gap = entropy[i] - entropy[i + 1]
        smoothed = _smooth(gap, 5)
        axes[1].plot(steps_snap, smoothed, color=layer_colors[i], linewidth=2,
                     label=f"L{i}−L{i+1}")
        axes[1].axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    axes[1].set_title("Inter-layer Entropy Gap (>0 = correct ordering)", fontweight="bold")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("ΔEntropy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "03_entropy_ordering.png"), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    # ── Chart 4: Entropy decay rate + inflection detection ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    inflection_points = {}
    for i in range(num_layers):
        smoothed = _smooth(entropy[i], 7)
        # First derivative (rate of change)
        dH = np.gradient(smoothed, steps_snap)
        # Second derivative (acceleration — inflection where d²H/dt² = 0)
        d2H = np.gradient(dH, steps_snap)

        axes[0].plot(steps_snap, dH, color=layer_colors[i], linewidth=1.5,
                     label=layer_names[i])

        axes[1].plot(steps_snap, d2H, color=layer_colors[i], linewidth=1.5,
                     label=layer_names[i])

        # Find zero crossings of second derivative (inflection points)
        crossings = _zero_crossings(d2H)
        if len(crossings) > 0:
            # Take the most prominent (largest magnitude swing)
            diffs = np.abs(np.diff(d2H))
            valid = crossings[crossings < len(diffs)]
            if len(valid) > 0:
                best = valid[np.argmax(diffs[valid])]
                inflection_step = steps_snap[best]
                inflection_points[i] = {
                    "step": int(inflection_step),
                    "entropy": float(smoothed[best]),
                    "dH_dt": float(dH[best]),
                }
                axes[0].axvline(x=inflection_step, color=layer_colors[i],
                                linewidth=1, linestyle=":", alpha=0.7)
                axes[1].axvline(x=inflection_step, color=layer_colors[i],
                                linewidth=1, linestyle=":", alpha=0.7)

    axes[0].set_title("dH/dt — Entropy Rate of Change", fontweight="bold")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("dH/dt")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].axhline(y=0, color="black", linewidth=0.5, linestyle="--")

    axes[1].set_title("d²H/dt² — Inflection Detection", fontweight="bold")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("d²H/dt²")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].axhline(y=0, color="black", linewidth=0.5, linestyle="--")

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "04_entropy_inflection.png"), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    # Print inflection points
    print("\n── Entropy Inflection Points ──")
    for i, info in inflection_points.items():
        print(f"  Layer {i}: step {info['step']}, H={info['entropy']:.4f}, dH/dt={info['dH_dt']:.6f}")
    if inflection_points:
        avg_step = np.mean([v["step"] for v in inflection_points.values()])
        print(f"  Average inflection: step {avg_step:.0f} "
              f"({avg_step / steps_snap[-1] * 100:.1f}% through training)")

    # ── Chart 5: Blend evolution ──
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps_snap, blend, color="#8e44ad", linewidth=2)
    ax.axhline(y=0.5, color="gray", linewidth=1, linestyle="--", label="β=0.5 (equal mix)")
    ax.set_title("Residual Blend β (0=CLIP only, 1=geo only)", fontweight="bold")
    ax.set_xlabel("Step")
    ax.set_ylabel("β")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "05_blend.png"), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    # ── Chart 6: Volume² per layer ──
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(num_layers):
        ax.plot(steps_snap, vol_sq[i], color=layer_colors[i], linewidth=0.8, alpha=0.4)
        ax.plot(steps_snap, _smooth(vol_sq[i], 5), color=layer_colors[i],
                linewidth=2.5, label=layer_names[i])
    ax.set_title("Simplex Volume² per Layer", fontweight="bold")
    ax.set_xlabel("Step")
    ax.set_ylabel("vol²")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "06_volume_sq.png"), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    # ── Chart 7: Deformation scale per layer ──
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(num_layers):
        ax.plot(steps_snap, deform[i], color=layer_colors[i], linewidth=2.5,
                label=layer_names[i])
    ax.axhline(y=0.25, color="gray", linewidth=1, linestyle="--", label="δ_base=0.25")
    ax.set_title("Deformation Scale δ per Layer (timestep-conditioned)", fontweight="bold")
    ax.set_xlabel("Step")
    ax.set_ylabel("δ_eff")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "07_deformation_scale.png"), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    # ── Chart 8: Summary dashboard ──
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.3)

    # 8a: Total loss
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(steps_log, _smooth(loss_total, 10), color="#2c3e50", linewidth=2)
    ax.set_title("Total Loss", fontsize=10)
    ax.grid(alpha=0.3)

    # 8b: Task loss
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(steps_log, _smooth(loss_task, 10), color="#2980b9", linewidth=2)
    ax.set_title("Task Loss", fontsize=10)
    ax.grid(alpha=0.3)

    # 8c: Geo loss
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(steps_log, _smooth(loss_geo, 10), color="#c0392b", linewidth=2)
    ax.set_title("Geo Loss", fontsize=10)
    ax.grid(alpha=0.3)

    # 8d: Entropy per layer
    ax = fig.add_subplot(gs[1, 0:2])
    for i in range(num_layers):
        ax.plot(steps_snap, _smooth(entropy[i], 5), color=layer_colors[i],
                linewidth=2, label=layer_names[i])
    for i, info in inflection_points.items():
        ax.axvline(x=info["step"], color=layer_colors[i], linewidth=1, linestyle=":", alpha=0.5)
    ax.set_title("Entropy per Layer + Inflection Points", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 8e: Blend
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(steps_snap, blend, color="#8e44ad", linewidth=2)
    ax.axhline(y=0.5, color="gray", linewidth=0.5, linestyle="--")
    ax.set_title("Blend β", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)

    # 8f: Volume²
    ax = fig.add_subplot(gs[2, 0])
    for i in range(num_layers):
        ax.plot(steps_snap, _smooth(vol_sq[i], 5), color=layer_colors[i], linewidth=2)
    ax.set_title("Volume²", fontsize=10)
    ax.grid(alpha=0.3)

    # 8g: Deformation
    ax = fig.add_subplot(gs[2, 1])
    for i in range(num_layers):
        ax.plot(steps_snap, deform[i], color=layer_colors[i], linewidth=2)
    ax.set_title("Deformation δ", fontsize=10)
    ax.grid(alpha=0.3)

    # 8h: Entropy ordering final
    ax = fig.add_subplot(gs[2, 2])
    bars = ax.bar(layer_names, final_entropy, color=layer_colors)
    ax.set_title(f"Final Entropy — {verdict}", fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("KSimplex Geometry Analysis Dashboard", fontsize=14, fontweight="bold", y=0.98)
    fig.savefig(os.path.join(save_dir, "08_dashboard.png"), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    # ── Save numerical summary ──
    summary = {
        "total_steps": int(steps_log[-1]),
        "loss_final": float(loss_total[-1]),
        "task_loss_final": float(loss_task[-1]),
        "geo_loss_final": float(loss_geo[-1]),
        "blend_final": float(blend[-1]),
        "entropy_final": {f"layer_{i}": float(entropy[i][-1]) for i in range(num_layers)},
        "entropy_monotonic": monotonic,
        "inflection_points": inflection_points,
        "vol_sq_final": {f"layer_{i}": float(vol_sq[i][-1]) for i in range(num_layers)},
        "deform_final": {f"layer_{i}": float(deform[i][-1]) for i in range(num_layers)},
    }

    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n── Summary ──")
    print(f"  Loss:    {summary['loss_final']:.4f} (task={summary['task_loss_final']:.4f}, geo={summary['geo_loss_final']:.6f})")
    print(f"  Blend:   {summary['blend_final']:.4f}")
    entropy_parts = []
    for i in range(num_layers):
        key = f"layer_{i}"
        val = summary["entropy_final"][key]
        entropy_parts.append(f"L{i}={val:.3f}")
    print(f"  Entropy: {' > '.join(entropy_parts)}")
    print(f"  Monotonic: {verdict}")
    print(f"  Charts → {save_dir}/")

    return summary


# =============================================================================
# Utilities
# =============================================================================

def _smooth(arr: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average, padded to same length."""
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    smoothed = np.convolve(arr, kernel, mode="valid")
    # Pad front to align
    pad = len(arr) - len(smoothed)
    return np.concatenate([arr[:pad], smoothed])


def _zero_crossings(arr: np.ndarray) -> np.ndarray:
    """Find indices where array crosses zero."""
    signs = np.sign(arr)
    crossings = np.where(np.diff(signs) != 0)[0]
    return crossings


# =============================================================================
# Standalone: load from saved files
# =============================================================================

def analyze_from_files(
    log_path: str,
    profiler_path: str,
    save_dir: str = "/content/analysis",
    show: bool = True,
):
    """
    Run analysis from saved JSON files.

    Usage:
        analyze_from_files(
            "/content/geo_prior_object_relations/log_history.json",
            "/content/geo_prior_object_relations/profiler.json",
        )
    """
    with open(log_path) as f:
        log_history = json.load(f)
    with open(profiler_path) as f:
        snapshots = json.load(f)

    return analyze(
        log_history=log_history,
        snapshots=snapshots,
        save_dir=save_dir,
        show=show,
    )