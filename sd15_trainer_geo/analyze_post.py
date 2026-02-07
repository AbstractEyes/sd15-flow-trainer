# =============================================================================
# analyze_post.py — Post-training model analyzer
#
# Diagnoses a trained KSimplex geometric prior using only the model weights.
# No training logs or profiler data required.
#
# Usage:
#   from analyze_post import PostTrainingAnalyzer
#   analyzer = PostTrainingAnalyzer(pipe)
#   results = analyzer.run_all(save_dir="/content/post_analysis")
#
# Or piecemeal:
#   analyzer.weight_inspection()
#   analyzer.timestep_sweep(prompts)
#   analyzer.prompt_complexity(simple, spatial, complex)
#   analyzer.compare_to_fresh()
# =============================================================================

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import os
import copy
from typing import Dict, List, Optional, Any, Tuple


class PostTrainingAnalyzer:
    """
    Post-training diagnostic tool for KSimplex geometric prior.
    All analysis from forward passes on the current weights — no training data needed.
    """

    def __init__(self, pipe, device: str = "cuda", dtype=torch.float16):
        self.pipe = pipe
        self.unet = pipe.unet
        self.device = device
        self.dtype = dtype
        self.num_layers = len(self.unet.geo_prior.attention.layers)

        # Colors
        self.layer_colors = ["#e74c3c", "#f39c12", "#27ae60", "#3498db"][:self.num_layers]
        self.layer_names = [f"Layer {i}" for i in range(self.num_layers)]

    # =========================================================================
    # 1. Weight inspection (no forward pass)
    # =========================================================================

    def weight_inspection(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Inspect raw parameter values of the trained geo_prior.
        No forward pass needed.
        """
        prior = self.unet.geo_prior
        results = {}

        # Blend β
        if hasattr(prior, "blend_logit"):
            logit = prior.blend_logit.item()
            beta = torch.sigmoid(prior.blend_logit).item()
            results["blend_logit"] = logit
            results["blend_beta"] = beta
        elif hasattr(prior, "blend_mlp"):
            results["blend_type"] = "timestep_conditioned"
            # Probe at a few timesteps
            blend_at_t = {}
            for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
                with torch.no_grad():
                    t_in = torch.tensor([[t_val]], device=self.device, dtype=torch.float32)
                    b = torch.sigmoid(prior.blend_mlp(t_in)).item()
                    blend_at_t[f"t={t_val}"] = b
            results["blend_at_t"] = blend_at_t

        # Per-layer parameters
        results["layers"] = {}
        for i, layer in enumerate(prior.attention.layers):
            layer_info = {}

            # Deformation scale
            ds = torch.clamp(layer.deformation_scale, 0.05, 0.5).item()
            layer_info["deformation_scale"] = ds
            layer_info["deformation_scale_raw"] = layer.deformation_scale.item()

            # Deformation offset norms
            offsets = layer.deformation_offsets.data
            layer_info["offset_norm_mean"] = offsets.norm(dim=-1).mean().item()
            layer_info["offset_norm_std"] = offsets.norm(dim=-1).std().item()
            layer_info["offset_norm_per_vertex"] = offsets.norm(dim=-1).tolist()

            # Template (frozen buffer)
            template = layer.template
            layer_info["template_norm_mean"] = template.norm(dim=-1).mean().item()

            # Deformed template (what the model actually uses)
            deformed = template + offsets * ds
            layer_info["deformed_norm_mean"] = deformed.norm(dim=-1).mean().item()

            # Pairwise distances between deformed vertices
            diff = deformed.unsqueeze(0) - deformed.unsqueeze(1)  # (k+1, k+1, edim)
            dists = diff.pow(2).sum(-1).sqrt()  # (k+1, k+1)
            mask = torch.triu(torch.ones_like(dists, dtype=torch.bool), diagonal=1)
            edge_dists = dists[mask]
            layer_info["vertex_dist_mean"] = edge_dists.mean().item()
            layer_info["vertex_dist_std"] = edge_dists.std().item()
            layer_info["vertex_dist_min"] = edge_dists.min().item()
            layer_info["vertex_dist_max"] = edge_dists.max().item()

            # Projection weight norms
            layer_info["to_coords_norm"] = layer.to_coords.weight.data.norm().item()
            layer_info["to_v_norm"] = layer.to_v.weight.data.norm().item()
            layer_info["to_out_norm"] = layer.to_out.weight.data.norm().item()
            layer_info["vertex_proj_norm"] = layer.token_to_vertex.weight.data.norm().item()

            results["layers"][f"layer_{i}"] = layer_info

        # Deformation schedule MLP (if timestep conditioned)
        if hasattr(prior, "deform_schedule"):
            results["deform_schedule"] = {
                "w1_norm": prior.deform_schedule[0].weight.data.norm().item(),
                "w2_norm": prior.deform_schedule[2].weight.data.norm().item(),
            }

        # Total parameter stats
        all_params = list(prior.parameters())
        total_norm = sum(p.data.norm().item() ** 2 for p in all_params) ** 0.5
        results["total_param_norm"] = total_norm
        results["num_params"] = sum(p.numel() for p in all_params)

        if verbose:
            print("── Weight Inspection ──")
            if "blend_beta" in results:
                print(f"  Blend β = {results['blend_beta']:.4f} (logit={results['blend_logit']:.4f})")
            elif "blend_at_t" in results:
                print("  Blend β (timestep-conditioned):")
                for k, v in results["blend_at_t"].items():
                    print(f"    {k}: {v:.4f}")
            print(f"  Total param norm: {total_norm:.2f}")
            print()
            for i in range(self.num_layers):
                li = results["layers"][f"layer_{i}"]
                print(f"  Layer {i}:")
                print(f"    δ = {li['deformation_scale']:.4f} (raw={li['deformation_scale_raw']:.4f})")
                print(f"    offset norms: {li['offset_norm_mean']:.4f} ± {li['offset_norm_std']:.4f}")
                print(f"    vertex dists: {li['vertex_dist_mean']:.4f} ± {li['vertex_dist_std']:.4f}"
                      f" [{li['vertex_dist_min']:.4f}, {li['vertex_dist_max']:.4f}]")

        return results

    # =========================================================================
    # 2. Forward pass diagnostics (single prompt)
    # =========================================================================

    @torch.no_grad()
    def forward_diagnostics(
        self,
        prompts: List[str],
        timestep: float = 0.5,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run forward pass and extract geometry diagnostics.

        Args:
            prompts: list of text prompts
            timestep: normalized timestep [0, 1] to evaluate at
        """
        self.unet.eval()

        # Encode prompts
        enc_hs = self._encode_prompts(prompts)
        B = enc_hs.shape[0]

        # Create dummy noisy latent
        latent = torch.randn(B, 4, 64, 64, device=self.device, dtype=self.dtype)
        t_int = torch.full((B,), int(timestep * 1000), device=self.device, dtype=torch.long)
        t_cont = torch.full((B,), timestep, device=self.device, dtype=torch.float32)

        # Forward pass
        with torch.amp.autocast("cuda", dtype=self.dtype):
            _ = self.unet(latent, t_int, enc_hs, t_continuous=t_cont)

        # Extract from _last_prior_info
        stats = self.unet.get_geometry_stats()
        info = self.unet._last_prior_info

        results = {
            "timestep": timestep,
            "prompts": prompts,
            "stats": stats,
        }

        # Per-layer detail
        for i, geom in enumerate(info["all_geometry"]):
            layer_data = {}

            # Attention entropy
            dist_sq = geom["dist_sq"].float()
            scale = self.unet.simplex_config.edim ** -0.5
            attn = F.softmax(-dist_sq * scale, dim=-1)
            entropy = -(attn * (attn + 1e-10).log()).sum(dim=-1)  # (B, T)
            layer_data["entropy_mean"] = entropy.mean().item()
            layer_data["entropy_std"] = entropy.std().item()
            layer_data["entropy_per_token_mean"] = entropy.mean(dim=0).cpu().numpy()

            # Vertex weight distribution
            vw = geom["vertex_weights"]  # (B, T, k+1)
            layer_data["vertex_weight_entropy"] = (
                -(vw * (vw + 1e-10).log()).sum(dim=-1).mean().item()
            )
            layer_data["vertex_assignment_hard"] = vw.argmax(dim=-1)[0].cpu().numpy()

            # Volume²
            from sd15_trainer_geo.unet.base_simplex import compute_simplex_volume_sq
            vol_sq = compute_simplex_volume_sq(
                geom["sampled_dist_sq"].float(), geom["effective_k"]
            )
            layer_data["vol_sq_mean"] = vol_sq.mean().item()
            layer_data["vol_sq_std"] = vol_sq.std().item()

            results[f"layer_{i}"] = layer_data

        if verbose:
            print(f"── Forward Diagnostics (t={timestep}) ──")
            blend = info["blend"]
            if isinstance(blend, torch.Tensor):
                print(f"  Blend β = {blend.mean().item():.4f}")
            for i in range(self.num_layers):
                ld = results[f"layer_{i}"]
                print(f"  Layer {i}: entropy={ld['entropy_mean']:.4f} ± {ld['entropy_std']:.4f}, "
                      f"vol²={ld['vol_sq_mean']:.4e}, "
                      f"vw_entropy={ld['vertex_weight_entropy']:.4f}")

        return results

    # =========================================================================
    # 3. Timestep sweep (the critical one for branching)
    # =========================================================================

    @torch.no_grad()
    def timestep_sweep(
        self,
        prompts: List[str],
        t_values: Optional[List[float]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the same prompts across a range of timesteps.
        Maps entropy, deformation, volume vs noise level.

        This is the curve you need for branching inflection detection.
        """
        if t_values is None:
            t_values = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5,
                        0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

        self.unet.eval()
        enc_hs = self._encode_prompts(prompts)
        B = enc_hs.shape[0]

        # Fixed noise for consistency
        latent_noise = torch.randn(B, 4, 64, 64, device=self.device, dtype=self.dtype)
        # Fixed clean latent
        latent_clean = torch.randn(B, 4, 64, 64, device=self.device, dtype=self.dtype) * 0.1

        sweep_data = []

        for t_val in t_values:
            # Interpolate latent
            t_expand = torch.full((B, 1, 1, 1), t_val, device=self.device, dtype=self.dtype)
            x_t = (1.0 - t_expand) * latent_clean + t_expand * latent_noise

            t_int = torch.full((B,), int(t_val * 1000), device=self.device, dtype=torch.long)
            t_cont = torch.full((B,), t_val, device=self.device, dtype=torch.float32)

            with torch.amp.autocast("cuda", dtype=self.dtype):
                _ = self.unet(x_t, t_int, enc_hs, t_continuous=t_cont)

            info = self.unet._last_prior_info
            stats = self.unet.get_geometry_stats()

            entry = {"t": t_val, **stats}

            # Blend at this timestep
            blend = info["blend"]
            entry["blend"] = blend.mean().item() if isinstance(blend, torch.Tensor) else blend

            # Per-layer attention entropy (recompute for precision)
            for i, geom in enumerate(info["all_geometry"]):
                dist_sq = geom["dist_sq"].float()
                scale = self.unet.simplex_config.edim ** -0.5
                attn = F.softmax(-dist_sq * scale, dim=-1)
                ent = -(attn * (attn + 1e-10).log()).sum(dim=-1).mean().item()
                entry[f"layer_{i}/entropy"] = ent

                # Vertex weight concentration
                vw = geom["vertex_weights"]
                max_weight = vw.max(dim=-1).values.mean().item()
                entry[f"layer_{i}/vertex_max_weight"] = max_weight

            sweep_data.append(entry)

        if verbose:
            print("── Timestep Sweep ──")
            print(f"  {'t':>5} | {'blend':>6} | " + " | ".join(
                f"H_{i}" for i in range(self.num_layers)) + " | " + " | ".join(
                f"δ_{i}" for i in range(self.num_layers)))
            print("  " + "-" * (12 + 9 * self.num_layers * 2))
            for entry in sweep_data:
                parts = [f"{entry['t']:5.2f}", f"{entry['blend']:6.4f}"]
                for i in range(self.num_layers):
                    parts.append(f"{entry.get(f'layer_{i}/entropy', 0):5.3f}")
                for i in range(self.num_layers):
                    parts.append(f"{entry.get(f'layer_{i}/deform_scale', 0):5.4f}")
                print("  " + " | ".join(parts))

        return {"t_values": t_values, "sweep": sweep_data}

    # =========================================================================
    # 4. Prompt complexity comparison
    # =========================================================================

    @torch.no_grad()
    def prompt_complexity(
        self,
        simple: List[str] = None,
        spatial: List[str] = None,
        complex_prompts: List[str] = None,
        timestep: float = 0.5,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Compare geometry metrics across prompt complexity levels.
        Tests whether the prior responds differently to compositional structure.
        """
        if simple is None:
            simple = ["a cat", "a red ball", "a wooden chair", "a glass vase"]
        if spatial is None:
            spatial = [
                "a cup on top of a book",
                "a cat beside a vase",
                "shoes next to an umbrella",
                "a ball inside a bowl",
            ]
        if complex_prompts is None:
            complex_prompts = [
                "a red apple behind a green bottle on a wooden table",
                "a key inside a shoe next to the front door",
                "three candles arranged in a triangle on a tray",
                "a hat resting on a basketball beside a bookshelf",
            ]

        categories = {
            "simple": simple,
            "spatial": spatial,
            "complex": complex_prompts,
        }

        results = {}
        for label, prompts in categories.items():
            diag = self.forward_diagnostics(prompts, timestep=timestep, verbose=False)
            summary = {"prompts": prompts}
            for i in range(self.num_layers):
                ld = diag[f"layer_{i}"]
                summary[f"layer_{i}/entropy"] = ld["entropy_mean"]
                summary[f"layer_{i}/vol_sq"] = ld["vol_sq_mean"]
                summary[f"layer_{i}/vw_entropy"] = ld["vertex_weight_entropy"]
            results[label] = summary

        if verbose:
            print(f"── Prompt Complexity (t={timestep}) ──")
            for label, summary in results.items():
                entropies = [summary[f"layer_{i}/entropy"] for i in range(self.num_layers)]
                vw_ents = [summary[f"layer_{i}/vw_entropy"] for i in range(self.num_layers)]
                print(f"  {label:>8}: entropy=[{', '.join(f'{e:.3f}' for e in entropies)}]  "
                      f"vw_ent=[{', '.join(f'{e:.3f}' for e in vw_ents)}]")

        return results

    # =========================================================================
    # 5. Comparison to fresh init
    # =========================================================================

    @torch.no_grad()
    def compare_to_fresh(
        self,
        prompts: List[str] = None,
        timestep: float = 0.5,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Compare trained weights to freshly initialized geo_prior.
        Shows exactly what training moved.
        """
        if prompts is None:
            prompts = [
                "a red cup on top of a blue book",
                "a cat sitting beside a vase of flowers",
            ]

        # Get trained diagnostics
        trained = self.forward_diagnostics(prompts, timestep=timestep, verbose=False)
        trained_weights = self.weight_inspection(verbose=False)

        # Save trained state, reinit, measure, restore
        trained_state = {
            k: v.clone() for k, v in self.unet.geo_prior.state_dict().items()
        }

        # Reinitialize geo_prior
        self._reinit_geo_prior()

        fresh = self.forward_diagnostics(prompts, timestep=timestep, verbose=False)
        fresh_weights = self.weight_inspection(verbose=False)

        # Restore trained weights
        self.unet.geo_prior.load_state_dict(trained_state)

        # Compute deltas
        results = {"trained": {}, "fresh": {}, "delta": {}}

        for i in range(self.num_layers):
            for metric in ["entropy_mean", "vol_sq_mean", "vertex_weight_entropy"]:
                t_val = trained[f"layer_{i}"][metric]
                f_val = fresh[f"layer_{i}"][metric]
                results["trained"][f"layer_{i}/{metric}"] = t_val
                results["fresh"][f"layer_{i}/{metric}"] = f_val
                results["delta"][f"layer_{i}/{metric}"] = t_val - f_val

        # Blend comparison
        if "blend_beta" in trained_weights:
            results["trained"]["blend"] = trained_weights["blend_beta"]
            results["fresh"]["blend"] = fresh_weights["blend_beta"]
            results["delta"]["blend"] = trained_weights["blend_beta"] - fresh_weights["blend_beta"]

        # Deformation comparison
        for i in range(self.num_layers):
            t_ds = trained_weights["layers"][f"layer_{i}"]["deformation_scale"]
            f_ds = fresh_weights["layers"][f"layer_{i}"]["deformation_scale"]
            results["trained"][f"layer_{i}/deform_scale"] = t_ds
            results["fresh"][f"layer_{i}/deform_scale"] = f_ds
            results["delta"][f"layer_{i}/deform_scale"] = t_ds - f_ds

        if verbose:
            print(f"── Trained vs Fresh (t={timestep}) ──")
            print(f"  {'metric':>30} | {'trained':>10} | {'fresh':>10} | {'Δ':>10}")
            print("  " + "-" * 68)
            for k in sorted(results["delta"].keys()):
                t = results["trained"][k]
                f = results["fresh"][k]
                d = results["delta"][k]
                print(f"  {k:>30} | {t:10.4f} | {f:10.4f} | {d:+10.4f}")

        return results

    # =========================================================================
    # 6. Full analysis with charts
    # =========================================================================

    def run_all(
        self,
        save_dir: str = "/content/post_analysis",
        prompts: Optional[List[str]] = None,
        show: bool = True,
    ) -> Dict[str, Any]:
        """Run all analyses and generate charts."""
        os.makedirs(save_dir, exist_ok=True)

        if prompts is None:
            prompts = [
                "a red cup on top of a blue book",
                "a cat sitting beside a vase of flowers",
                "a small ball inside a glass bowl on a table",
                "a pair of shoes next to an umbrella by the door",
            ]

        print("=" * 60)
        print("POST-TRAINING ANALYSIS")
        print("=" * 60)

        # 1. Weight inspection
        print("\n[1/5] Weight inspection...")
        weights = self.weight_inspection(verbose=True)

        # 2. Forward diagnostics
        print("\n[2/5] Forward diagnostics...")
        fwd = self.forward_diagnostics(prompts, timestep=0.5, verbose=True)

        # 3. Timestep sweep
        print("\n[3/5] Timestep sweep...")
        sweep = self.timestep_sweep(prompts, verbose=True)

        # 4. Prompt complexity
        print("\n[4/5] Prompt complexity comparison...")
        complexity = self.prompt_complexity(verbose=True)

        # 5. Fresh comparison
        print("\n[5/5] Trained vs fresh comparison...")
        comparison = self.compare_to_fresh(prompts=prompts[:2], verbose=True)

        # ── Generate charts ──
        print("\nGenerating charts...")
        self._plot_timestep_sweep(sweep, save_dir, show)
        self._plot_prompt_complexity(complexity, save_dir, show)
        self._plot_comparison(comparison, save_dir, show)
        self._plot_weight_summary(weights, save_dir, show)
        self._plot_dashboard(weights, sweep, complexity, comparison, save_dir, show)

        # Save results
        all_results = {
            "weights": _make_serializable(weights),
            "sweep": _make_serializable(sweep),
            "complexity": _make_serializable(complexity),
            "comparison": _make_serializable(comparison),
        }
        with open(os.path.join(save_dir, "post_analysis.json"), "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        print(f"\n✓ All results saved → {save_dir}/")
        return all_results

    # =========================================================================
    # Charts
    # =========================================================================

    def _plot_timestep_sweep(self, sweep, save_dir, show):
        """Charts 1-2: Timestep sweep — entropy and deformation vs t."""
        data = sweep["sweep"]
        t_arr = np.array([d["t"] for d in data])

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Timestep Sweep — Geometry vs Noise Level", fontweight="bold", fontsize=13)

        # Entropy vs t
        ax = axes[0, 0]
        for i in range(self.num_layers):
            vals = np.array([d.get(f"layer_{i}/entropy", 0) for d in data])
            ax.plot(t_arr, vals, color=self.layer_colors[i], linewidth=2.5,
                    label=self.layer_names[i], marker="o", markersize=4)
        ax.set_title("Attention Entropy vs Timestep")
        ax.set_xlabel("t (0=clean, 1=noise)")
        ax.set_ylabel("Entropy (nats)")
        ax.legend()
        ax.grid(alpha=0.3)

        # Deformation vs t
        ax = axes[0, 1]
        for i in range(self.num_layers):
            vals = np.array([d.get(f"layer_{i}/deform_scale", 0) for d in data])
            ax.plot(t_arr, vals, color=self.layer_colors[i], linewidth=2.5,
                    label=self.layer_names[i], marker="o", markersize=4)
        ax.axhline(y=0.25, color="gray", linewidth=1, linestyle="--", label="δ_base")
        ax.set_title("Effective Deformation δ vs Timestep")
        ax.set_xlabel("t")
        ax.set_ylabel("δ_eff")
        ax.legend()
        ax.grid(alpha=0.3)

        # Volume² vs t
        ax = axes[1, 0]
        for i in range(self.num_layers):
            vals = np.array([d.get(f"layer_{i}/vol_sq", 0) for d in data])
            ax.plot(t_arr, vals, color=self.layer_colors[i], linewidth=2.5,
                    label=self.layer_names[i], marker="o", markersize=4)
        ax.set_title("Simplex Volume² vs Timestep")
        ax.set_xlabel("t")
        ax.set_ylabel("vol²")
        ax.legend()
        ax.grid(alpha=0.3)

        # Blend vs t
        ax = axes[1, 1]
        blends = np.array([d.get("blend", 0.5) for d in data])
        ax.plot(t_arr, blends, color="#8e44ad", linewidth=2.5, marker="o", markersize=4)
        ax.axhline(y=0.5, color="gray", linewidth=1, linestyle="--")
        ax.set_title("Blend β vs Timestep")
        ax.set_xlabel("t")
        ax.set_ylabel("β")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "01_timestep_sweep.png"), dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()

        # Entropy derivatives for inflection detection
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Entropy Derivatives vs Timestep — Inflection Detection", fontweight="bold")

        inflection_points = {}
        for i in range(self.num_layers):
            vals = np.array([d.get(f"layer_{i}/entropy", 0) for d in data])
            dH = np.gradient(vals, t_arr)
            d2H = np.gradient(dH, t_arr)

            axes[0].plot(t_arr, dH, color=self.layer_colors[i], linewidth=2,
                         label=self.layer_names[i])
            axes[1].plot(t_arr, d2H, color=self.layer_colors[i], linewidth=2,
                         label=self.layer_names[i])

            # Find zero crossings of d²H/dt²
            signs = np.sign(d2H)
            crossings = np.where(np.diff(signs) != 0)[0]
            if len(crossings) > 0:
                diffs = np.abs(np.diff(d2H))
                valid = crossings[crossings < len(diffs)]
                if len(valid) > 0:
                    best = valid[np.argmax(diffs[valid])]
                    t_inflect = t_arr[best]
                    inflection_points[i] = {
                        "t": float(t_inflect),
                        "entropy": float(vals[best]),
                        "dH_dt": float(dH[best]),
                    }
                    for ax in axes:
                        ax.axvline(x=t_inflect, color=self.layer_colors[i],
                                   linewidth=1, linestyle=":", alpha=0.7)

        axes[0].set_title("dH/dt")
        axes[0].set_xlabel("t")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[0].axhline(y=0, color="black", linewidth=0.5, linestyle="--")

        axes[1].set_title("d²H/dt² (zero crossings = inflection)")
        axes[1].set_xlabel("t")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        axes[1].axhline(y=0, color="black", linewidth=0.5, linestyle="--")

        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "02_timestep_inflection.png"), dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()

        if inflection_points:
            print("\n── Timestep Inflection Points ──")
            for i, info in inflection_points.items():
                print(f"  Layer {i}: t={info['t']:.3f}, H={info['entropy']:.4f}")
            avg_t = np.mean([v["t"] for v in inflection_points.values()])
            print(f"  Average inflection: t={avg_t:.3f}")
            print(f"  → Branch point recommendation: t ≈ {avg_t:.2f}")

    def _plot_prompt_complexity(self, complexity, save_dir, show):
        """Chart 3: Entropy by prompt complexity level."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Geometry vs Prompt Complexity", fontweight="bold", fontsize=13)

        categories = list(complexity.keys())
        x = np.arange(self.num_layers)
        width = 0.25
        cat_colors = ["#3498db", "#e67e22", "#e74c3c"]

        # Attention entropy
        ax = axes[0]
        for j, cat in enumerate(categories):
            vals = [complexity[cat][f"layer_{i}/entropy"] for i in range(self.num_layers)]
            ax.bar(x + j * width, vals, width, label=cat, color=cat_colors[j], alpha=0.8)
        ax.set_title("Attention Entropy by Complexity")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Entropy")
        ax.set_xticks(x + width)
        ax.set_xticklabels(self.layer_names)
        ax.legend()
        ax.grid(alpha=0.3, axis="y")

        # Vertex weight entropy
        ax = axes[1]
        for j, cat in enumerate(categories):
            vals = [complexity[cat][f"layer_{i}/vw_entropy"] for i in range(self.num_layers)]
            ax.bar(x + j * width, vals, width, label=cat, color=cat_colors[j], alpha=0.8)
        ax.set_title("Vertex Assignment Entropy by Complexity")
        ax.set_xlabel("Layer")
        ax.set_ylabel("VW Entropy")
        ax.set_xticks(x + width)
        ax.set_xticklabels(self.layer_names)
        ax.legend()
        ax.grid(alpha=0.3, axis="y")

        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "03_prompt_complexity.png"), dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()

    def _plot_comparison(self, comparison, save_dir, show):
        """Chart 4: Trained vs fresh delta."""
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle("Trained vs Fresh Init — What Training Moved", fontweight="bold", fontsize=13)

        keys = sorted(comparison["delta"].keys())
        deltas = [comparison["delta"][k] for k in keys]
        colors = ["#27ae60" if d > 0 else "#e74c3c" for d in deltas]

        y_pos = np.arange(len(keys))
        ax.barh(y_pos, deltas, color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(keys, fontsize=8)
        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.set_xlabel("Δ (trained − fresh)")
        ax.grid(alpha=0.3, axis="x")

        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "04_trained_vs_fresh.png"), dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()

    def _plot_weight_summary(self, weights, save_dir, show):
        """Chart 5: Weight structure overview."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle("Weight Structure", fontweight="bold", fontsize=13)

        # Deformation scales
        ax = axes[0]
        ds = [weights["layers"][f"layer_{i}"]["deformation_scale"] for i in range(self.num_layers)]
        ax.bar(self.layer_names, ds, color=self.layer_colors, alpha=0.8)
        ax.axhline(y=0.25, color="gray", linewidth=1, linestyle="--", label="δ_base")
        ax.set_title("Deformation Scale δ")
        ax.set_ylabel("δ")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")

        # Offset norms
        ax = axes[1]
        norms = [weights["layers"][f"layer_{i}"]["offset_norm_mean"] for i in range(self.num_layers)]
        stds = [weights["layers"][f"layer_{i}"]["offset_norm_std"] for i in range(self.num_layers)]
        ax.bar(self.layer_names, norms, yerr=stds, color=self.layer_colors, alpha=0.8, capsize=5)
        ax.set_title("Deformation Offset Norms")
        ax.set_ylabel("‖offset‖")
        ax.grid(alpha=0.3, axis="y")

        # Vertex distances
        ax = axes[2]
        means = [weights["layers"][f"layer_{i}"]["vertex_dist_mean"] for i in range(self.num_layers)]
        stds = [weights["layers"][f"layer_{i}"]["vertex_dist_std"] for i in range(self.num_layers)]
        ax.bar(self.layer_names, means, yerr=stds, color=self.layer_colors, alpha=0.8, capsize=5)
        ax.set_title("Deformed Vertex Distances")
        ax.set_ylabel("dist")
        ax.grid(alpha=0.3, axis="y")

        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "05_weight_structure.png"), dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()

    def _plot_dashboard(self, weights, sweep, complexity, comparison, save_dir, show):
        """Chart 6: Combined dashboard."""
        fig = plt.figure(figsize=(20, 14))
        gs = gridspec.GridSpec(3, 4, hspace=0.35, wspace=0.35)

        data = sweep["sweep"]
        t_arr = np.array([d["t"] for d in data])

        # Row 1: Timestep sweep
        ax = fig.add_subplot(gs[0, 0:2])
        for i in range(self.num_layers):
            vals = np.array([d.get(f"layer_{i}/entropy", 0) for d in data])
            ax.plot(t_arr, vals, color=self.layer_colors[i], linewidth=2,
                    label=self.layer_names[i], marker="o", markersize=3)
        ax.set_title("Entropy vs Timestep", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        ax = fig.add_subplot(gs[0, 2:4])
        for i in range(self.num_layers):
            vals = np.array([d.get(f"layer_{i}/deform_scale", 0) for d in data])
            ax.plot(t_arr, vals, color=self.layer_colors[i], linewidth=2,
                    label=self.layer_names[i], marker="o", markersize=3)
        ax.set_title("Deformation vs Timestep", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        # Row 2: Complexity + weights
        ax = fig.add_subplot(gs[1, 0:2])
        categories = list(complexity.keys())
        x = np.arange(self.num_layers)
        width = 0.25
        cat_colors = ["#3498db", "#e67e22", "#e74c3c"]
        for j, cat in enumerate(categories):
            vals = [complexity[cat][f"layer_{i}/entropy"] for i in range(self.num_layers)]
            ax.bar(x + j * width, vals, width, label=cat, color=cat_colors[j], alpha=0.8)
        ax.set_title("Entropy by Complexity", fontsize=10)
        ax.set_xticks(x + width)
        ax.set_xticklabels(self.layer_names, fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, axis="y")

        ax = fig.add_subplot(gs[1, 2])
        ds = [weights["layers"][f"layer_{i}"]["deformation_scale"] for i in range(self.num_layers)]
        ax.bar(self.layer_names, ds, color=self.layer_colors, alpha=0.8)
        ax.set_title("δ per Layer", fontsize=10)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(alpha=0.3, axis="y")

        ax = fig.add_subplot(gs[1, 3])
        if "blend_beta" in weights:
            ax.bar(["β"], [weights["blend_beta"]], color="#8e44ad", alpha=0.8)
            ax.axhline(y=0.5, color="gray", linewidth=1, linestyle="--")
            ax.set_ylim(0, 1)
        ax.set_title("Blend", fontsize=10)
        ax.grid(alpha=0.3, axis="y")

        # Row 3: Trained vs fresh
        ax = fig.add_subplot(gs[2, :])
        keys = sorted(comparison["delta"].keys())
        deltas = [comparison["delta"][k] for k in keys]
        colors = ["#27ae60" if d > 0 else "#e74c3c" for d in deltas]
        y_pos = np.arange(len(keys))
        ax.barh(y_pos, deltas, color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(keys, fontsize=7)
        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.set_title("Trained − Fresh Δ", fontsize=10)
        ax.grid(alpha=0.3, axis="x")

        fig.suptitle("KSimplex Post-Training Dashboard", fontsize=14, fontweight="bold", y=0.98)
        fig.savefig(os.path.join(save_dir, "06_dashboard.png"), dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()

    # =========================================================================
    # Helpers
    # =========================================================================

    def _encode_prompts(self, prompts: List[str]) -> torch.Tensor:
        """Encode prompts to CLIP hidden states."""
        from transformers import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        tokens = tokenizer(
            prompts,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(self.device)
        with torch.no_grad():
            enc_hs = self.pipe.clip(input_ids)
        return enc_hs.to(self.dtype)

    def _reinit_geo_prior(self):
        """Reinitialize geo_prior to fresh random weights."""
        cfg = self.unet.simplex_config
        from sd15_trainer_geo.unet.base_simplex import KSimplexCrossAttentionPrior
        fresh = KSimplexCrossAttentionPrior(cfg).to(self.device)
        self.unet.geo_prior.load_state_dict(fresh.state_dict())


# =============================================================================
# Utilities
# =============================================================================

def _make_serializable(obj):
    """Convert numpy/torch types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().tolist()
    return obj