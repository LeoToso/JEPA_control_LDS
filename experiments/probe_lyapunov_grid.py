"""4-panel grid of Lyapunov-decrease certificates comparing SIGReg vs.
endpoint action-reconstruction (EP-IDM), for Examples 1 and 3:
(Example 1, SIG), (Example 1, EP-IDM), (Example 3, SIG), (Example 3, EP-IDM).

Each panel is exactly `probe_local_stability.py`'s panel 3: pointwise sign
of Delta V = V(x') - V(x) after one closed-loop step under that checkpoint's
own learned controller, where V(x) = x^T P x is the TRUE system's own
quadratic Lyapunov function -- an honest, model-free check of whether the
learned controller's action is a valid descent direction for the true
system's own cost.

    python experiments/probe_lyapunov_grid.py \\
        --ex1-sig-checkpoint results/example1_sigreg_naive/checkpoint_sigreg_naive_H4_seed0.pt \\
        --ex1-epidm-checkpoint results/example1_actreconendpoint_naive/checkpoint_actreconendpoint_naive_H4_seed0.pt \\
        --ex3-sig-checkpoint results/example3_sigreg_naive/checkpoint_sigreg_naive_H8_seed0.pt \\
        --ex3-epidm-checkpoint results/example3_actreconendpoint_naive/checkpoint_actreconendpoint_naive_H8_seed0.pt
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from checkpoint_io import load_checkpoint_with_env
from jepa_lds.control import design_latent_controller
from jepa_lds.diagnostics import lyapunov_decrease_panel
from jepa_lds.plotting import plot_lyapunov_grid

_DEFAULT_DIMS = {
    "cartpole_linear": (2, 3),  # pole angle, pole angular velocity
    "double_mode": (0, 1),
    "double_mode_stable": (0, 1),
}
_ANGLE_DIMS = {
    "cartpole_linear": {2, 3},
}


def _build_panel(checkpoint_path: str, args) -> tuple[dict, str]:
    system, obs_model, encoder, predictor, _dec, _cfg, _extra = load_checkpoint_with_env(checkpoint_path)
    dims = _DEFAULT_DIMS.get(system.name, (0, 1))
    use_degrees = set(dims) <= _ANGLE_DIMS.get(system.name, set())

    ctrl = design_latent_controller(predictor, q_scale=args.q_scale, r_scale=args.r_scale)
    if not ctrl["stabilizable"]:
        raise ValueError(f"no stabilizing latent controller could be designed for {checkpoint_path} -- nothing to plot")
    K_z = ctrl["K_z"]
    _K_gt, P_gt = system.dlqr(args.q_scale * np.eye(system.n), args.r_scale * np.eye(system.m))

    if use_degrees:
        lo = (-np.deg2rad(args.lyap_dim0_max), -np.deg2rad(args.lyap_dim1_max))
        hi = (np.deg2rad(args.lyap_dim0_max), np.deg2rad(args.lyap_dim1_max))
    else:
        lo = (-args.lyap_dim0_max, -args.lyap_dim1_max)
        hi = (args.lyap_dim0_max, args.lyap_dim1_max)

    lyap_panel = lyapunov_decrease_panel(
        system, obs_model, encoder, K_z, P_gt, dims=dims, lo=lo, hi=hi,
        n_points=(args.lyap_n_dim0, args.lyap_n_dim1),
    )
    if use_degrees:
        rad2deg = 180.0 / np.pi
        lyap_panel["XX"] = lyap_panel["XX"] * rad2deg
        lyap_panel["YY"] = lyap_panel["YY"] * rad2deg
    unit_suffix = "" if args.hide_axis_units else ("deg" if use_degrees else "")
    print(f"  {checkpoint_path}\n    fraction of grid satisfying decrease = {lyap_panel['frac_decrease']*100:.1f}%")
    return lyap_panel, unit_suffix


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ex1-sig-checkpoint", type=str, required=True)
    p.add_argument("--ex1-epidm-checkpoint", type=str, required=True)
    p.add_argument("--ex3-sig-checkpoint", type=str, required=True)
    p.add_argument("--ex3-epidm-checkpoint", type=str, required=True)
    p.add_argument("--out", type=str, default="results/lyapunov_grid_example1_example3.pdf")

    p.add_argument("--lyap-dim0-max", type=float, default=15.0, help="Lyapunov grid half-width along dims[0], applied to every panel")
    p.add_argument("--lyap-dim1-max", type=float, default=50.0, help="Lyapunov grid half-width along dims[1], applied to every panel")
    p.add_argument("--lyap-n-dim0", type=int, default=31)
    p.add_argument("--lyap-n-dim1", type=int, default=25)
    p.add_argument("--hide-axis-units", action="store_true", help="omit the '(deg)' unit suffix from axis labels")

    p.add_argument("--q-scale", type=float, default=1.0, help="LQR Q = q_scale * I, applied to each checkpoint's own latent+oracle systems")
    p.add_argument("--r-scale", type=float, default=1.0, help="LQR R = r_scale * I")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()

    torch.set_num_threads(args.threads)

    checkpoints = [
        ("Example 1 (SIG)", args.ex1_sig_checkpoint),
        ("Example 1 (EP-IDM)", args.ex1_epidm_checkpoint),
        ("Example 3 (SIG)", args.ex3_sig_checkpoint),
        ("Example 3 (EP-IDM)", args.ex3_epidm_checkpoint),
    ]
    panels, titles, unit_suffixes = [], [], []
    for title, ckpt in checkpoints:
        print(f"[{title}]")
        panel, unit_suffix = _build_panel(ckpt, args)
        panels.append(panel)
        titles.append(title)
        unit_suffixes.append(unit_suffix)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plot_lyapunov_grid(panels, titles, args.out, unit_suffixes=unit_suffixes, ncols=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
