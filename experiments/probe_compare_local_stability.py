"""Compare a SIGReg checkpoint against an endpoint action-reconstruction
(EP-IDM) checkpoint for the SAME example, in one 3-panel figure:

  Panel 1 (eigenvector alignment): BOTH configs' learned dominant
    eigenvector overlaid against the SAME ground-truth stable/unstable
    eigenvectors (crimson = SIG, orange = EP-IDM) -- a direct visual
    comparison of which one actually tracks the true unstable mode.
  Panel 2 / 3: each config's own closed-loop trajectories under its own
    learned controller, side by side (same starting states, same axis
    scale), so a converging vs. diverging outcome is immediately visible.

    python experiments/probe_compare_local_stability.py \\
        --sig-checkpoint results/example1_sigreg_naive/checkpoint_sigreg_naive_H4_seed0.pt \\
        --epidm-checkpoint results/example1_actreconendpoint_naive/checkpoint_actreconendpoint_naive_H4_seed0.pt \\
        --traj-x0 0 80 --traj-x0 25 25 --traj-x0 0 -70 --traj-x0 -20 -50
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
from jepa_lds.diagnostics import closed_loop_trajectory_panel, unstable_eigenvector_alignment
from jepa_lds.plotting import plot_compare_local_stability

_DEFAULT_DIMS = {
    "cartpole_linear": (2, 3),  # pole angle, pole angular velocity
    "double_mode": (0, 1),
    "double_mode_stable": (0, 1),
}
_ANGLE_DIMS = {
    "cartpole_linear": {2, 3},
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sig-checkpoint", type=str, required=True)
    p.add_argument("--epidm-checkpoint", type=str, required=True)
    p.add_argument("--out", type=str, default=None, help="output PDF path (default: alongside --sig-checkpoint)")
    p.add_argument("--dims", type=int, nargs=2, default=None, help="which 2 state dims to probe (default: system-specific)")
    p.add_argument(
        "--degrees", choices=["auto", "on", "off"], default="auto",
        help="display/grid angle-like dims in degrees instead of radians (auto: on for cartpole_linear's default angle dims)",
    )

    p.add_argument("--traj-dim0-max", type=float, default=25.0, help="default trajectory starting-corner magnitude along dims[0]")
    p.add_argument("--traj-dim1-max", type=float, default=80.0, help="default trajectory starting-corner magnitude along dims[1]")
    p.add_argument(
        "--traj-x0", type=float, nargs=2, action="append", default=None,
        help="explicit starting state (dims[0], dims[1] value) for panels 2-3; repeat for multiple trajectories "
        "(overrides the default 4-corner box from --traj-dim0-max/--traj-dim1-max)",
    )
    p.add_argument("--n-steps", type=int, default=300, help="closed-loop rollout length for panels 2-3's trajectories")
    p.add_argument(
        "--traj-substeps", type=int, default=1,
        help="intra-step points to draw per true discrete step in panels 2-3 (default 1 = none). See "
        "probe_local_stability.py's --traj-substeps for why this (not plain line subdivision) is the correct "
        "way to remove the straight-line-interpolation artifact.",
    )
    p.add_argument("--hide-axis-units", action="store_true", help="omit the '(deg)' / unit suffix from axis labels")
    p.add_argument("--eig-legend-loc", type=str, default="lower left", help="matplotlib legend location for panel 1 (e.g. 'upper left')")
    p.add_argument(
        "--traj-xlim", type=float, default=None,
        help="symmetric x-axis half-width for panels 2-3 (default: auto from the starting points' own extent, "
        "which can clip a trajectory's early transient if it swings out further than its own x0)",
    )
    p.add_argument("--traj-ylim", type=float, default=None, help="symmetric y-axis half-width for panels 2-3 (default: auto, see --traj-xlim)")
    p.add_argument("--traj-linewidth", type=float, default=1.2, help="line width for panels 2-3's trajectory curves")

    p.add_argument("--q-scale", type=float, default=1.0, help="LQR Q = q_scale * I (applied to each checkpoint's own latent system)")
    p.add_argument("--r-scale", type=float, default=1.0, help="LQR R = r_scale * I")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()

    torch.set_num_threads(args.threads)

    system_a, obs_model, encoder_a, predictor_a, _dec_a, _cfg_a, extra_a = load_checkpoint_with_env(args.sig_checkpoint)
    system_b, _obs_model_b, encoder_b, predictor_b, _dec_b, _cfg_b, extra_b = load_checkpoint_with_env(args.epidm_checkpoint)
    if system_a.name != system_b.name:
        raise ValueError(
            f"--sig-checkpoint is for system {system_a.name!r} but --epidm-checkpoint is for {system_b.name!r} -- "
            "pass a matching SIG/EP-IDM pair trained on the SAME example"
        )
    system = system_a  # identical ground truth for a matching pair; obs_model likewise (same obs-seed per example)

    dims = tuple(args.dims) if args.dims is not None else _DEFAULT_DIMS.get(system.name, (0, 1))
    if args.degrees == "auto":
        use_degrees = set(dims) <= _ANGLE_DIMS.get(system.name, set())
    else:
        use_degrees = args.degrees == "on"

    def _maybe_deg2rad(v: float) -> float:
        return np.deg2rad(v) if use_degrees else v

    def _corner_x0s(max0: float, max1: float) -> list[np.ndarray]:
        x0s = []
        for s0, s1 in [(1, 1), (-1, 1), (-1, -1), (1, -1)]:
            x0 = np.zeros(system.n)
            x0[dims[0]] = s0 * max0
            x0[dims[1]] = s1 * max1
            x0s.append(x0)
        return x0s

    if args.traj_x0 is not None:
        x0s = []
        for v0, v1 in args.traj_x0:
            x0 = np.zeros(system.n)
            x0[dims[0]] = _maybe_deg2rad(v0)
            x0[dims[1]] = _maybe_deg2rad(v1)
            x0s.append(x0)
    else:
        x0s = _corner_x0s(_maybe_deg2rad(args.traj_dim0_max), _maybe_deg2rad(args.traj_dim1_max))

    print(f"SIG checkpoint:    {args.sig_checkpoint}")
    print(f"EP-IDM checkpoint: {args.epidm_checkpoint}")
    print(f"system={system.name}  dims={dims}  degrees={use_degrees}")

    print("[panel 1] unstable eigenvector alignment (SIG vs EP-IDM)...")
    eig_a = unstable_eigenvector_alignment(system, obs_model, encoder_a, predictor_a, dims=dims)
    eig_b = unstable_eigenvector_alignment(system, obs_model, encoder_b, predictor_b, dims=dims)
    print(f"  SIG:    learned dominant eigenvalue = {eig_a['learned_dominant_eigval']}  cos_sim(true unstable) = {eig_a['cos_sim_unstable']:.4f}")
    print(f"  EP-IDM: learned dominant eigenvalue = {eig_b['learned_dominant_eigval']}  cos_sim(true unstable) = {eig_b['cos_sim_unstable']:.4f}")

    ctrl_a = design_latent_controller(predictor_a, q_scale=args.q_scale, r_scale=args.r_scale)
    ctrl_b = design_latent_controller(predictor_b, q_scale=args.q_scale, r_scale=args.r_scale)
    print(f"  SIG:    stabilizable(latent) = {ctrl_a['stabilizable']}")
    print(f"  EP-IDM: stabilizable(latent) = {ctrl_b['stabilizable']}")
    if not ctrl_a["stabilizable"] or not ctrl_b["stabilizable"]:
        raise ValueError(
            "no stabilizing latent controller could be designed for one of the two checkpoints -- "
            "nothing to plot for panels 2-3"
        )

    print("[panels 2-3] closed-loop trajectories...")
    traj_a = closed_loop_trajectory_panel(
        system, obs_model, encoder_a, ctrl_a["K_z"], x0s, dims=dims, n_steps=args.n_steps, plot_substeps=args.traj_substeps,
    )
    traj_b = closed_loop_trajectory_panel(
        system, obs_model, encoder_b, ctrl_b["K_z"], x0s, dims=dims, n_steps=args.n_steps, plot_substeps=args.traj_substeps,
    )
    for label, traj in (("SIG", traj_a), ("EP-IDM", traj_b)):
        for x0, xs in zip(x0s, traj["trajectories"]):
            print(f"  {label}: x0={x0[list(dims)]}  ->  final in-plane distance={float(np.linalg.norm(xs[-1])):.3g}")

    if use_degrees:
        rad2deg = 180.0 / np.pi
        for traj in (traj_a, traj_b):
            traj["trajectories"] = [xs * rad2deg for xs in traj["trajectories"]]
            traj["x0s"] = [x0 * rad2deg for x0 in traj["x0s"]]

    if system.is_open_loop_unstable():
        dominant_mode_label, other_mode_label = "ground truth (unstable)", "ground truth (stable)"
    else:
        # Both modes are actually stable (e.g. Example 2's double_mode_stable) --
        # calling the dominant one "unstable" would be wrong, so number them instead.
        dominant_mode_label, other_mode_label = "ground truth (stable 2)", "ground truth (stable 1)"

    out_path = args.out or os.path.splitext(args.sig_checkpoint)[0] + "_vs_epidm_compare.pdf"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    unit_suffix = "" if args.hide_axis_units else ("deg" if use_degrees else "")
    plot_compare_local_stability(
        eig_a, eig_b, traj_a, traj_b, out_path,
        label_a="SIG", label_b="EP-IDM", unit_suffix=unit_suffix,
        dominant_mode_label=dominant_mode_label, other_mode_label=other_mode_label,
        eig_legend_loc=args.eig_legend_loc, traj_xlim=args.traj_xlim, traj_ylim=args.traj_ylim,
        traj_linewidth=args.traj_linewidth,
    )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
