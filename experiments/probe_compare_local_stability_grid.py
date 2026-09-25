"""Stack two examples' SIG-vs-EP-IDM 3-panel comparisons (see
probe_compare_local_stability.py) into one 2-row x 3-column figure -- top
row for the first example, bottom row for the second. Panel 2/3 titles
("SIG"/"EP-IDM", bold) are shown only on the top row, and x-axis labels
only on the bottom row, since a stacked grid reads top-to-bottom and the
title/label would otherwise just repeat.

    python experiments/probe_compare_local_stability_grid.py \\
        --ex1-sig-checkpoint results/example1_sigreg_naive/checkpoint_sigreg_naive_H4_seed0.pt \\
        --ex1-epidm-checkpoint results/example1_actreconendpoint_naive/checkpoint_actreconendpoint_naive_H4_seed0.pt \\
        --ex3-sig-checkpoint results/example3_sigreg_naive/checkpoint_sigreg_naive_H8_seed0.pt \\
        --ex3-epidm-checkpoint results/example3_actreconendpoint_naive/checkpoint_actreconendpoint_naive_H8_seed0.pt \\
        --traj-x0 -20 80 --traj-x0 25 25 --traj-x0 10 -60 --traj-x0 -5 -10 \\
        --traj-linewidth 2.5 --traj-substeps 20
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
from jepa_lds.plotting import plot_compare_local_stability_grid

_DEFAULT_DIMS = {
    "cartpole_linear": (2, 3),  # pole angle, pole angular velocity
    "double_mode": (0, 1),
    "double_mode_stable": (0, 1),
}
_ANGLE_DIMS = {
    "cartpole_linear": {2, 3},
}


def _build_row(sig_checkpoint: str, epidm_checkpoint: str, args) -> dict:
    system_a, obs_model, encoder_a, predictor_a, _dec_a, _cfg_a, _extra_a = load_checkpoint_with_env(sig_checkpoint)
    system_b, _obs_model_b, encoder_b, predictor_b, _dec_b, _cfg_b, _extra_b = load_checkpoint_with_env(epidm_checkpoint)
    if system_a.name != system_b.name:
        raise ValueError(
            f"{sig_checkpoint} is for system {system_a.name!r} but {epidm_checkpoint} is for {system_b.name!r} -- "
            "pass a matching SIG/EP-IDM pair trained on the SAME example"
        )
    system = system_a

    dims = _DEFAULT_DIMS.get(system.name, (0, 1))
    use_degrees = set(dims) <= _ANGLE_DIMS.get(system.name, set())

    def _maybe_deg2rad(v: float) -> float:
        return np.deg2rad(v) if use_degrees else v

    x0s = []
    for v0, v1 in args.traj_x0:
        x0 = np.zeros(system.n)
        x0[dims[0]] = _maybe_deg2rad(v0)
        x0[dims[1]] = _maybe_deg2rad(v1)
        x0s.append(x0)

    print(f"[{system.name}]  SIG={sig_checkpoint}  EP-IDM={epidm_checkpoint}")
    print(f"  dims={dims}  degrees={use_degrees}")
    eig_a = unstable_eigenvector_alignment(system, obs_model, encoder_a, predictor_a, dims=dims)
    eig_b = unstable_eigenvector_alignment(system, obs_model, encoder_b, predictor_b, dims=dims)
    print(f"  SIG:    learned dominant eigenvalue = {eig_a['learned_dominant_eigval']}  cos_sim(true unstable) = {eig_a['cos_sim_unstable']:.4f}")
    print(f"  EP-IDM: learned dominant eigenvalue = {eig_b['learned_dominant_eigval']}  cos_sim(true unstable) = {eig_b['cos_sim_unstable']:.4f}")

    ctrl_a = design_latent_controller(predictor_a, q_scale=args.q_scale, r_scale=args.r_scale)
    ctrl_b = design_latent_controller(predictor_b, q_scale=args.q_scale, r_scale=args.r_scale)
    if not ctrl_a["stabilizable"] or not ctrl_b["stabilizable"]:
        raise ValueError(
            f"no stabilizing latent controller could be designed for one of {sig_checkpoint}/{epidm_checkpoint} -- "
            "nothing to plot for panels 2-3"
        )

    traj_a = closed_loop_trajectory_panel(
        system, obs_model, encoder_a, ctrl_a["K_z"], x0s, dims=dims, n_steps=args.n_steps, plot_substeps=args.traj_substeps,
    )
    traj_b = closed_loop_trajectory_panel(
        system, obs_model, encoder_b, ctrl_b["K_z"], x0s, dims=dims, n_steps=args.n_steps, plot_substeps=args.traj_substeps,
    )

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

    unit_suffix = "" if args.hide_axis_units else ("deg" if use_degrees else "")
    return dict(
        eig_panel_a=eig_a, eig_panel_b=eig_b, traj_panel_a=traj_a, traj_panel_b=traj_b,
        label_a="SIG", label_b="EP-IDM", unit_suffix=unit_suffix,
        dominant_mode_label=dominant_mode_label, other_mode_label=other_mode_label,
        eig_legend_loc=args.eig_legend_loc, traj_xlim=args.traj_xlim, traj_ylim=args.traj_ylim,
        traj_linewidth=args.traj_linewidth,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ex1-sig-checkpoint", type=str, required=True)
    p.add_argument("--ex1-epidm-checkpoint", type=str, required=True)
    p.add_argument("--ex3-sig-checkpoint", type=str, required=True)
    p.add_argument("--ex3-epidm-checkpoint", type=str, required=True)
    p.add_argument("--out", type=str, default="results/compare_local_stability_grid_example1_example3.pdf")

    p.add_argument(
        "--traj-x0", type=float, nargs=2, action="append", required=True,
        help="explicit starting state (dims[0], dims[1] value) for panels 2-3, shared by BOTH rows "
        "(each row's own dims, e.g. cartpole's angle dims, gets these same numbers -- in that row's own "
        "display units); repeat for multiple trajectories",
    )
    p.add_argument("--n-steps", type=int, default=300, help="closed-loop rollout length for panels 2-3's trajectories, shared by both rows")
    p.add_argument(
        "--traj-substeps", type=int, default=1,
        help="intra-step points to draw per true discrete step in panels 2-3, shared by both rows (default 1 = "
        "none). See probe_local_stability.py's --traj-substeps for details.",
    )
    p.add_argument("--traj-linewidth", type=float, default=1.2, help="line width for panels 2-3's trajectory curves, shared by both rows")
    p.add_argument(
        "--traj-xlim", type=float, default=None,
        help="symmetric x-axis half-width for panels 2-3, shared by both rows (default: auto per row)",
    )
    p.add_argument("--traj-ylim", type=float, default=None, help="symmetric y-axis half-width for panels 2-3, shared by both rows (default: auto)")
    p.add_argument("--eig-legend-loc", type=str, default="lower left", help="matplotlib legend location for panel 1, both rows")
    p.add_argument("--hide-axis-units", action="store_true", help="omit the '(deg)' / unit suffix from axis labels, both rows")

    p.add_argument("--label-fontsize", type=float, default=20, help="axis-label font size, every panel")
    p.add_argument("--tick-labelsize", type=float, default=15, help="axis tick-number font size, every panel")
    p.add_argument("--legend-fontsize", type=float, default=15, help="legend text font size, every panel")
    p.add_argument(
        "--title-fontsize", type=float, default=None,
        help="panel 2-3's bold 'SIG'/'EP-IDM' title font size on the top row (default: same as --label-fontsize)",
    )

    p.add_argument("--q-scale", type=float, default=1.0, help="LQR Q = q_scale * I (applied to each checkpoint's own latent system)")
    p.add_argument("--r-scale", type=float, default=1.0, help="LQR R = r_scale * I")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()

    torch.set_num_threads(args.threads)

    row1 = _build_row(args.ex1_sig_checkpoint, args.ex1_epidm_checkpoint, args)
    row3 = _build_row(args.ex3_sig_checkpoint, args.ex3_epidm_checkpoint, args)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plot_compare_local_stability_grid(
        [row1, row3], args.out,
        label_fontsize=args.label_fontsize, tick_labelsize=args.tick_labelsize,
        legend_fontsize=args.legend_fontsize, title_fontsize=args.title_fontsize,
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
