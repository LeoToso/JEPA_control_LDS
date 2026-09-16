"""Plotting helpers for the experiment scripts. All functions save a PDF/PNG
to `out_path` and also return the Figure in case the caller wants it."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Shared marker style for the origin/equilibrium point, kept identical across all three
# panels (and readable against green/red, plasma, and RdBu_r backgrounds alike).
_EQUILIBRIUM_STYLE = dict(marker="*", s=220, c="white", edgecolors="black", linewidths=1.3, zorder=6)


def _axis_label(dim: int, suffix: str) -> str:
    return f"$x_{{{dim + 1}}}${suffix}"


_TRAJ_COLORS = ["tab:blue", "tab:orange", "tab:purple", "tab:brown", "tab:cyan", "tab:pink"]


def _draw_eigenvector_alignment_panel(
    ax,
    eig_panel: dict,
    suffix: str = "",
    dominant_mode_label: str = "ground truth (unstable)",
    other_mode_label: str = "ground truth (stable)",
    label_fontsize: float = 20,
    tick_labelsize: float = 15,
    legend_fontsize: float = 15,
    legend_loc: str = "lower left",
):
    """Draws the eigenvector-alignment panel (true unstable eigenvector vs.
    learned dominant eigenvector, both unit vectors projected onto
    `eig_panel["dims"]`, with the true "other" eigenvector as a grey
    reference when available) onto a single axes -- shared by
    `plot_local_stability_probe`'s panel 1 and `plot_eigenvector_alignment_grid`.

    Thin shafts/heads so two nearly-parallel unit arrows don't visually
    merge into one blob; "learned" is drawn semi-transparent (and on top)
    so a perfectly-aligned overlap shows as a visibly blended color rather
    than fully hiding the ground-truth arrow underneath."""
    d0, d1 = eig_panel["dims"]
    arrow_kwargs = dict(angles="xy", scale_units="xy", scale=1, width=0.012, headwidth=3.5, headlength=4.5, headaxislength=4)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="grey", linewidth=0.7, linestyle=":", alpha=0.6)
    if eig_panel.get("v_true_stable") is not None:
        vs = eig_panel["v_true_stable"]
        ax.quiver(0, 0, vs[0], vs[1], color="grey", alpha=0.7, label=other_mode_label, **arrow_kwargs)
    vu = eig_panel["v_true_unstable"]
    vl = eig_panel["v_learned"]
    ax.quiver(0, 0, vu[0], vu[1], color="green", label=dominant_mode_label, zorder=4, **arrow_kwargs)
    ax.quiver(0, 0, vl[0], vl[1], color="darkorange", alpha=0.6, label="learned (dominant)", zorder=5, **arrow_kwargs)
    ax.scatter([0], [0], **_EQUILIBRIUM_STYLE)
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_aspect("equal")
    ax.set_xlabel(_axis_label(d0, suffix), fontsize=label_fontsize)
    ax.set_ylabel(_axis_label(d1, suffix), fontsize=label_fontsize)
    ax.legend(fontsize=legend_fontsize, loc=legend_loc)
    ax.tick_params(labelsize=tick_labelsize)


def _draw_eigenvector_alignment_panel_dual(
    ax,
    eig_panel_a: dict,
    eig_panel_b: dict,
    label_a: str,
    label_b: str,
    suffix: str = "",
    dominant_mode_label: str = "ground truth (unstable)",
    other_mode_label: str = "ground truth (stable)",
    label_fontsize: float = 20,
    tick_labelsize: float = 15,
    legend_fontsize: float = 15,
    legend_loc: str = "lower left",
):
    """Like `_draw_eigenvector_alignment_panel`, but overlays TWO learned
    eigenvector arrows (e.g. from a SIGReg-trained vs. an endpoint-action-
    reconstruction-trained encoder/predictor on the SAME ground-truth
    system) against the SAME ground-truth stable/unstable arrows -- a direct
    visual comparison of which config's learned dominant direction actually
    tracks the true unstable mode. `eig_panel_a`/`eig_panel_b` are both
    `unstable_eigenvector_alignment` outputs for the SAME system/dims (only
    their `v_learned` differs); ground truth is taken from `eig_panel_a`."""
    d0, d1 = eig_panel_a["dims"]
    arrow_kwargs = dict(angles="xy", scale_units="xy", scale=1, width=0.012, headwidth=3.5, headlength=4.5, headaxislength=4)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="grey", linewidth=0.7, linestyle=":", alpha=0.6)
    if eig_panel_a.get("v_true_stable") is not None:
        vs = eig_panel_a["v_true_stable"]
        ax.quiver(0, 0, vs[0], vs[1], color="grey", alpha=0.7, label=other_mode_label, **arrow_kwargs)
    vu = eig_panel_a["v_true_unstable"]
    ax.quiver(0, 0, vu[0], vu[1], color="green", label=dominant_mode_label, zorder=4, **arrow_kwargs)
    va = eig_panel_a["v_learned"]
    vb = eig_panel_b["v_learned"]
    ax.quiver(0, 0, va[0], va[1], color="crimson", alpha=0.65, label=f"learned ({label_a})", zorder=5, **arrow_kwargs)
    ax.quiver(0, 0, vb[0], vb[1], color="darkorange", alpha=0.65, label=f"learned ({label_b})", zorder=5, **arrow_kwargs)
    ax.scatter([0], [0], **_EQUILIBRIUM_STYLE)
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_aspect("equal")
    ax.set_xlabel(_axis_label(d0, suffix), fontsize=label_fontsize)
    ax.set_ylabel(_axis_label(d1, suffix), fontsize=label_fontsize)
    ax.legend(fontsize=legend_fontsize, loc=legend_loc)
    ax.tick_params(labelsize=tick_labelsize)


def _draw_trajectory_panel(
    ax,
    traj_panel: dict,
    suffix: str = "",
    label_fontsize: float = 20,
    tick_labelsize: float = 15,
    legend_fontsize: float = 15,
    legend_loc: str = "lower left",
    title: str | None = None,
    xlim: float | None = None,
    ylim: float | None = None,
):
    """Draws the closed-loop-trajectories panel (a handful of trajectories
    under the learned controller, starting from marked initial states,
    converging to the equilibrium vs. diverging away from it) onto a single
    axes -- shared by `plot_local_stability_probe`'s panel 2 and
    `plot_compare_local_stability`.

    `xlim`/`ylim` (default None): symmetric half-width override for the
    displayed axes (i.e. the axes span `(-xlim, xlim)`/`(-ylim, ylim)`).
    By default the half-width is auto-computed from the starting points'
    own extent (with a 15% margin), which can clip a trajectory's early
    transient if it swings out further than its own x0 -- pass an explicit
    value here to widen the frame instead."""
    d0t, d1t = traj_panel["dims"]
    for i, (xs, x0) in enumerate(zip(traj_panel["trajectories"], traj_panel["x0s"])):
        color = _TRAJ_COLORS[i % len(_TRAJ_COLORS)]
        ax.plot(xs[:, 0], xs[:, 1], color=color, linewidth=1.2, alpha=0.85, zorder=3)
        ax.scatter([x0[0]], [x0[1]], facecolors="none", edgecolors=color, marker="o", s=90, linewidths=1.8, zorder=4)
    ax.scatter([0], [0], **_EQUILIBRIUM_STYLE)
    x0s_arr = np.array(traj_panel["x0s"])
    lim0 = xlim if xlim is not None else (1.15 * np.max(np.abs(x0s_arr[:, 0])) if np.any(x0s_arr[:, 0] != 0) else 1.0)
    lim1 = ylim if ylim is not None else (1.15 * np.max(np.abs(x0s_arr[:, 1])) if np.any(x0s_arr[:, 1] != 0) else 1.0)
    ax.set_xlim(-lim0, lim0)
    ax.set_ylim(-lim1, lim1)
    legend_handles = [
        Line2D([0], [0], marker="o", markerfacecolor="none", markeredgecolor="black", linestyle="none", markersize=8, label="start ($x_0$)"),
        Line2D([0], [0], color="black", linewidth=1.2, label="closed-loop trajectory"),
    ]
    ax.legend(handles=legend_handles, fontsize=legend_fontsize, loc=legend_loc)
    ax.set_xlabel(_axis_label(d0t, suffix), fontsize=label_fontsize)
    ax.set_ylabel(_axis_label(d1t, suffix), fontsize=label_fontsize)
    ax.tick_params(labelsize=tick_labelsize)
    if title:
        ax.set_title(title, fontsize=label_fontsize)


def _draw_lyapunov_panel(
    ax,
    fig,
    lyap_panel: dict,
    suffix: str = "",
    label_fontsize: float = 20,
    tick_labelsize: float = 15,
    cbar_label_fontsize: float = 20,
    title: str | None = None,
):
    """Draws the Lyapunov-decrease certificate heatmap (pointwise sign of
    Delta V under the learned controller, against the ground-truth Lyapunov
    function) onto a single axes -- shared by `plot_local_stability_probe`'s
    panel 3 and `plot_lyapunov_grid`."""
    XX3, YY3 = lyap_panel["XX"], lyap_panel["YY"]
    d0l, d1l = lyap_panel["dims"]
    vmax = float(np.percentile(np.abs(lyap_panel["delta_V"]), 95))
    vmax = vmax if vmax > 1e-12 else 1.0
    cf = ax.pcolormesh(XX3, YY3, lyap_panel["delta_V"], cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    ax.contour(XX3, YY3, lyap_panel["delta_V"], levels=[0.0], colors="black", linewidths=1.2)
    ax.scatter([0], [0], **_EQUILIBRIUM_STYLE)
    cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\Delta V = V(x') - V(x)$", fontsize=cbar_label_fontsize)
    cbar.ax.tick_params(labelsize=tick_labelsize)
    ax.set_xlabel(_axis_label(d0l, suffix), fontsize=label_fontsize)
    ax.set_ylabel(_axis_label(d1l, suffix), fontsize=label_fontsize)
    ax.tick_params(labelsize=tick_labelsize)
    if title:
        ax.set_title(title, fontsize=label_fontsize)


def plot_compare_local_stability(
    eig_panel_a: dict,
    eig_panel_b: dict,
    traj_panel_a: dict,
    traj_panel_b: dict,
    out_path: str,
    label_a: str = "SIG",
    label_b: str = "EP-IDM",
    unit_suffix: str = "",
    dominant_mode_label: str = "ground truth (unstable)",
    other_mode_label: str = "ground truth (stable)",
    eig_legend_loc: str = "lower left",
    traj_xlim: float | None = None,
    traj_ylim: float | None = None,
):
    """3-panel side-by-side comparison of two training configs (e.g. SIGReg
    vs. endpoint action-reconstruction) on the SAME example: panel 1 overlays
    BOTH configs' learned dominant eigenvector against the shared ground
    truth (`_draw_eigenvector_alignment_panel_dual`); panels 2-3 are each
    config's own closed-loop trajectories, side by side, so a converging vs.
    diverging outcome is directly visible in one figure.

    `traj_xlim`/`traj_ylim`: shared symmetric axis half-width override for
    BOTH trajectory panels (see `_draw_trajectory_panel`) -- useful when the
    auto-computed frame (from the starting points' own extent) clips an
    early transient that swings out further than its own x0."""
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    suffix = f" ({unit_suffix})" if unit_suffix else ""
    _draw_eigenvector_alignment_panel_dual(
        axes[0], eig_panel_a, eig_panel_b, label_a, label_b, suffix=suffix,
        dominant_mode_label=dominant_mode_label, other_mode_label=other_mode_label, legend_loc=eig_legend_loc,
    )
    _draw_trajectory_panel(axes[1], traj_panel_a, suffix=suffix, title=label_a, xlim=traj_xlim, ylim=traj_ylim)
    _draw_trajectory_panel(axes[2], traj_panel_b, suffix=suffix, title=label_b, xlim=traj_xlim, ylim=traj_ylim)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return fig


def plot_lyapunov_grid(
    panels: list[dict],
    titles: list[str],
    out_path: str,
    unit_suffixes: list[str] | None = None,
    ncols: int = 2,
):
    """A grid of Lyapunov-decrease panels (see `plot_local_stability_probe`'s
    panel 3 / `_draw_lyapunov_panel`), one subplot per (panel, title) pair --
    e.g. every (example, config) combination laid out for a direct
    side-by-side comparison."""
    n = len(panels)
    if n == 0:
        raise ValueError("panels must be non-empty")
    if len(titles) != n:
        raise ValueError("titles must have the same length as panels")
    if unit_suffixes is None:
        unit_suffixes = [""] * n
    ncols = max(1, min(ncols, n))
    nrows = -(-n // ncols)  # ceil division
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5 * ncols, 6.5 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    for ax, lyap_panel, title, unit_suffix in zip(axes_flat, panels, titles, unit_suffixes):
        suffix = f" ({unit_suffix})" if unit_suffix else ""
        _draw_lyapunov_panel(ax, fig, lyap_panel, suffix=suffix, title=title)
    for ax in axes_flat[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return fig


def plot_eigenvector_alignment_grid(
    eig_panels: list[dict],
    out_path: str,
    unit_suffix: str = "",
    dominant_mode_label: str = "ground truth (unstable)",
    other_mode_label: str = "ground truth (stable)",
    ncols: int = 3,
):
    """A grid of eigenvector-alignment panels (see `plot_local_stability_probe`'s
    panel 1 / `_draw_eigenvector_alignment_panel`), one subplot per entry in
    `eig_panels` -- typically one per (dims[0], dims[1]) pair covering every
    2D projection of an n-state system, e.g. via
    `itertools.combinations(range(system.n), 2)`, rather than a single fixed
    --dims choice. Gives a full picture of where the learned and true
    dominant directions do (and don't) align across the whole state space."""
    n = len(eig_panels)
    if n == 0:
        raise ValueError("eig_panels must be non-empty")
    ncols = max(1, min(ncols, n))
    nrows = -(-n // ncols)  # ceil division
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    suffix = f" ({unit_suffix})" if unit_suffix else ""
    for ax, eig_panel in zip(axes_flat, eig_panels):
        _draw_eigenvector_alignment_panel(
            ax, eig_panel, suffix=suffix, dominant_mode_label=dominant_mode_label,
            other_mode_label=other_mode_label, label_fontsize=14, tick_labelsize=11,
            legend_fontsize=9, legend_loc="best",
        )
    for ax in axes_flat[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return fig


def plot_local_stability_probe(
    eig_panel: dict,
    traj_panel: dict,
    lyap_panel: dict | None,
    out_path: str,
    unit_suffix: str = "",
    dominant_mode_label: str = "ground truth (unstable)",
    other_mode_label: str = "ground truth (stable)",
):
    """The local-stability probe: unstable-eigenvector alignment (true vs.
    learned, projected back to state space), a handful of closed-loop
    trajectories under the learned controller starting from marked initial
    states (converging to the equilibrium vs. diverging away from it), and
    -- unless `lyap_panel` is None -- a Lyapunov-decrease certificate map.

    Deliberately title-free (no per-panel title, no figure suptitle) --
    the quantitative summary (cosine similarities, success rates, decrease
    fraction) is printed to the console by the calling script instead,
    keeping the figure itself uncluttered.

    `dominant_mode_label`/`other_mode_label` name panel 1's two ground-truth
    eigenvector arrows -- default to "unstable"/"stable" (for a system with
    a genuine unstable mode); pass e.g. "ground truth (stable 2)" /
    "ground truth (stable 1)" for a system where both modes are stable,
    where calling the dominant one "unstable" would be wrong.

    Pass `lyap_panel=None` to render only panels 1-2 (a 1x2 figure) and
    skip the Lyapunov certificate entirely."""
    n_panels = 3 if lyap_panel is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(19 if n_panels == 3 else 13, 6))
    suffix = f" ({unit_suffix})" if unit_suffix else ""

    # Panel 1: unstable eigenvector alignment -- true (green) vs learned (orange),
    # both unit vectors, with the true stable eigenvector (grey, dashed) as a
    # reference so it's visually obvious if "learned" has aligned with the wrong mode.
    _draw_eigenvector_alignment_panel(
        axes[0], eig_panel, suffix=suffix, dominant_mode_label=dominant_mode_label, other_mode_label=other_mode_label,
    )

    # Panel 2: closed-loop trajectories from a handful of marked initial
    # states -- converging to the equilibrium (star) or diverging away from
    # it, under the LEARNED controller. Axis limits are set from the
    # starting points' own extent (with a margin), so a trajectory that
    # diverges simply exits the visible frame rather than blowing out the
    # scale for everything else.
    _draw_trajectory_panel(axes[1], traj_panel, suffix=suffix)

    # Panel 3: Lyapunov decrease (skipped entirely when lyap_panel is None).
    if lyap_panel is not None:
        _draw_lyapunov_panel(axes[2], fig, lyap_panel, suffix=suffix)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return fig
