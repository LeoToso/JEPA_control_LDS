"""Hankel/subspace-identification-style filtering architecture: predicts a
FUTURE window of raw observations from a PAST window of embedded
observations via an explicit rank-constrained bottleneck (G1, M), instead of
this codebase's usual recursive (A_z, B_z) latent predictor.

This is the FILTERING/observability dual of the rest of the repo's
stabilization/controllability story: no actions are involved at all, and
the system doesn't need to be open-loop unstable. The question is whether a
representation trained with the standard "multistep prediction + SIGReg"
recipe can discard an OBSERVABLE mode of the system -- making it impossible
to ever reconstruct/filter the true state from the latent sequence -- and
whether an explicit rank regularizer (RankReg) on the predictor's bottleneck
provably prevents that, the way SIGReg (a pure marginal-distribution
penalty) cannot.

IMPORTANT: RankReg must be computable from ONLY the learnable matrices
(G_1, M, E) and empirical data statistics -- NEVER the true system matrices
(A, C), which JEPA training has no access to.

This follows the notes' reduced (E, G_1, M) formulation (G_2 eliminated via
the weight-sharing tie G_2 = D_H(E) M) exactly:

    min_{E,G1,M}  (1/2T) sum_t || D_H(E) y_{t:t+H-1} - D_H(E) M G1 D_L(E) z_t ||_F^2
    s.t.   G1 G1^T                        >= gamma1^2 I
           M^T D_H(E)^T D_H(E) M          >= gamma2^2 I     (= G2^T G2)
           E Sigma_y E^T = I

The first two constraints are PURE MATRIX conditions -- no reference to the
data distribution at all. Constraining G_1 (or M/G2) alone, with no
anchor tying E to the data, is NOT sufficient on its own: G_1 can be
abstractly full-rank while the REAL data it receives only explores a
lower-dimensional slice of its domain (exactly what happens when E has
already discarded a mode upstream) -- an early version of this module tried
exactly that and it failed. The notes close that loophole differently: the
THIRD constraint, E Sigma_y E^T = I, pins E's own output to be exactly
isotropic under the real data distribution, one layer upstream of G_1/M.
Once that anchor exists, G_1 and M (via D_H(E)) can safely be floored as
plain matrices, since a collapsed E could never satisfy the whitening
constraint in the first place.

We implement the E Sigma_y E^T = I equality constraint as the existing soft
SIGReg penalty (`losses.sigreg_loss`) rather than a hard constraint -- so
"RankReg" here is SIGReg (anchoring E to the data) PLUS hinge penalties on
the two bottleneck floors (`rank_reg_loss`, below), not an alternative to
SIGReg. SIGReg alone (no floors) is the baseline this is meant to improve
on: it anchors E but has no mechanism to prevent a *particular* observable
mode from being discarded from the subspace E does keep.

See experiments/example4_filtering/ for the full derivation and the
synthetic construction used to trigger the collapse.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import EpisodeBatch, HankelWindowDataset
from .diagnostics import r2_score, ridge_fit, ridge_predict
from .losses import encode_window, sigreg_loss
from .models import LinearEncoder
from .systems import LTISystem


class HankelPredictor(nn.Module):
    """Predicts a FUTURE window of RAW observations from a PAST window of
    EMBEDDED observations, via an explicit rank-constrained bottleneck: G1
    compresses the (already-embedded) past window z_bar in R^{L*d} down to
    an n-dim state estimate, and M expands that state estimate back out to a
    predicted RAW future window in R^{H*p}. The caller re-embeds both the
    prediction and the true future window with the SAME encoder used to
    build z_bar before comparing (see `hankel_prediction_loss`) -- this is
    the weight-sharing tie G_2 = D_H(E) M from the derivation, applied
    without ever materializing G_2 itself.

    `bottleneck_dim` (n) is deliberately < system.n -- the same undercomplete
    -latent trick used throughout this codebase -- so the windowed
    prediction loss alone ties exactly at zero between multiple candidate
    n-dimensional subspaces of the true state, leaving it to SIGReg/RankReg
    to decide which one survives."""

    def __init__(self, latent_dim: int, obs_dim: int, past_window: int, future_window: int, bottleneck_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.past_window = past_window
        self.future_window = future_window
        self.bottleneck_dim = bottleneck_dim
        self.G1 = nn.Linear(latent_dim * past_window, bottleneck_dim, bias=False)
        self.M = nn.Linear(bottleneck_dim, obs_dim * future_window, bias=False)

    def forward(self, z_past: torch.Tensor) -> torch.Tensor:
        """z_past: (batch, L, d) embedded past window. Returns a predicted
        RAW future observation window (batch, H, p)."""
        b = z_past.shape[0]
        state = self.G1(z_past.reshape(b, -1))
        y_hat = self.M(state)
        return y_hat.reshape(b, self.future_window, self.obs_dim)

    @torch.no_grad()
    def g1_matrix(self) -> np.ndarray:
        return self.G1.weight.detach().cpu().numpy()

    def g2_embedded(self, encoder) -> torch.Tensor:
        """G2 = D_H(E) @ M (the weight-shared decoding map, state -> embedded
        future window), computed WITHOUT ever materializing the
        block-diagonal D_H(E): reshape M's weight into its H per-timestep
        blocks and apply E independently to each. Shape
        (future_window * latent_dim, bottleneck_dim)."""
        blocks = self.M.weight.reshape(self.future_window, self.obs_dim, self.bottleneck_dim)  # (H, p, n_hat)
        E = encoder.linear.weight  # (d, p)
        embedded = torch.einsum("dp,Hpb->Hdb", E, blocks)  # (H, d, n_hat)
        return embedded.reshape(self.future_window * E.shape[0], self.bottleneck_dim)


def hankel_prediction_loss(predictor: HankelPredictor, encoder, z_past: torch.Tensor, y_future: torch.Tensor) -> torch.Tensor:
    """Compares the TRUE and PREDICTED future observation windows in
    EMBEDDED space (both re-embedded with the SAME encoder used to build
    z_past) -- implements the weight-sharing tie G_2 = D_H(E) M."""
    y_hat_future = predictor(z_past)
    z_future_true = encode_window(encoder, y_future)
    z_future_hat = encode_window(encoder, y_hat_future)
    err = z_future_hat - z_future_true
    return (err ** 2).sum(dim=-1).mean()


def rank_reg_loss(predictor: HankelPredictor, encoder, gamma1: float, gamma2: float) -> torch.Tensor:
    """Hinge penalty on the two bottleneck rank-floor constraints from the
    notes' reduced (E, G_1, M) formulation:

        G1 G1^T                  >= gamma1^2 I
        M^T D_H(E)^T D_H(E) M    >= gamma2^2 I     (= G2^T G2, G2 = D_H(E) M)

    Both are PURE MATRIX conditions on the learnable matrices alone -- no
    data covariance involved. That's only valid because `E` is separately
    anchored to the real data distribution by SIGReg (see module docstring
    for why a bottleneck floor with no such anchor is not sufficient on its
    own). Zero exactly when every eigenvalue of both G1 G1^T and G2^T G2
    already clears its floor."""
    loss = torch.tensor(0.0)
    if gamma1 > 0:
        G1 = predictor.G1.weight  # (bottleneck_dim, latent_dim*past_window)
        eig1 = torch.linalg.eigvalsh(G1 @ G1.T)
        loss = loss + torch.relu(gamma1 ** 2 - eig1).sum()
    if gamma2 > 0:
        G2 = predictor.g2_embedded(encoder)  # (future_window*latent_dim, bottleneck_dim)
        eig2 = torch.linalg.eigvalsh(G2.T @ G2)
        loss = loss + torch.relu(gamma2 ** 2 - eig2).sum()
    return loss



@dataclass
class FilteringTrainConfig:
    latent_dim: int = 1  # per-timestep encoder output -- deliberately < system.n, the collapse-prone dimension
    past_window: int = 4
    future_window: int = 4
    bottleneck_dim: int = 2  # G_1's output / state-estimate dim -- must be >= system.n for RankReg's guarantee to apply
    lr: float = 1e-2
    epochs: int = 400
    batch_size: int = 256
    lambda_pred: float = 1.0
    lambda_rank: float = 0.0
    gamma1: float = 1.0  # floor on G1 G1^T
    gamma2: float = 1.0  # floor on M^T D_H(E)^T D_H(E) M (= G2^T G2)
    lambda_sigreg: float = 0.0  # anchors E to the data (replaces the notes' hard E Sigma_y E^T = I)
    sigreg_directions: int = 16
    weight_decay: float = 0.0
    seed: int = 0


def train_filtering(
    system,
    obs_model,
    train_batch: EpisodeBatch,
    cfg: FilteringTrainConfig,
    verbose: bool = True,
    log_every: int = 10,
):
    """Naive joint Adam SGD over (encoder, HankelPredictor) -- no closed-form
    solves, nothing frozen, consistent with `train.train_jepa_naive`'s
    convention elsewhere in this codebase. `system` is accepted (matching
    every other training entry point's signature in this codebase) but
    unused: every loss here is computable from `obs_model`/`train_batch`
    (i.e. from data) and the learnable matrices alone, deliberately never
    touching the ground-truth system matrices."""
    del system
    torch.manual_seed(cfg.seed)
    encoder = LinearEncoder(obs_model.p, cfg.latent_dim, bias=False)
    predictor = HankelPredictor(cfg.latent_dim, obs_model.p, cfg.past_window, cfg.future_window, cfg.bottleneck_dim)

    train_ds = HankelWindowDataset(train_batch, cfg.past_window, cfg.future_window)
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    params = list(encoder.parameters()) + list(predictor.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = []
    for epoch in range(cfg.epochs):
        totals = {"pred": 0.0, "rank": 0.0, "sigreg": 0.0, "total": 0.0}
        n_batches = 0
        for y_past, y_future in loader:
            opt.zero_grad()
            z_past = encode_window(encoder, y_past)
            loss_pred = hankel_prediction_loss(predictor, encoder, z_past, y_future)
            loss = cfg.lambda_pred * loss_pred

            loss_rank = torch.tensor(0.0)
            if cfg.lambda_rank > 0:
                loss_rank = rank_reg_loss(predictor, encoder, cfg.gamma1, cfg.gamma2)
                loss = loss + cfg.lambda_rank * loss_rank

            loss_sigreg = torch.tensor(0.0)
            if cfg.lambda_sigreg > 0:
                z_future = encode_window(encoder, y_future)
                z_all = torch.cat([z_past.reshape(-1, cfg.latent_dim), z_future.reshape(-1, cfg.latent_dim)], dim=0)
                loss_sigreg = sigreg_loss(z_all, n_directions=cfg.sigreg_directions)
                loss = loss + cfg.lambda_sigreg * loss_sigreg

            loss.backward()
            opt.step()

            totals["pred"] += (cfg.lambda_pred * loss_pred).item()
            totals["rank"] += (cfg.lambda_rank * loss_rank).item()
            totals["sigreg"] += (cfg.lambda_sigreg * loss_sigreg).item()
            totals["total"] += loss.item()
            n_batches += 1
        for k in totals:
            totals[k] /= n_batches
        history.append(totals)

        if verbose and (epoch % log_every == 0 or epoch == cfg.epochs - 1):
            t = history[-1]
            print(f"[epoch {epoch:4d}] total={t['total']:.4f} pred={t['pred']:.4f} rank={t['rank']:.4f} sigreg={t['sigreg']:.4f}")

    return encoder, predictor, history


def observable_mode_retention(
    system: LTISystem, encoder, past_window: int, train_batch: EpisodeBatch, val_batch: EpisodeBatch, alpha: float = 1e-2,
) -> float:
    """Fit the stacked EMBEDDED past window [z_{t-L};...;z_{t-1}] -> the true
    at-risk mode's modal coordinate at the window's last timestep, with
    ridge regression on `train_batch`; report held-out R^2 on `val_batch`.

    This is the WINDOW-based analogue of `diagnostics.unstable_mode_retention`
    (which regresses from a SINGLE z_t alone) -- the right tool for this
    filtering construction, where `latent_dim` is deliberately < system.n, so
    a single z_t can correlate poorly with EITHER mode even when a short
    window of such z's jointly determines the full state exactly (classical
    observability from a scalar output, recovered over several timesteps).
    A low single-step R^2 does not by itself mean a mode's information is
    gone from the representation; this does directly test that.

    Reuses `system.unstable_mode_index()`/`unstable_modal_coordinate` as the
    "at-risk" mode -- in this module's intended construction that's simply
    whichever of the two (both-stable) eigenvalues was given the
    scale-mixture-contaminated initial condition via
    `data.make_modal_contaminated_x0_sampler`."""
    def _windowed_features_and_targets(batch: EpisodeBatch):
        n_ep, Tp1, p = batch.y.shape
        y_windows, targets = [], []
        for e in range(n_ep):
            for t in range(past_window, Tp1):
                y_windows.append(batch.y[e, t - past_window : t])
                targets.append(batch.x[e, t - 1])
        y_windows = np.stack(y_windows)  # (N, L, p)
        targets = np.stack(targets)  # (N, n)
        N = y_windows.shape[0]
        with torch.no_grad():
            z_windows = encoder(torch.tensor(y_windows.reshape(-1, p), dtype=torch.float32)).numpy()
        feats = z_windows.reshape(N, -1)
        return feats, targets

    Z_tr, x_tr = _windowed_features_and_targets(train_batch)
    Z_val, x_val = _windowed_features_and_targets(val_batch)
    xi_tr = system.unstable_modal_coordinate(x_tr)
    xi_val = system.unstable_modal_coordinate(x_val)
    beta = ridge_fit(Z_tr, xi_tr, alpha=alpha)
    xi_pred = ridge_predict(Z_val, beta)
    return r2_score(xi_val, xi_pred)
