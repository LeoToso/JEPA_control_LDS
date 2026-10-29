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
(A, C), which JEPA training has no access to. A first version of this
penalty used the TRUE system's own observability matrix directly, which is
not a legitimate regularizer; this version instead uses the EMPIRICAL
covariance of the raw past window, estimated from the training batch --
exactly the same move the SIGReg surrogate in the notes already makes
(L_SIG(E) = ||E Sigma_y E^T - I||^2 uses the DATA covariance Sigma_y, never
A or C). Constraining G_1's own abstract matrix rank in isolation also
turned out to be insufficient on its own: G_1 can be abstractly full-rank
while the REAL data it receives only explores a lower-dimensional slice of
its domain (exactly what happens when E has discarded a mode) -- so the
rank condition has to be checked against the actual data distribution, not
the matrix alone. `rank_reg_loss` below penalizes the smallest eigenvalue
of Q_cov = G_1 @ D_L(E) @ Sigma_{y,window} @ D_L(E)^T @ G_1^T (the
COVARIANCE of the bottleneck state estimate under the real data
distribution) falling below gamma^2 -- if E has really discarded a mode,
the real data simply carries no variance along that direction by the time
it reaches G_1, regardless of G_1's own abstract rank.

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


def hankel_prediction_loss(predictor: HankelPredictor, encoder, z_past: torch.Tensor, y_future: torch.Tensor) -> torch.Tensor:
    """Compares the TRUE and PREDICTED future observation windows in
    EMBEDDED space (both re-embedded with the SAME encoder used to build
    z_past) -- implements the weight-sharing tie G_2 = D_H(E) M."""
    y_hat_future = predictor(z_past)
    z_future_true = encode_window(encoder, y_future)
    z_future_hat = encode_window(encoder, y_hat_future)
    err = z_future_hat - z_future_true
    return (err ** 2).sum(dim=-1).mean()


def past_window_data_covariance(train_batch: EpisodeBatch, past_window: int) -> np.ndarray:
    """Sigma_{y,window} = Cov([y_{t-L};...;y_{t-1}]) (shape (L*p, L*p)),
    estimated EMPIRICALLY from `train_batch` -- no system matrices involved
    at all, exactly the same kind of quantity (the data's own covariance)
    the SIGReg surrogate L_SIG(E) = ||E Sigma_y E^T - I||^2 already uses.
    This is a FIXED matrix computed once before training; it does not
    depend on anything learned."""
    n_ep, Tp1, p = train_batch.y.shape
    windows = []
    for e in range(n_ep):
        for t in range(past_window, Tp1 + 1):
            windows.append(train_batch.y[e, t - past_window : t].reshape(-1))
    W = np.stack(windows)  # (N, L*p)
    W = W - W.mean(axis=0, keepdims=True)
    return (W.T @ W) / W.shape[0]


def embed_data_covariance(encoder, sigma_y_window: torch.Tensor, past_window: int, obs_dim: int) -> torch.Tensor:
    """Applies the CURRENT (learnable, differentiable) encoder to BOTH sides
    of `sigma_y_window` (shape (L*p, L*p)) independently per L-block -- i.e.
    computes D_L(E) @ Sigma_{y,window} @ D_L(E)^T without ever materializing
    the block-diagonal D_L(E) -- returning the (L*d, L*d) covariance of the
    stacked EMBEDDED past window under the real data distribution."""
    d = encoder.linear.weight.shape[0]
    blocks = sigma_y_window.reshape(past_window, obs_dim, past_window, obs_dim)  # (L, p, L, p)
    E = encoder.linear.weight  # (d, p)
    embedded = torch.einsum("ap,LpMq,bq->LaMb", E, blocks, E)  # (L, d, L, d)
    return embedded.reshape(past_window * d, past_window * d)


def rank_reg_loss(predictor: HankelPredictor, encoder, sigma_y_window: torch.Tensor, gamma: float) -> torch.Tensor:
    """Soft version of a rank-floor constraint on the COVARIANCE of the
    bottleneck state estimate under the REAL DATA distribution,
    Q_cov = G_1 @ D_L(E) @ Sigma_{y,window} @ D_L(E)^T @ G_1^T (shape
    (bottleneck_dim, bottleneck_dim)): a hinge penalty on each eigenvalue of
    Q_cov that falls below gamma^2. Zero exactly when every direction of the
    state estimate already has variance >= gamma^2 under the data; otherwise
    it directly fights an encoder/predictor pair that would cheapen the
    prediction loss by making SOME state-estimate direction carry (near)
    zero variance -- i.e. by discarding an observable mode -- the one
    failure mode SIGReg (a pure marginal-distribution penalty on z, with no
    reference to multi-step structure at all) has no mechanism to prevent.

    Computed from ONLY the learnable matrices (G_1, E) and the empirical
    data covariance `sigma_y_window` -- never the true system matrices.
    Constraining G_1's own abstract matrix rank alone is NOT sufficient:
    G_1 can be full rank as a matrix while the REAL data it receives only
    explores a lower-dimensional slice of its domain (exactly what happens
    when E has discarded a mode) -- hence this checks the rank against the
    actual data distribution, via `sigma_y_window`, rather than against G_1
    in isolation. Requires `predictor.bottleneck_dim >= system.n` for
    Q_cov to possibly have full rank at all."""
    if gamma <= 0:
        return torch.tensor(0.0)
    embedded_cov = embed_data_covariance(encoder, sigma_y_window, predictor.past_window, predictor.obs_dim)
    Q_cov = predictor.G1.weight @ embedded_cov @ predictor.G1.weight.T  # (bottleneck_dim, bottleneck_dim)
    eigvals = torch.linalg.eigvalsh(Q_cov)
    return torch.relu(gamma ** 2 - eigvals).sum()



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
    gamma: float = 1.0
    lambda_sigreg: float = 0.0
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

    sigma_y_window = torch.tensor(past_window_data_covariance(train_batch, cfg.past_window), dtype=torch.float32)

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
                loss_rank = rank_reg_loss(predictor, encoder, sigma_y_window, cfg.gamma)
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
