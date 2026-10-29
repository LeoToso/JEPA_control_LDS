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

IMPORTANT: RankReg must constrain the *composed* map from the TRUE STATE,
through the encoder E, through the past window, to the bottleneck state
estimate -- not G_1's own matrix rank in isolation. G_1 only ever sees
z's that E already produced; if E itself has zero sensitivity to a mode
(its row is exactly orthogonal to that mode's influence on y), no amount of
downstream processing by a full-rank G_1 can recover information that was
never in z to begin with. `rank_reg_loss` below penalizes the smallest
eigenvalue of Q^T Q, where Q = G_1 @ D_L(E) @ Obs_L and Obs_L = [C; CA;
...; C A^{L-1}] is the TRUE system's own (fixed) finite-window observability
matrix -- i.e. it penalizes any TRUE state direction that would map to (near)
zero change in the bottleneck state estimate, directly closing the gap a
G_1-only penalty leaves open.

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


def past_window_observability_matrix(system: LTISystem, obs_model, past_window: int) -> np.ndarray:
    """Obs_L = [C; C A; ...; C A^{L-1}] (shape (L*p, n)), using the TRUE
    system's (A, C) -- C padded with zero columns for the distractor
    channels, which carry no x-dependent signal. This is a FIXED matrix
    (depends only on the ground-truth system, never on anything learned):
    it maps the state at the START of a length-`past_window` window to the
    stacked RAW observations across that window, `[y_{t-L};...;y_{t-1}] =
    Obs_L @ x_{t-L}` (noiseless)."""
    n = system.n
    C_full = np.zeros((obs_model.p, n))
    C_full[: obs_model.signal_dim, :] = obs_model.C
    mats = [C_full]
    Ak = np.eye(n)
    for _ in range(1, past_window):
        Ak = system.A @ Ak
        mats.append(C_full @ Ak)
    return np.vstack(mats)


def embed_observability_matrix(encoder, obs_L: torch.Tensor, past_window: int, obs_dim: int) -> torch.Tensor:
    """Applies the CURRENT (learnable, differentiable) encoder independently
    to each (p, n) block of `obs_L` (shape (L*p, n)) -- i.e. computes
    D_L(E) @ Obs_L without ever materializing the block-diagonal D_L(E) --
    returning the (L*d, n) map from the true state at the window start to
    the stacked EMBEDDED past window."""
    blocks = obs_L.reshape(past_window, obs_dim, -1)  # (L, p, n)
    embedded = torch.einsum("dp,Lpn->Ldn", encoder.linear.weight, blocks)  # (L, d, n)
    return embedded.reshape(past_window * encoder.linear.weight.shape[0], -1)  # (L*d, n)


def rank_reg_loss(predictor: HankelPredictor, encoder, obs_L: torch.Tensor, gamma: float) -> torch.Tensor:
    """Soft version of a rank-floor constraint on the COMPOSED map from the
    TRUE STATE to the bottleneck state estimate, Q = G_1 @ D_L(E) @ Obs_L
    (shape (bottleneck_dim, n)): a hinge penalty on each eigenvalue of
    Q^T Q that falls below gamma^2. Zero exactly when every true-state
    direction already produces a state-estimate change of norm >= gamma;
    otherwise it directly fights an encoder/predictor pair that would
    cheapen the prediction loss by making SOME true-state direction
    (i.e. an observable mode) invisible to the state estimate -- the one
    failure mode SIGReg (a pure marginal-distribution penalty on z, with no
    reference to the true state at all) has no mechanism to prevent.

    Constraining G_1's own rank alone is NOT sufficient: G_1 only ever sees
    z's the encoder already produced, so if E itself has zero sensitivity to
    a mode, no amount of downstream full-rank G_1 processing can recover
    it -- hence this penalizes the FULL composed map through E, not G_1 in
    isolation. Requires `predictor.bottleneck_dim >= system.n` (`obs_L`'s
    column count) for Q to possibly have full rank at all."""
    if gamma <= 0:
        return torch.tensor(0.0)
    embedded_obs_L = embed_observability_matrix(encoder, obs_L, predictor.past_window, predictor.obs_dim)
    Q = predictor.G1.weight @ embedded_obs_L  # (bottleneck_dim, n)
    gram = Q.T @ Q  # (n, n), symmetric PSD
    eigvals = torch.linalg.eigvalsh(gram)
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
    convention elsewhere in this codebase."""
    torch.manual_seed(cfg.seed)
    encoder = LinearEncoder(obs_model.p, cfg.latent_dim, bias=False)
    predictor = HankelPredictor(cfg.latent_dim, obs_model.p, cfg.past_window, cfg.future_window, cfg.bottleneck_dim)

    train_ds = HankelWindowDataset(train_batch, cfg.past_window, cfg.future_window)
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    obs_L = torch.tensor(past_window_observability_matrix(system, obs_model, cfg.past_window), dtype=torch.float32)

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
                loss_rank = rank_reg_loss(predictor, encoder, obs_L, cfg.gamma)
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
