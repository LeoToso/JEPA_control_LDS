import numpy as np
import torch
import torch.nn as nn

from jepa_lds.systems import make_double_mode_system
from jepa_lds.data import make_observation_model, make_modal_contaminated_x0_sampler, generate_dataset
from jepa_lds.models import LinearEncoder
from jepa_lds.losses import encode_window, _char_fn_gaussian_stat

SYSTEM = make_double_mode_system(unstable_eig=1.0, stable_eig=0.6, mix_angle=0.6, name="double_mode_rho1")
OBS_MODEL = make_observation_model(SYSTEM, obs_dim_signal=6, n_distractor=0, measurement_noise_std=0.02, seed=0)
SIGMA_STABLE = 1.0
X0_SAMPLER = make_modal_contaminated_x0_sampler(
    SYSTEM, sigma_stable=SIGMA_STABLE, sigma_unstable_small=0.01, sigma_unstable_large=3.0, contamination_prob=0.1
)
PROCESS_NOISE_STD = 0.05
EP_LEN = 16
L, H = 4, 4
LATENT_DIM = 2
BOTTLENECK_DIM = 2
EPOCHS = 8000

TRAIN_BATCH = generate_dataset(
    SYSTEM, OBS_MODEL, 300, EP_LEN, seed=0, x0_sampler=X0_SAMPLER,
    action_std=0.0, process_noise_std=PROCESS_NOISE_STD, state_clip=250.0, mixture={"passive": 1.0},
)
TEST_BATCH = generate_dataset(
    SYSTEM, OBS_MODEL, 100, EP_LEN, seed=12345, x0_sampler=X0_SAMPLER,
    action_std=0.0, process_noise_std=PROCESS_NOISE_STD, state_clip=250.0, mixture={"passive": 1.0},
)

_, V, Vinv = SYSTEM.modal_decomposition()
idx_u = SYSTEM.unstable_mode_index()
idx_s = 1 - idx_u


def build_windows_with_state(batch, past_window, future_window):
    """Same sliding-window split as HankelWindowDataset, but also returns the
    TRUE simulated state at the START of each past window (the same
    alignment used by the exact observability map: window-start state ->
    forward through A, C, E to produce the embedded past window)."""
    total = past_window + future_window
    n_ep, Tp1, _p = batch.y.shape
    past_ys, future_ys, x_starts = [], [], []
    for e in range(n_ep):
        for t in range(0, Tp1 - total + 1):
            past_ys.append(batch.y[e, t : t + past_window])
            future_ys.append(batch.y[e, t + past_window : t + total])
            x_starts.append(batch.x[e, t])
    y_past = torch.tensor(np.stack(past_ys), dtype=torch.float32)
    y_future = torch.tensor(np.stack(future_ys), dtype=torch.float32)
    x_start = np.stack(x_starts)  # (N, n)
    return y_past, y_future, x_start


Y_PAST_FULL, Y_FUTURE_FULL, X_START_TRAIN = build_windows_with_state(TRAIN_BATCH, L, H)
Y_PAST_TEST, Y_FUTURE_TEST, X_START_TEST = build_windows_with_state(TEST_BATCH, L, H)


def to_modal_coords(x_raw):
    """Project true state (N, n) onto modal coordinates and reorder columns
    to [at-risk, safe] -- matches the eigenbasis V, Vinv from
    SYSTEM.modal_decomposition()."""
    w = (Vinv @ x_raw.T).T.real  # (N, n)
    return np.stack([w[:, idx_u], w[:, idx_s]], axis=1)


W_TRAIN = to_modal_coords(X_START_TRAIN)  # (N_train, 2): col0=at-risk, col1=safe
W_TEST = to_modal_coords(X_START_TEST)    # (N_test, 2)


class LatentPredictor(nn.Module):
    """Two-layer predictor operating ENTIRELY in embedding space: G1 compresses
    the past embedded window into a bottleneck state; G2 maps that state
    DIRECTLY to the predicted future EMBEDDED window (no decode to raw obs,
    no re-encoding) -- per-step blocks G2_h, h=1..H."""

    def __init__(self, latent_dim, past_window, future_window, bottleneck_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.future_window = future_window
        self.bottleneck_dim = bottleneck_dim
        self.G1 = nn.Linear(latent_dim * past_window, bottleneck_dim, bias=False)
        self.G2 = nn.Linear(bottleneck_dim, latent_dim * future_window, bias=False)

    def forward(self, z_past):
        b = z_past.shape[0]
        state = self.G1(z_past.reshape(b, -1))
        z_hat_future = self.G2(state).reshape(b, self.future_window, self.latent_dim)
        return z_hat_future, state

    @torch.no_grad()
    def g1_matrix(self):
        return self.G1.weight.detach().cpu().numpy()

    def g2_blocks(self):
        return self.G2.weight.reshape(self.future_window, self.latent_dim, self.bottleneck_dim)


def latent_prediction_loss(predictor, z_past, z_future_true):
    z_hat_future, _ = predictor(z_past)
    err = z_hat_future - z_future_true
    return (err ** 2).sum(dim=-1).mean()


def sigreg_loss_pinned(z, n_directions=16, n_t=17, t_max=4.0, eps=1e-6, lambda_var=1.0):
    mean = z.mean(dim=0, keepdim=True)
    std = z.std(dim=0, keepdim=True)
    var_penalty = ((std ** 2 - 1.0) ** 2).mean()
    z_std = (z - mean) / (std + eps)
    B, d = z_std.shape
    w_ = torch.randn(d, n_directions)
    w_ = w_ / w_.norm(dim=0, keepdim=True)
    proj = z_std @ w_
    t_grid = torch.linspace(-t_max, t_max, n_t)
    stats = [_char_fn_gaussian_stat(proj[:, k], t_grid) for k in range(n_directions)]
    return torch.stack(stats).mean() + lambda_var * var_penalty


def rank_reg_loss_latent(predictor, gamma1, gamma2):
    """Hinge floor on G1 G1^T and G2^T G2, both now FREE matrices (no
    encoder involved -- G2 is no longer implicitly defined via M)."""
    loss = torch.tensor(0.0)
    if gamma1 > 0:
        G1 = predictor.G1.weight
        eig1 = torch.linalg.eigvalsh(G1 @ G1.T)
        loss = loss + torch.relu(gamma1 ** 2 - eig1).sum()
    if gamma2 > 0:
        G2 = predictor.G2.weight  # (H*d, n_hat)
        eig2 = torch.linalg.eigvalsh(G2.T @ G2)
        loss = loss + torch.relu(gamma2 ** 2 - eig2).sum()
    return loss


def weight_sharing_loss(predictor, M, encoder):
    """Ties each latent per-step block G2_h to E @ M_h, where M_h is a FREE
    per-step raw decoder (state -> raw obs at that future step) used ONLY
    for this penalty -- never in the main prediction loss."""
    G2_blocks = predictor.g2_blocks()  # (H, d, n_hat)
    M_blocks = M.weight.reshape(predictor.future_window, OBS_MODEL.p, predictor.bottleneck_dim)  # (H, p, n_hat)
    E = encoder.linear.weight  # (d, p)
    EM_blocks = torch.einsum("dp,Hpb->Hdb", E, M_blocks)
    return ((G2_blocks - EM_blocks) ** 2).sum()


def bottleneck_states(encoder, G1_matrix, y_past):
    """ŝ_t = G1 · z̄_t for every window in y_past, no grad. Returns (N, n_hat)."""
    with torch.no_grad():
        z_past = encode_window(encoder, y_past)
        b = y_past.shape[0]
        z_flat = z_past.reshape(b, -1).numpy()
    return z_flat @ G1_matrix.T


def eval_r2(encoder, G1_matrix):
    """Fits Ŝ (bottleneck state -> true modal state) by least squares on the
    TRAIN windows, then reports R^2 per mode on the HELD-OUT test windows:

        R^2_i = 1 - sum_test (w_i - (Ŝ ŝ)_i)^2 / sum_test (w_i - mean(w_i))^2

    matching the paper's coordinate-wise R^2 evaluation protocol."""
    state_train = bottleneck_states(encoder, G1_matrix, Y_PAST_FULL)
    state_test = bottleneck_states(encoder, G1_matrix, Y_PAST_TEST)
    S_hat, *_ = np.linalg.lstsq(state_train, W_TRAIN, rcond=None)
    w_pred_test = state_test @ S_hat
    ss_res = ((W_TEST - w_pred_test) ** 2).sum(axis=0)
    ss_tot = ((W_TEST - W_TEST.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / ss_tot
    return float(r2[0]), float(r2[1])  # r2_at_risk, r2_safe


def train_standard_jepa(seed, epochs=EPOCHS, log_every=500, log_prefix=""):
    """L = L_pred + 25*L_SIGReg. No RankReg, no weight-sharing, no M."""
    torch.manual_seed(seed)
    encoder = LinearEncoder(OBS_MODEL.p, LATENT_DIM, bias=False)
    predictor = LatentPredictor(LATENT_DIM, L, H, BOTTLENECK_DIM)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=1e-2)
    last_total = None
    for epoch in range(epochs):
        opt.zero_grad()
        z_past = encode_window(encoder, Y_PAST_FULL)
        z_future_true = encode_window(encoder, Y_FUTURE_FULL)
        loss_pred = latent_prediction_loss(predictor, z_past, z_future_true)
        z_all = torch.cat([z_past.reshape(-1, LATENT_DIM), z_future_true.reshape(-1, LATENT_DIM)], dim=0)
        loss_sig = sigreg_loss_pinned(z_all, n_directions=16)
        loss = loss_pred + 25.0 * loss_sig
        loss.backward()
        opt.step()
        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            r2_u, r2_s = eval_r2(encoder, predictor.g1_matrix())
            print(f"{log_prefix}epoch={epoch}  loss_pred={loss_pred.item():.5f}  loss_sig={loss_sig.item():.5f}  "
                  f"R2(at-risk)={r2_u:.4f}  R2(safe)={r2_s:.4f}  mean_R2={0.5*(r2_u+r2_s):.4f}", flush=True)
        if epoch == epochs - 1:
            last_total = loss.item()
    return encoder, predictor.g1_matrix(), last_total


def train_jepa_rankreg(seed, lambda_ws=1.0, gamma=0.5, lambda_rank=25.0, epochs=EPOCHS, log_every=500, log_prefix=""):
    """L = L_pred + 25*L_SIGReg + lambda_rank*RankReg(G1,G2) + lambda_ws*L_ws."""
    torch.manual_seed(seed)
    encoder = LinearEncoder(OBS_MODEL.p, LATENT_DIM, bias=False)
    predictor = LatentPredictor(LATENT_DIM, L, H, BOTTLENECK_DIM)
    M = nn.Linear(BOTTLENECK_DIM, OBS_MODEL.p * H, bias=False)
    params = list(encoder.parameters()) + list(predictor.parameters()) + list(M.parameters())
    opt = torch.optim.Adam(params, lr=1e-2)
    last_total = None
    for epoch in range(epochs):
        opt.zero_grad()
        z_past = encode_window(encoder, Y_PAST_FULL)
        z_future_true = encode_window(encoder, Y_FUTURE_FULL)
        loss_pred = latent_prediction_loss(predictor, z_past, z_future_true)
        z_all = torch.cat([z_past.reshape(-1, LATENT_DIM), z_future_true.reshape(-1, LATENT_DIM)], dim=0)
        loss_sig = sigreg_loss_pinned(z_all, n_directions=16)
        loss_rank = rank_reg_loss_latent(predictor, gamma, gamma)
        loss_ws = weight_sharing_loss(predictor, M, encoder)
        loss = loss_pred + 25.0 * loss_sig + lambda_rank * loss_rank + lambda_ws * loss_ws
        loss.backward()
        opt.step()
        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            r2_u, r2_s = eval_r2(encoder, predictor.g1_matrix())
            print(f"{log_prefix}epoch={epoch}  loss_pred={loss_pred.item():.5f}  loss_sig={loss_sig.item():.5f}  "
                  f"loss_rank={loss_rank.item():.5f}  loss_ws={loss_ws.item():.5f}  "
                  f"R2(at-risk)={r2_u:.4f}  R2(safe)={r2_s:.4f}  mean_R2={0.5*(r2_u+r2_s):.4f}", flush=True)
        if epoch == epochs - 1:
            last_total = loss.item()
    return encoder, predictor.g1_matrix(), last_total


if __name__ == "__main__":
    configs = [
        ("Standard JEPA", train_standard_jepa),
        ("JEPA with RankReg", train_jepa_rankreg),
    ]

    print(f"Latent-space JEPA comparison, full-batch, {EPOCHS} epochs, sigma_stable={SIGMA_STABLE}", flush=True)
    for label, train_fn in configs:
        ru_vals, rs_vals, loss_vals = [], [], []
        for seed in range(3):
            encoder, G1_matrix, total_loss = train_fn(seed, log_prefix=f"  [{label} seed={seed}] ")
            r2_u, r2_s = eval_r2(encoder, G1_matrix)
            ru_vals.append(r2_u); rs_vals.append(r2_s); loss_vals.append(total_loss)
            print(f"  {label}  seed={seed}  R2(at-risk)={r2_u:.4f}  R2(safe)={r2_s:.4f}  "
                  f"mean_R2={0.5*(r2_u+r2_s):.4f}  final_loss={total_loss:.4f}", flush=True)
        ru_mean, rs_mean = np.mean(ru_vals), np.mean(rs_vals)
        print(f"{label}: MEAN R2(at-risk)={ru_mean:.4f}  R2(safe)={rs_mean:.4f}  "
              f"mean_R2={0.5*(ru_mean+rs_mean):.4f}  mean_final_loss={np.mean(loss_vals):.4f}", flush=True)
    print("DONE", flush=True)
