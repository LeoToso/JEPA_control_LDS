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
N_SEEDS = 2

# PASSIVE system (no actions) -- the original setting, before the later
# action-conditioned detour. Fully observable (rank(C)=2): both modes are
# genuinely present and recoverable from y.
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
    total = past_window + future_window
    n_ep, Tp1, _p = batch.y.shape
    past_ys, future_ys, x_starts, x_pasts = [], [], [], []
    for e in range(n_ep):
        for t in range(0, Tp1 - total + 1):
            past_ys.append(batch.y[e, t : t + past_window])
            future_ys.append(batch.y[e, t + past_window : t + total])
            x_starts.append(batch.x[e, t])
            x_pasts.append(batch.x[e, t : t + past_window])  # true state over the SAME past window
    y_past = torch.tensor(np.stack(past_ys), dtype=torch.float32)
    y_future = torch.tensor(np.stack(future_ys), dtype=torch.float32)
    x_start = np.stack(x_starts)
    x_past = torch.tensor(np.stack(x_pasts), dtype=torch.float32)  # (N, L, n)
    return y_past, y_future, x_start, x_past


Y_PAST_FULL, Y_FUTURE_FULL, X_START_TRAIN, X_PAST_FULL = build_windows_with_state(TRAIN_BATCH, L, H)
Y_PAST_TEST, Y_FUTURE_TEST, X_START_TEST, X_PAST_TEST = build_windows_with_state(TEST_BATCH, L, H)


def to_modal_coords(x_raw):
    w = (Vinv @ x_raw.T).T.real
    return np.stack([w[:, idx_u], w[:, idx_s]], axis=1)


W_TRAIN = to_modal_coords(X_START_TRAIN)
W_TEST = to_modal_coords(X_START_TEST)


class LatentPredictor(nn.Module):
    """G1 compresses the past embedded window into a bottleneck state; G2
    maps that state DIRECTLY to the predicted future EMBEDDED window (no
    decode to raw observations) -- per-step blocks G2_h, h=1..H."""

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


def bottleneck_states(encoder, G1_matrix, y_past):
    with torch.no_grad():
        z_past = encode_window(encoder, y_past)
        b = y_past.shape[0]
        z_flat = z_past.reshape(b, -1).numpy()
    return z_flat @ G1_matrix.T


def eval_r2(encoder, G1_matrix):
    """Fits a linear decoder (bottleneck state -> true modal state) by least
    squares on TRAIN windows, reports R^2 per mode on HELD-OUT test windows."""
    state_train = bottleneck_states(encoder, G1_matrix, Y_PAST_FULL)
    state_test = bottleneck_states(encoder, G1_matrix, Y_PAST_TEST)
    S_hat, *_ = np.linalg.lstsq(state_train, W_TRAIN, rcond=None)
    w_pred_test = state_test @ S_hat
    ss_res = ((W_TEST - w_pred_test) ** 2).sum(axis=0)
    ss_tot = ((W_TEST - W_TEST.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / ss_tot
    return float(r2[0]), float(r2[1])  # r2_at_risk, r2_safe


def train_standard_jepa(seed, epochs=EPOCHS, log_every=500, log_prefix=""):
    """L = L_pred + 25*L_SIGReg, random init."""
    torch.manual_seed(seed)
    encoder = LinearEncoder(OBS_MODEL.p, LATENT_DIM, bias=False)
    predictor = LatentPredictor(LATENT_DIM, L, H, BOTTLENECK_DIM)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=1e-2)
    for epoch in range(epochs):
        opt.zero_grad()
        z_past = encode_window(encoder, Y_PAST_FULL)
        z_future_true = encode_window(encoder, Y_FUTURE_FULL)
        z_hat_future, _ = predictor(z_past)
        loss_pred = ((z_hat_future - z_future_true) ** 2).sum(dim=-1).mean()
        z_all = torch.cat([z_past.reshape(-1, LATENT_DIM), z_future_true.reshape(-1, LATENT_DIM)], dim=0)
        loss_sig = sigreg_loss_pinned(z_all, n_directions=16)
        loss = loss_pred + 25.0 * loss_sig
        loss.backward()
        opt.step()
        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            r2_u, r2_s = eval_r2(encoder, predictor.g1_matrix())
            print(f"{log_prefix}epoch={epoch}  loss_pred={loss_pred.item():.5f}  loss_sig={loss_sig.item():.5f}  "
                  f"R2(at-risk)={r2_u:.4f}  R2(safe)={r2_s:.4f}  mean_R2={0.5*(r2_u+r2_s):.4f}", flush=True)
    return encoder, predictor.g1_matrix()


def train_pure_autoencoder(seed, bottleneck_dim=BOTTLENECK_DIM, epochs=EPOCHS, log_every=500, log_prefix=""):
    """Reconstruction-only baseline: encoder + G1 + decoder, MSE on the past
    window. No prediction loss, no SIGReg -- a pure rank-2 linear
    autoencoder, which by Eckart-Young converges to the exact PCA solution."""
    torch.manual_seed(seed)
    encoder = LinearEncoder(OBS_MODEL.p, LATENT_DIM, bias=False)
    G1 = nn.Linear(LATENT_DIM * L, bottleneck_dim, bias=False)
    decoder = nn.Linear(bottleneck_dim, OBS_MODEL.p * L, bias=False)
    params = list(encoder.parameters()) + list(G1.parameters()) + list(decoder.parameters())
    opt = torch.optim.Adam(params, lr=1e-2)
    for epoch in range(epochs):
        opt.zero_grad()
        z_past = encode_window(encoder, Y_PAST_FULL)
        b = Y_PAST_FULL.shape[0]
        state = G1(z_past.reshape(b, -1))
        y_hat = decoder(state).reshape(b, L, OBS_MODEL.p)
        loss = ((y_hat - Y_PAST_FULL) ** 2).sum(dim=-1).mean()
        loss.backward()
        opt.step()
        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            r2_u, r2_s = eval_r2(encoder, G1.weight.detach().cpu().numpy())
            print(f"{log_prefix}epoch={epoch}  loss={loss.item():.5f}  "
                  f"R2(at-risk)={r2_u:.4f}  R2(safe)={r2_s:.4f}  mean_R2={0.5*(r2_u+r2_s):.4f}", flush=True)
    return encoder, G1.weight.detach().cpu().numpy()


def train_state_autoencoder(seed, bottleneck_dim=BOTTLENECK_DIM, epochs=EPOCHS, log_every=500, log_prefix=""):
    """Same encoder+G1 pipeline, but the decoder reconstructs the TRUE STATE
    window (X_PAST_FULL, clean/low-noise proprioceptive info) instead of the
    raw noisy observation window Y_PAST_FULL -- an oracle-ish comparison
    point: how much does reconstructing the cleaner target change things,
    given the identical encoder->bottleneck architecture?"""
    torch.manual_seed(seed)
    encoder = LinearEncoder(OBS_MODEL.p, LATENT_DIM, bias=False)
    G1 = nn.Linear(LATENT_DIM * L, bottleneck_dim, bias=False)
    n = SYSTEM.A.shape[0]
    decoder = nn.Linear(bottleneck_dim, n * L, bias=False)
    params = list(encoder.parameters()) + list(G1.parameters()) + list(decoder.parameters())
    opt = torch.optim.Adam(params, lr=1e-2)
    for epoch in range(epochs):
        opt.zero_grad()
        z_past = encode_window(encoder, Y_PAST_FULL)
        b = Y_PAST_FULL.shape[0]
        state = G1(z_past.reshape(b, -1))
        x_hat = decoder(state).reshape(b, L, n)
        loss = ((x_hat - X_PAST_FULL) ** 2).sum(dim=-1).mean()
        loss.backward()
        opt.step()
        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            r2_u, r2_s = eval_r2(encoder, G1.weight.detach().cpu().numpy())
            print(f"{log_prefix}epoch={epoch}  loss={loss.item():.5f}  "
                  f"R2(at-risk)={r2_u:.4f}  R2(safe)={r2_s:.4f}  mean_R2={0.5*(r2_u+r2_s):.4f}", flush=True)
    return encoder, G1.weight.detach().cpu().numpy()


if __name__ == "__main__":
    print(f"Standard JEPA vs Pure Autoencoder (observation) vs State Autoencoder, passive system, random init, "
          f"full-batch, {EPOCHS} epochs, {N_SEEDS} seeds, sigma_stable={SIGMA_STABLE}", flush=True)

    for label, train_fn in [
        ("Standard JEPA", train_standard_jepa),
        ("Pure Autoencoder (obs)", train_pure_autoencoder),
        ("State Autoencoder", train_state_autoencoder),
    ]:
        ru_vals, rs_vals = [], []
        for seed in range(N_SEEDS):
            encoder, G1_matrix = train_fn(seed, log_prefix=f"  [{label} seed={seed}] ")
            r2_u, r2_s = eval_r2(encoder, G1_matrix)
            ru_vals.append(r2_u); rs_vals.append(r2_s)
            print(f"  {label}  seed={seed}  R2(at-risk)={r2_u:.4f}  R2(safe)={r2_s:.4f}  "
                  f"mean_R2={0.5*(r2_u+r2_s):.4f}", flush=True)
        ru_mean, rs_mean = np.mean(ru_vals), np.mean(rs_vals)
        print(f"{label}: MEAN R2(at-risk)={ru_mean:.4f}  R2(safe)={rs_mean:.4f}  "
              f"mean_R2={0.5*(ru_mean+rs_mean):.4f}", flush=True)
    print("DONE", flush=True)
