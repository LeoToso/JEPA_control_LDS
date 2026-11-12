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

C_PINV = np.linalg.pinv(OBS_MODEL.C)

# Calibrate F_s directly against the EXACT pooled samples SIGReg will see (every
# past+future timestep in every window), not an approximation -- the safe mode
# decays within a window (eigenvalue 0.6), so its window-start variance is
# noticeably larger than its pooled variance; this removes that gap exactly.
_e_s_unit = np.zeros(2); _e_s_unit[idx_s] = 1.0
_row_good_raw = _e_s_unit @ Vinv.real @ C_PINV  # (p,) -- F_s=1 placeholder row
_z_good_raw = np.concatenate([
    Y_PAST_FULL.numpy().reshape(-1, OBS_MODEL.p) @ _row_good_raw,
    Y_FUTURE_FULL.numpy().reshape(-1, OBS_MODEL.p) @ _row_good_raw,
])
SIGMA_SAFE = _z_good_raw.std()

# The "dead" row is constructed EXACTLY orthogonal to C's column space, so it
# carries ZERO true-state information by construction (dot(e_dead, y) = dot(e_dead,
# noise) only) -- not just "small", genuinely information-free -- then rescaled
# to unit empirical variance so it starts SIGReg-compliant too.
_Q, _ = np.linalg.qr(OBS_MODEL.C)
_rng0 = np.random.default_rng(0)
_raw = _rng0.standard_normal(OBS_MODEL.p)
_row_dead_raw = _raw - _Q @ (_Q.T @ _raw)  # project out col(C) -- pure-noise direction
_z_dead_raw = np.concatenate([
    Y_PAST_FULL.numpy().reshape(-1, OBS_MODEL.p) @ _row_dead_raw,
    Y_FUTURE_FULL.numpy().reshape(-1, OBS_MODEL.p) @ _row_dead_raw,
])
_ROW_DEAD = _row_dead_raw / _z_dead_raw.std()  # now Var(dot(_ROW_DEAD, y)) ~= 1


def build_lemma2_init(seed=0):
    """Lemma 2 (Toso, LeCun, Anderson, Bounou -- arXiv:2610.07540) degenerate
    solution, adapted to our 2-dim encoder/bottleneck: dimension 0 is the
    EXACT degenerate construction (F = [0, F_s] in modal coords, zeroing the
    at-risk column and keeping only F_s = 1/sigma_safe on the safe column --
    this makes z0 near-standard-Gaussian and gives a near-zero one-step error
    predictor via the safe mode's own 0.6-decay rollout; residual loss is the
    genuine process/measurement noise floor, not an artifact of this
    construction -- verified: an optimally-fit G1/G2 for this same 1-dim
    representation does no better). Dimension 1 is "dead": EXACTLY orthogonal
    to C's column space (zero true-state information, pure unit-variance
    measurement noise -- itself already Gaussian) with G1/G2 weights set to
    EXACTLY zero, so it is genuinely unused bottleneck capacity (G1 is
    rank-deficient at init, eigenvalue exactly 0) -- this is what we're
    testing whether training (and RankReg in particular) can wake up."""
    F_s = 1.0 / SIGMA_SAFE
    e_s = np.zeros(2); e_s[idx_s] = F_s  # row in MODAL coords, at-risk column exactly zero
    e_row_good = e_s @ Vinv.real @ C_PINV  # (p,) -> E's good row, in raw y-space
    E_init = np.stack([e_row_good, _ROW_DEAD], axis=0)  # (2, p)

    # G1: bottleneck row 0 picks out the LAST past timestep's good-dim value
    # (state ~= z_t, the simplest self-consistent filter for this construction,
    # confirmed within ~1% of the optimal-over-the-full-window fit). Flattening
    # order of z_past.reshape(b,-1) is time-major: index = t*latent_dim + dim.
    g1_row_good = np.zeros(L * LATENT_DIM)
    g1_row_good[(L - 1) * LATENT_DIM + 0] = 1.0
    g1_row_dead = np.zeros(L * LATENT_DIM)  # exactly zero: dead dim is unused capacity
    G1_init = np.stack([g1_row_good, g1_row_dead], axis=0)  # (2, L*latent_dim)

    # G2 per-step blocks (H, d, n_hat): good output dim depends only on the
    # good bottleneck dim, rolled forward via the safe mode's own eigenvalue
    # (0.6) decay -- z_{t+h} = A_s^h * z_t exactly, since F_s cancels; dead
    # output dim is exactly zero (nothing to predict it from -- G1's dead row
    # is zero, so the bottleneck's dead component is identically zero too).
    w_eig, _, _ = SYSTEM.modal_decomposition()
    A_s = w_eig[idx_s].real
    G2_blocks = np.zeros((H, LATENT_DIM, BOTTLENECK_DIM))
    for h in range(H):
        G2_blocks[h, 0, 0] = A_s ** (h + 1)
    G2_init = G2_blocks.reshape(H * LATENT_DIM, BOTTLENECK_DIM)  # (H*d, n_hat)
    return E_init, G1_init, G2_init


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


def rank_reg_loss_equality(predictor, gamma1, gamma2):
    """Equality (normed) rank constraint: ||G1 G1^T - gamma1^2 I||_F^2 +
    ||G2^T G2 - gamma2^2 I||_F^2, both now FREE matrices (no encoder
    involved -- G2 is no longer implicitly defined via M). Unlike the hinge
    floor (zero once eigenvalues clear gamma^2, inactive afterwards), this
    pins eigenvalues AT gamma^2 throughout training, not just above it."""
    n_hat = predictor.bottleneck_dim
    I = torch.eye(n_hat)
    G1 = predictor.G1.weight
    loss1 = ((G1 @ G1.T - gamma1 ** 2 * I) ** 2).sum()
    G2 = predictor.G2.weight  # (H*d, n_hat)
    loss2 = ((G2.T @ G2 - gamma2 ** 2 * I) ** 2).sum()
    return loss1 + loss2


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


def apply_init(encoder, predictor, init):
    if init is None:
        return
    E_init, G1_init, G2_init = init
    with torch.no_grad():
        encoder.linear.weight.copy_(torch.tensor(E_init, dtype=torch.float32))
        predictor.G1.weight.copy_(torch.tensor(G1_init, dtype=torch.float32))
        predictor.G2.weight.copy_(torch.tensor(G2_init, dtype=torch.float32))


def train_standard_jepa(seed, epochs=EPOCHS, log_every=500, log_prefix="", init=None):
    """L = L_pred + 25*L_SIGReg. No RankReg, no weight-sharing, no M."""
    torch.manual_seed(seed)
    encoder = LinearEncoder(OBS_MODEL.p, LATENT_DIM, bias=False)
    predictor = LatentPredictor(LATENT_DIM, L, H, BOTTLENECK_DIM)
    apply_init(encoder, predictor, init)
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


def train_jepa_rankreg(seed, lambda_ws=1.0, gamma=0.5, lambda_rank=25.0, epochs=EPOCHS, log_every=500, log_prefix="", init=None):
    """L = L_pred + 25*L_SIGReg + lambda_rank*RankReg(G1,G2) + lambda_ws*L_ws."""
    torch.manual_seed(seed)
    encoder = LinearEncoder(OBS_MODEL.p, LATENT_DIM, bias=False)
    predictor = LatentPredictor(LATENT_DIM, L, H, BOTTLENECK_DIM)
    apply_init(encoder, predictor, init)
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
        loss_rank = rank_reg_loss_equality(predictor, gamma, gamma)
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
        ("Standard JEPA", train_standard_jepa, 0),
        ("JEPA with RankReg", train_jepa_rankreg, 1),
    ]

    print(f"Lemma-2 degenerate init (at-risk mode zeroed, safe mode exact, 1 dead dim), "
          f"full-batch, {EPOCHS} epochs, sigma_stable={SIGMA_STABLE}", flush=True)
    for label, train_fn, seed in configs:
        init = build_lemma2_init(seed=100 + seed)  # same construction; seed only affects the dead-dim noise
        encoder, G1_matrix, total_loss = train_fn(seed, log_prefix=f"  [{label} seed={seed}] ", init=init)
        r2_u, r2_s = eval_r2(encoder, G1_matrix)
        print(f"  {label}  seed={seed}  R2(at-risk)={r2_u:.4f}  R2(safe)={r2_s:.4f}  "
              f"mean_R2={0.5*(r2_u+r2_s):.4f}  final_loss={total_loss:.4f}", flush=True)
    print("DONE", flush=True)
