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
ACTION_STD = 1.0

# KEY CHANGE vs. the passive-only setup used throughout: real, persistently-
# exciting actions (prbs), matching the paper's EP-IDM premise -- it needs
# actions to exist and excite the reachable subspace, which a passive system
# (action_std=0, mixture={"passive":1.0}) never provides.
TRAIN_BATCH = generate_dataset(
    SYSTEM, OBS_MODEL, 300, EP_LEN, seed=0, x0_sampler=X0_SAMPLER,
    action_std=ACTION_STD, process_noise_std=PROCESS_NOISE_STD, state_clip=250.0, mixture={"prbs": 1.0},
)
TEST_BATCH = generate_dataset(
    SYSTEM, OBS_MODEL, 100, EP_LEN, seed=12345, x0_sampler=X0_SAMPLER,
    action_std=ACTION_STD, process_noise_std=PROCESS_NOISE_STD, state_clip=250.0, mixture={"prbs": 1.0},
)

_, V, Vinv = SYSTEM.modal_decomposition()
idx_u = SYSTEM.unstable_mode_index()
idx_s = 1 - idx_u
M_ACTIONS = SYSTEM.B.shape[1]  # action dimension (m)


def build_windows_with_actions(batch, past_window, future_window):
    """Past/future observation windows, the true state at window-start (for
    R^2 eval), and the H actions spanning the FUTURE window (a_{t+L},...,
    a_{t+L+H-1}) -- the span EP-IDM's endpoint decoder needs to predict."""
    total = past_window + future_window
    n_ep, Tp1, _p = batch.y.shape
    past_ys, future_ys, x_starts, fut_actions = [], [], [], []
    for e in range(n_ep):
        # actions array has length T=Tp1-1 (one less than y/x), and we need
        # t+total <= T, i.e. t <= Tp1-1-total, i.e. t in range(0, Tp1-total).
        for t in range(0, Tp1 - total):
            past_ys.append(batch.y[e, t : t + past_window])
            future_ys.append(batch.y[e, t + past_window : t + total])
            x_starts.append(batch.x[e, t])
            fut_actions.append(batch.a[e, t + past_window : t + total])
    y_past = torch.tensor(np.stack(past_ys), dtype=torch.float32)
    y_future = torch.tensor(np.stack(future_ys), dtype=torch.float32)
    x_start = np.stack(x_starts)
    actions_future = torch.tensor(np.stack(fut_actions), dtype=torch.float32)  # (N, H, m)
    return y_past, y_future, x_start, actions_future


Y_PAST_FULL, Y_FUTURE_FULL, X_START_TRAIN, ACTIONS_FUTURE_TRAIN = build_windows_with_actions(TRAIN_BATCH, L, H)
Y_PAST_TEST, Y_FUTURE_TEST, X_START_TEST, ACTIONS_FUTURE_TEST = build_windows_with_actions(TEST_BATCH, L, H)


def to_modal_coords(x_raw):
    w = (Vinv @ x_raw.T).T.real
    return np.stack([w[:, idx_u], w[:, idx_s]], axis=1)


W_TRAIN = to_modal_coords(X_START_TRAIN)
W_TEST = to_modal_coords(X_START_TEST)

C_PINV = np.linalg.pinv(OBS_MODEL.C)
_e_s_unit = np.zeros(2); _e_s_unit[idx_s] = 1.0
_row_good_raw = _e_s_unit @ Vinv.real @ C_PINV
_z_good_raw = np.concatenate([
    Y_PAST_FULL.numpy().reshape(-1, OBS_MODEL.p) @ _row_good_raw,
    Y_FUTURE_FULL.numpy().reshape(-1, OBS_MODEL.p) @ _row_good_raw,
])
SIGMA_SAFE = _z_good_raw.std()

_Q, _ = np.linalg.qr(OBS_MODEL.C)
_rng0 = np.random.default_rng(0)
_raw = _rng0.standard_normal(OBS_MODEL.p)
_row_dead_raw = _raw - _Q @ (_Q.T @ _raw)
_z_dead_raw = np.concatenate([
    Y_PAST_FULL.numpy().reshape(-1, OBS_MODEL.p) @ _row_dead_raw,
    Y_FUTURE_FULL.numpy().reshape(-1, OBS_MODEL.p) @ _row_dead_raw,
])
_ROW_DEAD = _row_dead_raw / _z_dead_raw.std()


def build_lemma2_init():
    """Same degenerate construction as before (at-risk column zeroed, safe
    column exact, one dead dimension) -- recalibrated against THIS (action-
    driven) dataset's samples. The encoder-side construction doesn't care
    about actions at all (E only ever sees y), so the same formula applies;
    only the calibration constants (SIGMA_SAFE, _ROW_DEAD) need the new data,
    already recomputed above."""
    F_s = 1.0 / SIGMA_SAFE
    e_s = np.zeros(2); e_s[idx_s] = F_s
    e_row_good = e_s @ Vinv.real @ C_PINV
    E_init = np.stack([e_row_good, _ROW_DEAD], axis=0)

    g1_row_good = np.zeros(L * LATENT_DIM)
    g1_row_good[(L - 1) * LATENT_DIM + 0] = 1.0
    g1_row_dead = np.zeros(L * LATENT_DIM)
    G1_init = np.stack([g1_row_good, g1_row_dead], axis=0)
    return E_init, G1_init


class ActionConditionedPredictor(nn.Module):
    """G1 compresses the past embedded window to a bottleneck state, as
    before. G2 now ALSO takes the FUTURE actions (known at training time,
    since we're told what was actually done) when predicting the future
    embedded window -- without this, prediction is ill-posed once actions
    drive the system (the predictor has no way to guess what action will be
    taken). Mirrors the paper's P_theta(z_t, G_theta(a_t)) structure; with
    m=1 we feed raw actions directly, no separate action encoder needed."""

    def __init__(self, latent_dim, past_window, future_window, bottleneck_dim, m_actions):
        super().__init__()
        self.latent_dim = latent_dim
        self.future_window = future_window
        self.bottleneck_dim = bottleneck_dim
        self.m_actions = m_actions
        self.G1 = nn.Linear(latent_dim * past_window, bottleneck_dim, bias=False)
        self.G2 = nn.Linear(bottleneck_dim + future_window * m_actions, latent_dim * future_window, bias=False)

    def forward(self, z_past, actions_future):
        b = z_past.shape[0]
        state = self.G1(z_past.reshape(b, -1))
        g2_in = torch.cat([state, actions_future.reshape(b, -1)], dim=-1)
        z_hat_future = self.G2(g2_in).reshape(b, self.future_window, self.latent_dim)
        return z_hat_future, state

    @torch.no_grad()
    def g1_matrix(self):
        return self.G1.weight.detach().cpu().numpy()


class EPIDMDecoder(nn.Module):
    """D_theta(z_0, z_H) -> predicted action sequence a_{t},...,a_{t+H-1},
    eq. (14)/(15) in the paper. Small MLP (paper uses a 2-hidden-layer MLP
    in their real experiments; linear in their theory)."""

    def __init__(self, latent_dim, future_window, m_actions, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, future_window * m_actions),
        )
        self.future_window = future_window
        self.m_actions = m_actions

    def forward(self, z0, zH):
        b = z0.shape[0]
        out = self.net(torch.cat([z0, zH], dim=-1))
        return out.reshape(b, self.future_window, self.m_actions)


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
    state_train = bottleneck_states(encoder, G1_matrix, Y_PAST_FULL)
    state_test = bottleneck_states(encoder, G1_matrix, Y_PAST_TEST)
    S_hat, *_ = np.linalg.lstsq(state_train, W_TRAIN, rcond=None)
    w_pred_test = state_test @ S_hat
    ss_res = ((W_TEST - w_pred_test) ** 2).sum(axis=0)
    ss_tot = ((W_TEST - W_TEST.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / ss_tot
    return float(r2[0]), float(r2[1])


def apply_init(encoder, predictor):
    E_init, G1_init = build_lemma2_init()
    with torch.no_grad():
        encoder.linear.weight.copy_(torch.tensor(E_init, dtype=torch.float32))
        predictor.G1.weight.copy_(torch.tensor(G1_init, dtype=torch.float32))
        # G2 keeps its default random init (now action-conditioned, no clean
        # analytic "exact decay rollout" since G2 also depends on actions) --
        # this means loss_pred at init will NOT be near its floor the way the
        # passive-system version was; that's expected and fine, we only need
        # a degenerate ENCODER to test whether training can un-collapse it.


def train_sigreg(seed, epochs=EPOCHS, log_every=500, log_prefix=""):
    """L = L_pred + 25*L_SIGReg, action-conditioned predictor, Lemma-2 init."""
    torch.manual_seed(seed)
    encoder = LinearEncoder(OBS_MODEL.p, LATENT_DIM, bias=False)
    predictor = ActionConditionedPredictor(LATENT_DIM, L, H, BOTTLENECK_DIM, M_ACTIONS)
    apply_init(encoder, predictor)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=1e-2)
    last_total = None
    for epoch in range(epochs):
        opt.zero_grad()
        z_past = encode_window(encoder, Y_PAST_FULL)
        z_future_true = encode_window(encoder, Y_FUTURE_FULL)
        z_hat_future, state = predictor(z_past, ACTIONS_FUTURE_TRAIN)
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
        if epoch == epochs - 1:
            last_total = loss.item()
    return encoder, predictor.g1_matrix(), last_total


def train_epidm(seed, lambda_idm=1.0, epochs=EPOCHS, log_every=500, log_prefix=""):
    """L = L_pred + lambda_idm*L_EP-IDM, replacing SIGReg entirely (per the
    paper: "in lieu of L_reg"), action-conditioned predictor, Lemma-2 init."""
    torch.manual_seed(seed)
    encoder = LinearEncoder(OBS_MODEL.p, LATENT_DIM, bias=False)
    predictor = ActionConditionedPredictor(LATENT_DIM, L, H, BOTTLENECK_DIM, M_ACTIONS)
    apply_init(encoder, predictor)
    decoder = EPIDMDecoder(LATENT_DIM, H, M_ACTIONS)
    params = list(encoder.parameters()) + list(predictor.parameters()) + list(decoder.parameters())
    opt = torch.optim.Adam(params, lr=1e-2)
    last_total = None
    for epoch in range(epochs):
        opt.zero_grad()
        z_past = encode_window(encoder, Y_PAST_FULL)
        z_future_true = encode_window(encoder, Y_FUTURE_FULL)
        z_hat_future, state = predictor(z_past, ACTIONS_FUTURE_TRAIN)
        loss_pred = ((z_hat_future - z_future_true) ** 2).sum(dim=-1).mean()
        z0 = z_past[:, -1, :]          # embedding at window-start-of-future (t+L)
        zH = z_future_true[:, -1, :]   # embedding at t+L+H
        a_hat = decoder(z0, zH)
        loss_idm = ((a_hat - ACTIONS_FUTURE_TRAIN) ** 2).mean()
        loss = loss_pred + lambda_idm * loss_idm
        loss.backward()
        opt.step()
        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            r2_u, r2_s = eval_r2(encoder, predictor.g1_matrix())
            print(f"{log_prefix}epoch={epoch}  loss_pred={loss_pred.item():.5f}  loss_idm={loss_idm.item():.5f}  "
                  f"R2(at-risk)={r2_u:.4f}  R2(safe)={r2_s:.4f}  mean_R2={0.5*(r2_u+r2_s):.4f}", flush=True)
        if epoch == epochs - 1:
            last_total = loss.item()
    return encoder, predictor.g1_matrix(), last_total



def train_pred_only(seed, epochs=EPOCHS, log_every=500, log_prefix=""):
    """Action-conditioned predictor, NO regularizer at all (lambda=0) --
    isolates whether action-conditioning alone fixes the degenerate init,
    independent of SIGReg/EP-IDM."""
    torch.manual_seed(seed)
    encoder = LinearEncoder(OBS_MODEL.p, LATENT_DIM, bias=False)
    predictor = ActionConditionedPredictor(LATENT_DIM, L, H, BOTTLENECK_DIM, M_ACTIONS)
    apply_init(encoder, predictor)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=1e-2)
    last_total = None
    for epoch in range(epochs):
        opt.zero_grad()
        z_past = encode_window(encoder, Y_PAST_FULL)
        z_future_true = encode_window(encoder, Y_FUTURE_FULL)
        z_hat_future, state = predictor(z_past, ACTIONS_FUTURE_TRAIN)
        loss = ((z_hat_future - z_future_true) ** 2).sum(dim=-1).mean()
        loss.backward()
        opt.step()
        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            r2_u, r2_s = eval_r2(encoder, predictor.g1_matrix())
            print(f"{log_prefix}epoch={epoch}  loss_pred={loss.item():.5f}  "
                  f"R2(at-risk)={r2_u:.4f}  R2(safe)={r2_s:.4f}  mean_R2={0.5*(r2_u+r2_s):.4f}", flush=True)
        if epoch == epochs - 1:
            last_total = loss.item()
    return encoder, predictor.g1_matrix(), last_total


if __name__ == "__main__":
    print(f"ABLATION: action-conditioned predictor, NO regularizer, Lemma-2 init, {EPOCHS} epochs", flush=True)
    enc, G1, loss = train_pred_only(0, log_prefix="  [PredOnly] ")
    r2_u, r2_s = eval_r2(enc, G1)
    print(f"PredOnly: R2(at-risk)={r2_u:.4f}  R2(safe)={r2_s:.4f}  mean_R2={0.5*(r2_u+r2_s):.4f}  final_loss={loss:.4f}", flush=True)
    print("DONE", flush=True)
