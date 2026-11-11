import numpy as np
import torch
import torch.nn as nn

from jepa_lds.systems import make_double_mode_system
from jepa_lds.data import make_observation_model, make_modal_contaminated_x0_sampler, generate_dataset, HankelWindowDataset
from jepa_lds.models import LinearEncoder
from jepa_lds.losses import encode_window, _char_fn_gaussian_stat
from jepa_lds.control import encoder_state_to_latent_map

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
w, V, Vinv = SYSTEM.modal_decomposition()
idx_u = SYSTEM.unstable_mode_index()
idx_s = 1 - idx_u

ds = HankelWindowDataset(TRAIN_BATCH, L, H)
Y_PAST_FULL = torch.stack([ds[i][0] for i in range(len(ds))])
Y_FUTURE_FULL = torch.stack([ds[i][1] for i in range(len(ds))])


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


def exact_M_eff(system, obs_model, encoder, G1_matrix, L):
    M_step = encoder_state_to_latent_map(system, obs_model, encoder)
    A = system.A
    O_L_eff = np.vstack([M_step @ np.linalg.matrix_power(A, k) for k in range(L)])
    return G1_matrix @ O_L_eff


def eval_cols(encoder, G1_matrix):
    M_eff_modal = exact_M_eff(SYSTEM, OBS_MODEL, encoder, G1_matrix, L) @ V
    return np.linalg.norm(M_eff_modal[:, idx_u]), np.linalg.norm(M_eff_modal[:, idx_s])


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
            with torch.no_grad():
                cu, cs = eval_cols(encoder, predictor.g1_matrix())
            print(f"{log_prefix}epoch={epoch}  loss_pred={loss_pred.item():.5f}  loss_sig={loss_sig.item():.5f}  "
                  f"col(at-risk)={cu:.4f}  col(safe)={cs:.4f}  ratio={cu/cs:.2f}", flush=True)
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
            with torch.no_grad():
                cu, cs = eval_cols(encoder, predictor.g1_matrix())
            print(f"{log_prefix}epoch={epoch}  loss_pred={loss_pred.item():.5f}  loss_sig={loss_sig.item():.5f}  "
                  f"loss_rank={loss_rank.item():.5f}  loss_ws={loss_ws.item():.5f}  "
                  f"col(at-risk)={cu:.4f}  col(safe)={cs:.4f}  ratio={cu/cs:.2f}", flush=True)
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
        u_vals, s_vals, loss_vals = [], [], []
        for seed in range(5):
            encoder, G1_matrix, total_loss = train_fn(seed, log_prefix=f"  [{label} seed={seed}] ")
            cu, cs = eval_cols(encoder, G1_matrix)
            u_vals.append(cu); s_vals.append(cs); loss_vals.append(total_loss)
            print(f"  {label}  seed={seed}  col(at-risk)={cu:.4f}  col(safe)={cs:.4f}  ratio={cu/cs:.2f}  final_loss={total_loss:.4f}", flush=True)
        u_mean, s_mean = np.mean(u_vals), np.mean(s_vals)
        print(f"{label}: MEAN col(at-risk)={u_mean:.4f}  col(safe)={s_mean:.4f}  ratio={u_mean/s_mean:.2f}  mean_final_loss={np.mean(loss_vals):.4f}", flush=True)
    print("DONE", flush=True)
