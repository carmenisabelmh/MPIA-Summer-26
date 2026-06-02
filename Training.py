import numpy as np 
import torch

from SpecML import SpecML
from Tokeniser import patch_size, step_size, P, V, X
import time
import optuna
import torch.nn as nn


start_time = time.time()
#------------------------------------------TRAINING PARAMETERS----------------------------------------------------#

N_STEPS_PER_RESTART = 100000  # gradient steps
BATCH_SIZE = 64  # spectra per batch
LR = 5e-4  # AdamW learning rate
WEIGHT_DECAY = 0.01  # AdamW weight decay
BETAS = (0.9, 0.95)  # AdamW β₁, β₂
GRAD_CLIP = 1.0  # gradient clip max norm
SCHED_ETA_MIN = 1e-6  # minimum LR after annealing
NUM_RESTARTS = 2  # number of annealing cycles
N_STEPS = N_STEPS_PER_RESTART * NUM_RESTARTS

#-------------------------------------------------MASKING-----------------------------------------------------------#

CHUNK_WIDTH = int(np.floor(2.5 * patch_size / step_size)) # chunk width in tokens
MASK_RATIO = 0.75  # fraction of max_chunks to mask per spectrum

def apply_chunk_mask_batch(
    y_b: torch.Tensor,
    v_b: torch.Tensor,
    mask_ratio: float = MASK_RATIO,
    chunk_width: int = CHUNK_WIDTH,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask one batch entirely on device"""
    B, T, _ = y_b.shape
    device = y_b.device
    M = torch.zeros(B, T, dtype=torch.bool, device=device)

    valid_lens = v_b.sum(dim=1).int()
    n_chunks = valid_lens // chunk_width
    k = ((((n_chunks + 1) // 2) * mask_ratio).int()).clamp(min=1)
    section = valid_lens // k.clamp(min=1)
    usable = (valid_lens >= chunk_width) & (section >= chunk_width)

    offsets = torch.arange(chunk_width, device=device)
    k_max = int(k[usable].max()) if usable.any() else 0

    for j in range(k_max):
        active = torch.where(usable & (k > j))[0]
        sec = section[active]
        u = torch.rand(len(active), device=device)
        starts = (j * sec + u * (sec - chunk_width + 1)).int()
        M[active[:, None], starts[:, None] + offsets] = True

    x_b = y_b.clone()
    x_b[M] = 0.0
    return x_b, M


#-------------------------------------------------LOSS-----------------------------------------------------------#

def mse_loss(Y, Yhat, M):
    err = ((Y - Yhat) ** 2).sum(dim=-1)  # [B, T]  squared L2 norm over patch dim
    return err[M].mean()  # mean over masked positions only


#-------------------------------------------------TRAINING-----------------------------------------------------------#
class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, T_0, eta_min=0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.T_0 = T_0
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * self.last_epoch / max(1, self.warmup_steps)
                    for base_lr in self.base_lrs]
        progress = (self.last_epoch - self.warmup_steps) / max(1, self.T_0 - self.warmup_steps)
        cosine = 0.5 * (1 + np.cos(np.pi * (progress % 1.0)))
        return [self.eta_min + (base_lr - self.eta_min) * cosine
                for base_lr in self.base_lrs]

if __name__ == '__main__':
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    if torch.backends.mps.is_available():
        device = 'mps'

    # Entire dataset fits in memory — pin Y and V on device permanently.
    # P is (T, D_EMB) — shared across all spectra; kept on device, not batched.
    N = X.shape[0]
    Y_dev = torch.from_numpy(X).float().to(device)
    V_dev = torch.from_numpy(V).bool().to(device)
    P_dev = torch.from_numpy(P).float().to(device)

    model = SpecML(patch_dim=patch_size + 2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=BETAS)
    WARMUP_STEPS = 2000
    scheduler = WarmupCosineScheduler(opt, warmup_steps=WARMUP_STEPS, T_0=N_STEPS_PER_RESTART, eta_min=SCHED_ETA_MIN)
    # Set random seed
    rng = torch.Generator(device=device)
    rng.manual_seed(0)

    loss_curve = []
    step = 0
    while step < N_STEPS:
        perm = torch.randperm(N, device=device, generator=rng)
        for start in range(0, N, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            y_b, v_b = Y_dev[idx], V_dev[idx]
            x_b, m_b = apply_chunk_mask_batch(y_b, v_b)
            Yhat = model(x_b, v_b, P_dev)
            loss = mse_loss(y_b, Yhat, m_b)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            scheduler.step()
            loss_val = loss.item()
            loss_curve.append(loss_val)
            print(f'step {step:4d}  loss {loss_val:.4f}')
            step += 1
            if step >= N_STEPS:
                break

    torch.save(model.state_dict(), 'SpecML.pt')
    np.save('loss_curve.npy', np.array(loss_curve))

end_time = time.time() - start_time
print(end_time)
