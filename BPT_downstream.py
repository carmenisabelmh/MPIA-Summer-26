# ─────────────────────────────────────────────────────────────────────────────
# BPT downstream prediction — paste as separate cells in your notebook.
# Requires: ipympl  (pip install ipympl)  for the interactive slider.
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════
# CELL 1 — Imports
# ══════════════════════════════════════════════════════════════════
%matplotlib widget

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider
from astropy.table import Table

from SpecML import load_specml
from Tokeniser import tokenize


# ══════════════════════════════════════════════════════════════════
# CELL 2 — Load model  ← only thing you change when swapping checkpoints
# ══════════════════════════════════════════════════════════════════
MODEL_FILE   = 'SpecML 20260602 n20000 lr5e-4 4r 10PS 4OL.pt'
MIN_SNR      = 3.0    # S/N floor for all four BPT emission lines
Z_BIN        = 0.25   # half-width of the redshift window shown per slider tick
Z_SLIDER_MIN = 1.0    # slider left edge (z = 1)

device = ('cuda' if torch.cuda.is_available() else
          'mps'  if torch.backends.mps.is_available() else 'cpu')

model, cfg = load_specml(MODEL_FILE, device=device)
print(cfg)   # shows patch_size, overlap, D_emb etc. pulled from the checkpoint


# ══════════════════════════════════════════════════════════════════
# CELL 3 — Load spectra + catalogue, tokenise using cfg from the model
# ══════════════════════════════════════════════════════════════════
fits_data = Table.read(
    'https://s3.amazonaws.com/msaexp-nirspec/extractions/dja_msaexp_emission_lines_v4.5.prism_spectra.fits',
    cache=True,
)
catalog = Table.read('dja_msaexp_emission_lines_v4.5.csv.gz', format='ascii')
catalog = catalog[catalog['grating'] == 'PRISM']

valid_w        = np.any(fits_data['valid'], 1)
valid_spectrum = np.any(fits_data['valid'], 0)

w      = fits_data['wave'][valid_w]
f_raw  = fits_data['flux'][np.ix_(valid_w, valid_spectrum)].T / (w**2)
dq     = fits_data['valid'][np.ix_(valid_w, valid_spectrum)].T

keep_std  = np.std(f_raw, axis=1) > 0.0
f_raw     = f_raw[keep_std]
dq        = dq[keep_std]
cat_align = catalog[valid_spectrum][keep_std]

mu     = np.mean(f_raw, axis=1, keepdims=True)
sigma  = np.std( f_raw, axis=1, keepdims=True).clip(1e-10)
f_norm = (f_raw - mu) / sigma

# tokenize() uses patch_size/overlap/D_emb straight from the loaded model config
X_all, V_all, P_enc = tokenize(f_norm, dq, w,
                                cfg['patch_size'], cfg['overlap'], cfg['D_emb'])
print(f'Spectra: {len(X_all)},  tokens: {X_all.shape[1]},  patch_dim: {X_all.shape[2]}')


# ══════════════════════════════════════════════════════════════════
# CELL 4 — Batch-encode all spectra
# ══════════════════════════════════════════════════════════════════
P_t  = torch.from_numpy(P_enc).to(device)
embs = []
with torch.no_grad():
    for i in range(0, len(X_all), 512):
        e = model.encode(
            torch.from_numpy(X_all[i:i+512]).float().to(device),
            torch.from_numpy(V_all[i:i+512]).bool().to(device),
            P_t,
        )
        embs.append(e.cpu())
embs = torch.cat(embs)     # (N_all, D_emb)
print(f'Embeddings: {embs.shape}')


# ══════════════════════════════════════════════════════════════════
# CELL 5 — BPT quality cuts and compute true log line ratios
# ══════════════════════════════════════════════════════════════════
def _col(name, fill=np.nan):
    c = cat_align[name]
    arr = np.array(c.filled(fill) if hasattr(c, 'filled') else c, dtype=np.float32)
    arr[~np.isfinite(arr)] = fill
    return arr

def _err(name, fill=np.inf):
    c = cat_align[name]
    arr = np.array(c.filled(fill) if hasattr(c, 'filled') else c, dtype=np.float32)
    arr[~np.isfinite(arr) | (arr <= 0)] = fill
    return arr

hb_f   = _col('line_hb');        hb_e   = _err('line_hb_err')
oiii_f = _col('line_oiii_5007'); oiii_e = _err('line_oiii_5007_err')
ha_f   = _col('line_ha');        ha_e   = _err('line_ha_err')
nii_f  = _col('line_nii_6584');  nii_e  = _err('line_nii_6584_err')
z_all  = _col('z_best', fill=-1.0)

keep_bpt = (
    (hb_f   / hb_e   >= MIN_SNR) & (oiii_f / oiii_e >= MIN_SNR) &
    (ha_f   / ha_e   >= MIN_SNR) & (nii_f  / nii_e  >= MIN_SNR) &
    (hb_f   > 0) & (oiii_f > 0) & (ha_f > 0) & (nii_f > 0) &
    (z_all  >= Z_SLIDER_MIN)
)

log_nii_ha  = np.log10(nii_f[keep_bpt]  / ha_f[keep_bpt]).astype(np.float32)
log_oiii_hb = np.log10(oiii_f[keep_bpt] / hb_f[keep_bpt]).astype(np.float32)
z_bpt       = z_all[keep_bpt]
embs_bpt    = embs[np.where(keep_bpt)[0]]

# Remove non-finite log ratios (unphysical)
finite = np.isfinite(log_nii_ha) & np.isfinite(log_oiii_hb)
log_nii_ha  = log_nii_ha[finite]
log_oiii_hb = log_oiii_hb[finite]
z_bpt       = z_bpt[finite]
embs_bpt    = embs_bpt[finite]

Z_SLIDER_MAX = float(z_bpt.max())
print(f'BPT galaxies: {len(embs_bpt)},  z range: {z_bpt.min():.2f} – {Z_SLIDER_MAX:.2f}')


# ══════════════════════════════════════════════════════════════════
# CELL 6 — Train linear probes for both BPT axes (encoder frozen)
# ══════════════════════════════════════════════════════════════════
n     = len(embs_bpt)
perm  = torch.randperm(n, generator=torch.Generator().manual_seed(42))
split = int(0.8 * n)
tr, te = perm[:split].numpy(), perm[split:].numpy()

emb_tr = embs_bpt[tr];  emb_te = embs_bpt[te]
x_tr   = torch.from_numpy(log_nii_ha[tr])
y_tr   = torch.from_numpy(log_oiii_hb[tr])
x_te   = torch.from_numpy(log_nii_ha[te])
y_te   = torch.from_numpy(log_oiii_hb[te])
z_te   = z_bpt[te]

# Normalise targets for stable training
xm, xs = x_tr.mean(), x_tr.std()
ym, ys = y_tr.mean(), y_tr.std()

head_x = nn.Sequential(nn.LayerNorm(cfg['D_emb']), nn.Linear(cfg['D_emb'], 1))
head_y = nn.Sequential(nn.LayerNorm(cfg['D_emb']), nn.Linear(cfg['D_emb'], 1))
opt = torch.optim.AdamW(
    list(head_x.parameters()) + list(head_y.parameters()), lr=1e-2
)

for step in range(3000):
    b  = torch.randint(len(emb_tr), (512,))
    lx = F.mse_loss(head_x(emb_tr[b]).squeeze(), (x_tr[b] - xm) / xs)
    ly = F.mse_loss(head_y(emb_tr[b]).squeeze(), (y_tr[b] - ym) / ys)
    (lx + ly).backward()
    opt.step()
    opt.zero_grad()
    if step % 500 == 0:
        print(f'  step {step:4d}  loss_x {lx.item():.4f}  loss_y {ly.item():.4f}')

head_x.eval(); head_y.eval()
with torch.no_grad():
    x_pred = head_x(emb_te).squeeze() * xs + xm
    y_pred = head_y(emb_te).squeeze() * ys + ym

x_true_np = x_te.numpy();   y_true_np = y_te.numpy()
x_pred_np = x_pred.numpy(); y_pred_np = y_pred.numpy()
print(f'\nTest MAE  log(NII/Hα):   {np.abs(x_pred_np - x_true_np).mean():.4f}')
print(f'Test MAE  log(OIII/Hβ): {np.abs(y_pred_np - y_true_np).mean():.4f}')


# ══════════════════════════════════════════════════════════════════
# CELL 7 — Interactive BPT diagram with redshift slider
# ══════════════════════════════════════════════════════════════════

# Demarcation lines from BPT.py
_Xk   = np.linspace(-1.5,  0.30, 300)
_Yk   = 0.61 / (_Xk   - 0.47) + 1.19    # Kewley+01
_Xkau = np.linspace(-1.5,  0.00, 300)
_Ykau = 0.61 / (_Xkau - 0.05) + 1.30    # Kauffmann+03

XLIM = (-1.5, 1.0)
YLIM = (-1.5, 1.5)

fig = plt.figure(figsize=(14, 6.5))
plt.subplots_adjust(bottom=0.20, wspace=0.3)
ax_true = fig.add_subplot(1, 2, 1)
ax_pred = fig.add_subplot(1, 2, 2)

for ax in (ax_true, ax_pred):
    ax.set_xlim(*XLIM);  ax.set_ylim(*YLIM)
    ax.set_xlabel(r'log([NII]$\lambda$6584 / H$\alpha$)', fontsize=11)
    ax.set_ylabel(r'log([OIII]$\lambda$5007 / H$\beta$)', fontsize=11)
    ax.plot(_Xk,   _Yk,   'k-',  lw=1.5, label='Kewley+01')
    ax.plot(_Xkau, _Ykau, 'k--', lw=1.5, label='Kauffmann+03')
    # Shade AGN region label
    ax.text(0.55,  0.85, 'AGN', transform=ax.transAxes, fontsize=9,
            color='0.4', style='italic')
    ax.text(0.10, 0.15, 'SF', transform=ax.transAxes, fontsize=9,
            color='0.4', style='italic')
    ax.legend(fontsize=8, loc='upper left')

# Faint background: all BPT-valid true positions (left) / all test predictions (right)
ax_true.scatter(log_nii_ha,  log_oiii_hb, s=2, alpha=0.06,
                color='steelblue', zorder=1, rasterized=True)
ax_pred.scatter(x_pred_np, y_pred_np,     s=2, alpha=0.06,
                color='tomato',    zorder=1, rasterized=True)

# Foreground scatter objects updated by the slider
cmap = plt.cm.plasma
norm = mcolors.Normalize(vmin=Z_SLIDER_MIN, vmax=Z_SLIDER_MAX)

sc_tr = ax_true.scatter([], [], s=22, c=[], cmap=cmap, norm=norm,
                        edgecolors='k', linewidths=0.2, zorder=4)
sc_pr = ax_pred.scatter([], [], s=22, c=[], cmap=cmap, norm=norm,
                        edgecolors='k', linewidths=0.2, zorder=4)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
fig.colorbar(sm, ax=[ax_true, ax_pred], label='spectroscopic redshift z',
             shrink=0.85, pad=0.01)

# Redshift slider
ax_sl  = fig.add_axes([0.15, 0.06, 0.70, 0.04])
slider = Slider(ax_sl, 'z', Z_SLIDER_MIN, Z_SLIDER_MAX,
                valinit=Z_SLIDER_MIN, valstep=0.05, color='mediumpurple')

def _update(val):
    zc  = slider.val
    sel = (z_te >= zc - Z_BIN) & (z_te <= zc + Z_BIN)
    n_s = int(sel.sum())

    sc_tr.set_offsets(np.c_[x_true_np[sel], y_true_np[sel]])
    sc_tr.set_array(z_te[sel])
    sc_pr.set_offsets(np.c_[x_pred_np[sel], y_pred_np[sel]])
    sc_pr.set_array(z_te[sel])

    ax_true.set_title(f'True BPT positions  (N = {n_s})', fontsize=11)
    ax_pred.set_title(f'SpecML predicted BPT  (N = {n_s})', fontsize=11)
    fig.suptitle(
        f'Redshift  z = {zc:.2f}  ±{Z_BIN}',
        fontsize=13, y=0.99
    )
    fig.canvas.draw_idle()

slider.on_changed(_update)
_update(Z_SLIDER_MIN)
plt.show()
