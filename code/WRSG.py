import os
import time
import math
import hashlib
import argparse
import warnings
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.amp import autocast, GradScaler

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*meshgrid.*indexing.*")


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Central experiment configuration: filesystem paths, DNS/physics parameters,
    the EDM diffusion schedule, training hyperparameters, evaluation, and statistics.

    This file lives in code/, so PROJECT_ROOT resolves one directory up to the
    repository root; Dataset/, Results/, and Checkpoints/ sit beside code/."""
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR     = PROJECT_ROOT / "Dataset"
    RESULTS_DIR  = PROJECT_ROOT / "Results"
    CKPT_DIR     = PROJECT_ROOT / "Checkpoints"
    FIG_DIR      = RESULTS_DIR / "figures"
    CSV_DIR      = RESULTS_DIR / "Tables"

    GRID                   = 128
    DOMAIN                 = 2.0 * math.pi
    NU_LIST                = [0.005, 0.007, 0.010, 0.013, 0.018, 0.024, 0.032]
    FORCING_K              = 4
    ALPHA_DRAG             = 0.1
    DT                     = 1e-3
    SPINUP_STEPS           = 4000
    SNAPSHOT_INTERVAL      = 200
    SNAPSHOTS_PER_REGIME   = 1000
    USE_DEALIASED_JACOBIAN = True
    SOLVER_REV             = 2

    SIGMA_MIN  = 0.002
    SIGMA_MAX  = 20.0
    SIGMA_DATA = 1.0
    RHO        = 7.0
    P_MEAN     = -1.0
    P_STD      = 1.2

    BATCH_SIZE     = 32
    EPOCHS         = 160
    LR             = 2e-4
    WEIGHT_DECAY   = 1e-5
    EMA_DECAY_MAX  = 0.999
    USE_AMP        = True
    NUM_WORKERS    = 2
    BASE_CHANNELS  = 48
    SEEDS          = [29, 47, 89, 101, 149]

    LAMBDA_ENSTROPHY = 0.05
    LAMBDA_SPECTRAL  = 0.05
    LAMBDA_STRUCT    = 0.05
    LAMBDA_INTLEN    = 0.10
    LAMBDA_FLUX      = 0.07
    LAMBDA_LOWK      = 0.0
    LOWK_KCUT        = 6
    STRUCT_R_MAX     = 8
    FLUX_SIGMA_MAX   = 1.0

    GATE_N_BINS      = 16
    GATE_RANK        = 2

    FNO_OP_MODES     = 12
    FNO_OP_WIDTH     = 64
    FNO_OP_LAYERS    = 4
    GATE_SWEEP_BINS  = [8, 16, 32]
    GATE_SWEEP_RANKS = [1, 2, 4]
    SWEEP_SEEDS      = [29, 47, 89]

    TEST_FRACTION       = 0.15
    INNER_VAL_FRACTION  = 0.10

    N_EVAL_SAMPLES  = 256
    N_ENSEMBLE      = 32
    N_SAMPLE_STEPS  = 50
    UQ_CAL_FRACTION = 0.5

    BOOTSTRAP_RESAMPLES = 2000
    BOOTSTRAP_CI        = 0.95

    @classmethod
    def setup(cls):
        """Create all output directories (data, results, checkpoints, figures, tables)."""
        for d in (cls.DATA_DIR, cls.RESULTS_DIR, cls.CKPT_DIR, cls.FIG_DIR, cls.CSV_DIR):
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def dataset_hash(cls):
        """Return a short hash of the DNS-defining parameters, used to name the dataset cache."""
        keys = ("GRID", "DOMAIN", "NU_LIST", "FORCING_K", "ALPHA_DRAG", "DT",
                "SPINUP_STEPS", "SNAPSHOT_INTERVAL", "SNAPSHOTS_PER_REGIME",
                "USE_DEALIASED_JACOBIAN", "SOLVER_REV")
        payload = repr({k: getattr(cls, k) for k in keys})
        return hashlib.sha1(payload.encode()).hexdigest()[:10]


def configure_device():
    """Select and report the compute device, preferring CUDA (cuda:0) over CPU."""
    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        print(f"[Device] CUDA visible={torch.cuda.device_count()}  "
              f"using {torch.cuda.get_device_name(0)} (cuda:0).")
        return dev
    print("[Device] No CUDA detected; using CPU.")
    return torch.device("cpu")


# ============================================================================
# Pseudo-Spectral DNS of Forced 2D Kolmogorov Flow
# ============================================================================

class KolmogorovDNS:
    """Pseudo-spectral DNS of forced 2D Kolmogorov flow in the vorticity formulation,
    integrated with explicit RK4 and 2/3-rule (plus optional 3/2-padded) dealiasing."""

    def __init__(self, N, L, nu, kf, alpha, dt, device, dealias_jacobian=True):
        """Precompute wavenumber grids, the inverse-Laplacian operator, dealiasing
        masks, and the Kolmogorov forcing (k_f^2 * cos(k_f y)) in spectral space."""
        self.N, self.L, self.nu, self.kf, self.alpha, self.dt = N, L, nu, kf, alpha, dt
        self.device = device
        self.dealias_jacobian = dealias_jacobian

        k = torch.fft.fftfreq(N, d=L / (N * 2.0 * math.pi)).to(device)
        kx, ky = torch.meshgrid(k, k, indexing="ij")
        self.kx, self.ky = kx, ky
        self.k2 = kx ** 2 + ky ** 2
        self.inv_k2 = torch.where(self.k2 > 0,
                                  1.0 / self.k2.clamp(min=1e-30),
                                  torch.zeros_like(self.k2))

        kmax = N // 3
        self.dealias = ((kx.abs() <= kmax) & (ky.abs() <= kmax)).float()
        self.N_pad = (3 * N) // 2

        x = torch.linspace(0, L, N + 1, device=device)[:-1]
        _, Y = torch.meshgrid(x, x, indexing="ij")
        self.forcing = (kf ** 2) * torch.cos(kf * Y)
        self.forcing_hat = torch.fft.fft2(self.forcing)

    @staticmethod
    def _pad_to(x_hat, N_pad):
        """Zero-pad a spectral field from N to N_pad modes (3/2 rule), rescaling so
        physical amplitudes are preserved through the later inverse FFT."""
        *batch, N, _ = x_hat.shape
        scale = (N_pad / N) ** 2
        out = x_hat.new_zeros(*batch, N_pad, N_pad)
        half = N // 2
        out[..., :half + 1, :half + 1] = x_hat[..., :half + 1, :half + 1]
        out[..., :half + 1, N_pad - half + 1:] = x_hat[..., :half + 1, half + 1:]
        out[..., N_pad - half + 1:, :half + 1] = x_hat[..., half + 1:, :half + 1]
        out[..., N_pad - half + 1:, N_pad - half + 1:] = x_hat[..., half + 1:, half + 1:]
        return out * scale

    @staticmethod
    def _truncate_to(x_hat_pad, N):
        """Truncate a padded spectral field from N_pad back to N modes; the inverse
        of _pad_to, with the matching amplitude rescaling."""
        *batch, N_pad, _ = x_hat_pad.shape
        scale = (N / N_pad) ** 2
        out = x_hat_pad.new_zeros(*batch, N, N)
        half = N // 2
        out[..., :half + 1, :half + 1] = x_hat_pad[..., :half + 1, :half + 1]
        out[..., :half + 1, half + 1:] = x_hat_pad[..., :half + 1, N_pad - half + 1:]
        out[..., half + 1:, :half + 1] = x_hat_pad[..., N_pad - half + 1:, :half + 1]
        out[..., half + 1:, half + 1:] = x_hat_pad[..., N_pad - half + 1:, N_pad - half + 1:]
        return out * scale

    def _product_dealiased(self, a_hat, b_hat):
        """Spectral product of two fields free of aliasing error, via 3/2 zero-padding,
        real-space multiplication, and truncation back to N modes."""
        a_pad = self._pad_to(a_hat, self.N_pad)
        b_pad = self._pad_to(b_hat, self.N_pad)
        a = torch.fft.ifft2(a_pad).real
        b = torch.fft.ifft2(b_pad).real
        return self._truncate_to(torch.fft.fft2(a * b), self.N)

    def _jacobian_hat(self, w_hat):
        """Return the spectral nonlinear advection term u.grad(omega), with velocity
        recovered from vorticity via the stream function; dealiased when enabled."""
        psi_hat = w_hat * self.inv_k2
        u_hat   = 1j * self.ky * psi_hat
        v_hat   = -1j * self.kx * psi_hat
        wx_hat  = 1j * self.kx * w_hat
        wy_hat  = 1j * self.ky * w_hat
        if self.dealias_jacobian:
            uwx = self._product_dealiased(u_hat, wx_hat)
            vwy = self._product_dealiased(v_hat, wy_hat)
            return (uwx + vwy) * self.dealias
        u  = torch.fft.ifft2(u_hat ).real
        v  = torch.fft.ifft2(v_hat ).real
        wx = torch.fft.ifft2(wx_hat).real
        wy = torch.fft.ifft2(wy_hat).real
        return torch.fft.fft2(u * wx + v * wy) * self.dealias

    def _rhs(self, w_hat):
        """Right-hand side of the spectral vorticity equation: advection, viscous
        diffusion, linear drag, and Kolmogorov forcing."""
        return (-self._jacobian_hat(w_hat)
                - self.nu * self.k2 * w_hat
                - self.alpha * w_hat
                + self.forcing_hat)

    def step_rk4(self, w_hat):
        """Advance the spectral vorticity one timestep with classical explicit RK4."""
        k1 = self._rhs(w_hat)
        k2 = self._rhs(w_hat + 0.5 * self.dt * k1)
        k3 = self._rhs(w_hat + 0.5 * self.dt * k2)
        k4 = self._rhs(w_hat + self.dt * k3)
        return w_hat + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _init_field(self, seed):
        """Build a random, dealiased, zero-mean initial vorticity field (spectral) for the seed."""
        g = torch.Generator(device=self.device).manual_seed(seed)
        w = 0.5 * torch.randn(self.N, self.N, generator=g, device=self.device)
        w_hat = torch.fft.fft2(w) * self.dealias
        w_hat[0, 0] = 0.0
        return w_hat

    def simulate(self, n_snapshots, snapshot_interval, spinup_steps, seed):
        """Spin up the flow, then collect n_snapshots real-space vorticity fields,
        one every snapshot_interval steps."""
        w_hat = self._init_field(seed)
        for _ in range(spinup_steps):
            w_hat = self.step_rk4(w_hat)
        snaps = torch.zeros(n_snapshots, self.N, self.N, device=self.device)
        for i in range(n_snapshots):
            for _ in range(snapshot_interval):
                w_hat = self.step_rk4(w_hat)
            snaps[i] = torch.fft.ifft2(w_hat).real
        return snaps

    def diagnostics(self, omega):
        """Compute bulk diagnostics (rms velocity, enstrophy, integral and Taylor
        scales, integral/Taylor-scale Reynolds numbers) for a batch of vorticity fields."""
        if omega.ndim == 2:
            omega = omega.unsqueeze(0)
        w_hat   = torch.fft.fft2(omega)
        psi_hat = w_hat * self.inv_k2
        u =  torch.fft.ifft2( 1j * self.ky * psi_hat).real
        v =  torch.fft.ifft2(-1j * self.kx * psi_hat).real
        u_rms     = torch.sqrt((u ** 2 + v ** 2).mean()).item()
        enstrophy = (omega ** 2).mean().item()
        lam   = math.sqrt(2.0 * u_rms ** 2 / max(enstrophy, 1e-12))
        Ek, kvals = compute_radial_spectrum(omega)
        k = kvals.float().clamp(min=1.0)
        Ek_mean = Ek.mean(0)
        L_int = math.pi * (Ek_mean / k).sum().item() / max(Ek_mean.sum().item(), 1e-12)
        return {
            "u_rms":     u_rms,
            "enstrophy": enstrophy,
            "L_int":     L_int,
            "lambda":    lam,
            "Re_int":    u_rms * L_int / self.nu,
            "Re_lam":    u_rms * lam   / self.nu,
        }


# ============================================================================
# Spectral Diagnostics
# ============================================================================

def compute_radial_spectrum(field):
    """Radially-binned power spectrum |field_hat(k)|^2 (the vorticity/enstrophy
    spectrum), summed over each integer-|k| shell; returns (Ek, k_bins)."""
    if field.ndim == 2:
        field = field.unsqueeze(0)
    field = field.float()
    B, H, W = field.shape
    Fh = torch.fft.fft2(field, norm="ortho")
    E2 = (Fh.abs() ** 2)
    ky = torch.fft.fftfreq(H, d=1.0 / H).to(field.device)
    kx = torch.fft.fftfreq(W, d=1.0 / W).to(field.device)
    KX, KY = torch.meshgrid(kx, ky, indexing="ij")
    K = torch.sqrt(KX ** 2 + KY ** 2)
    kmax = int(K.max().item()) + 1
    bins = torch.arange(0, kmax + 1, device=field.device)
    K_round = K.round().long().clamp(0, kmax - 1).flatten()
    K_idx = K_round.unsqueeze(0).expand(B, -1)
    Ek = torch.zeros(B, kmax, device=field.device).scatter_add_(1, K_idx, E2.view(B, -1))
    return Ek, bins[:kmax]


def compute_dealiased_spectrum(field, kmax_frac=0.66):
    """Radial spectrum restricted to the well-resolved band |k| <= kmax_frac * k_max,
    dropping aliasing-contaminated high modes before comparison."""
    Ek, kvals = compute_radial_spectrum(field)
    k_cut = int(kmax_frac * kvals.max().item())
    mask = kvals <= k_cut
    return Ek[:, mask], kvals[mask]


def compute_vorticity_structure_function(field, order, r_max=None):
    """Direction-averaged p-th order vorticity structure function
    S_p(r) = <|omega(x+r) - omega(x)|^p> for separations r = 1..r_max."""
    if field.ndim == 4:
        field = field.squeeze(1)
    field = field.float()
    B, H, _ = field.shape
    if r_max is None:
        r_max = H // 3
    rs = list(range(1, r_max + 1))
    S = torch.zeros(B, len(rs), device=field.device)
    for i, r in enumerate(rs):
        d1 = (field - torch.roll(field, shifts=r, dims=-1)).abs() ** order
        d2 = (field - torch.roll(field, shifts=r, dims=-2)).abs() ** order
        S[:, i] = 0.5 * (d1.mean(dim=(-1, -2)) + d2.mean(dim=(-1, -2)))
    return S, torch.tensor(rs, device=field.device, dtype=torch.float32)


def compute_integral_length(field):
    """Integral length scale L = pi * sum(E(k)/k) / sum(E(k)) from the radial
    spectrum, computed per field."""
    if field.ndim == 4:
        field = field.squeeze(1)
    Ek, kvals = compute_radial_spectrum(field)
    k = kvals.float().clamp(min=1.0)
    return math.pi * (Ek / k).sum(dim=1) / Ek.sum(dim=1).clamp(min=1e-12)


def _make_k_grid(H, W, device):
    """Build the fftfreq wavenumber grids (KX, KY, |K|, and K^2 with a small floor)
    for an H x W field."""
    kx = torch.fft.fftfreq(H, d=1.0 / H).to(device)
    ky = torch.fft.fftfreq(W, d=1.0 / W).to(device)
    KX, KY = torch.meshgrid(kx, ky, indexing="ij")
    K  = torch.sqrt(KX ** 2 + KY ** 2)
    K2 = (KX ** 2 + KY ** 2).clamp(min=1e-12)
    return KX, KY, K, K2


def compute_energy_flux(field):
    """Spectral kinetic-energy flux Pi_E(k) from the nonlinear advection transfer;
    negative values at low k signal the 2D inverse energy cascade."""
    if field.ndim == 4:
        field = field.squeeze(1)
    B, H, W = field.shape
    KX, KY, K, K2 = _make_k_grid(H, W, field.device)
    w_hat   = torch.fft.fft2(field, norm="ortho")
    psi_hat = w_hat / K2.unsqueeze(0)
    u_hat   =  1j * KY * psi_hat
    v_hat   = -1j * KX * psi_hat
    u  = torch.fft.ifft2(u_hat , norm="ortho").real
    v  = torch.fft.ifft2(v_hat , norm="ortho").real
    ux = torch.fft.ifft2( 1j * KX * u_hat, norm="ortho").real
    uy = torch.fft.ifft2( 1j * KY * u_hat, norm="ortho").real
    vx = torch.fft.ifft2( 1j * KX * v_hat, norm="ortho").real
    vy = torch.fft.ifft2( 1j * KY * v_hat, norm="ortho").real
    adv_u_hat = torch.fft.fft2(u * ux + v * uy, norm="ortho")
    adv_v_hat = torch.fft.fft2(u * vx + v * vy, norm="ortho")
    T_E = -(u_hat.conj() * adv_u_hat + v_hat.conj() * adv_v_hat).real
    kmax = int(K.max().item()) + 1
    K_round = K.round().long().clamp(0, kmax - 1).flatten()
    T_shell = torch.zeros(B, kmax, device=field.device).scatter_add_(
        1, K_round.unsqueeze(0).expand(B, -1), T_E.view(B, -1))
    Pi_E = -T_shell.cumsum(dim=1)
    return Pi_E, torch.arange(kmax, device=field.device, dtype=torch.float32)


def compute_enstrophy_flux(field):
    """Spectral enstrophy flux Pi_Z(k) from the vorticity advection transfer;
    positive values at high k signal the forward enstrophy cascade."""
    if field.ndim == 4:
        field = field.squeeze(1)
    B, H, W = field.shape
    KX, KY, K, K2 = _make_k_grid(H, W, field.device)
    w_hat = torch.fft.fft2(field, norm="ortho")
    psi_hat = w_hat / K2.unsqueeze(0)
    u = torch.fft.ifft2( 1j * KY * psi_hat, norm="ortho").real
    v = torch.fft.ifft2(-1j * KX * psi_hat, norm="ortho").real
    nl = (u * torch.fft.ifft2(1j * KX * w_hat, norm="ortho").real +
          v * torch.fft.ifft2(1j * KY * w_hat, norm="ortho").real)
    nl_hat = torch.fft.fft2(nl, norm="ortho")
    T_Z = -(w_hat.conj() * nl_hat).real
    kmax = int(K.max().item()) + 1
    K_round = K.round().long().clamp(0, kmax - 1).flatten()
    T_shell = torch.zeros(B, kmax, device=field.device).scatter_add_(
        1, K_round.unsqueeze(0).expand(B, -1), T_Z.view(B, -1))
    Pi_Z = -T_shell.cumsum(dim=1)
    return Pi_Z, torch.arange(kmax, device=field.device, dtype=torch.float32)


def fit_inertial_slope(Ek_mean, k_lo, k_hi):
    """Least-squares fit of the log-log spectrum slope over the band [k_lo, k_hi];
    returns (slope, intercept, R^2), or NaNs if too few valid modes."""
    device = Ek_mean.device
    k = torch.arange(Ek_mean.shape[0], device=device, dtype=torch.float32)
    mask = (k >= k_lo) & (k <= k_hi) & (Ek_mean > 1e-12)
    if mask.sum() < 4:
        return float("nan"), float("nan"), float("nan")
    logk = torch.log10(k[mask] + 1e-8)
    logE = torch.log10(Ek_mean[mask] + 1e-12)
    A = torch.stack([logk, torch.ones_like(logk)], dim=1)
    sol, *_ = torch.linalg.lstsq(A, logE.unsqueeze(1))
    slope, intercept = sol[0, 0].item(), sol[1, 0].item()
    pred = slope * logk + intercept
    ss_res = ((logE - pred) ** 2).sum().item()
    ss_tot = ((logE - logE.mean()) ** 2).sum().item()
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return slope, intercept, r2


# ============================================================================
# Dataset Generation and DNS Validation
# ============================================================================

def generate_dataset(cfg, device):
    """Generate (or load from cache) the multi-regime Kolmogorov dataset: run DNS at
    each viscosity, record diagnostics, globally normalize the fields, and save."""
    cache = cfg.DATA_DIR / f"kolmogorov_{cfg.dataset_hash()}.pt"
    if cache.exists():
        print(f"[Data] Cached dataset found: {cache.name}")
        return torch.load(cache, map_location="cpu")

    print(f"[Data] Generating Kolmogorov flow at N={cfg.GRID}, {len(cfg.NU_LIST)} regimes.")
    print(f"[Data] Dataset hash: {cfg.dataset_hash()}")
    all_fields, all_re, diagnostics = [], [], []
    for r_idx, nu in enumerate(cfg.NU_LIST):
        print(f"[Data]   Regime {r_idx+1}/{len(cfg.NU_LIST)}  nu={nu}")
        dns = KolmogorovDNS(cfg.GRID, cfg.DOMAIN, nu, cfg.FORCING_K, cfg.ALPHA_DRAG,
                            cfg.DT, device, dealias_jacobian=cfg.USE_DEALIASED_JACOBIAN)
        snaps = dns.simulate(cfg.SNAPSHOTS_PER_REGIME, cfg.SNAPSHOT_INTERVAL,
                             cfg.SPINUP_STEPS, seed=1000 + r_idx)
        diag = dns.diagnostics(snaps[-256:])
        diag["regime"] = r_idx; diag["nu"] = nu
        diagnostics.append(diag)
        all_fields.append(snaps.cpu())
        all_re.append(torch.full((cfg.SNAPSHOTS_PER_REGIME,), float(r_idx)))
        print(f"[Data]    u_rms={diag['u_rms']:.3f}  L_int={diag['L_int']:.3f}  "
              f"lambda={diag['lambda']:.3f}  Re_int={diag['Re_int']:.1f}  "
              f"Re_lam={diag['Re_lam']:.1f}")

    fields = torch.cat(all_fields, dim=0)
    re_labels = torch.cat(all_re, dim=0).long()
    mean, std = fields.mean(), fields.std()
    fields_norm = (fields - mean) / (std + 1e-8)
    payload = {
        "fields":      fields_norm.unsqueeze(1).float(),
        "re_labels":   re_labels,
        "raw_mean":    mean.item(),
        "raw_std":     std.item(),
        "nu_list":     cfg.NU_LIST,
        "grid":        cfg.GRID,
        "diagnostics": diagnostics,
        "hash":        cfg.dataset_hash(),
    }
    torch.save(payload, cache)
    print(f"[Data] Saved -> {cache.name}")
    return payload


def validate_dns(cfg, payload, device):
    """Sanity-check the DNS: plot per-regime vorticity power spectra, fit inertial-range
    slopes, and write the figure plus a CSV table of slopes and Reynolds numbers."""
    print("[DNS]  Validating vorticity power spectra and fitting inertial-range slopes.")
    fields = payload["fields"].to(device) * payload["raw_std"] + payload["raw_mean"]
    re_labels = payload["re_labels"].to(device)
    rows = []
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    cmap = plt.get_cmap("viridis")
    for r in range(len(cfg.NU_LIST)):
        sel = (re_labels == r)
        Ek, kvals = compute_dealiased_spectrum(fields[sel].squeeze(1)[:256])
        Ek_mean_t = Ek.mean(0)
        Ek_mean = Ek_mean_t.cpu().numpy()
        k = kvals.cpu().numpy()
        k_lo = cfg.FORCING_K + 2
        k_hi = max(k_lo + 3, int(kvals.max().item()) // 4)
        slope, intercept, r2 = fit_inertial_slope(Ek_mean_t, k_lo, k_hi)
        diag = payload["diagnostics"][r]
        rows.append({"regime": r, "nu": cfg.NU_LIST[r],
                     "Re_int": diag["Re_int"], "Re_lam": diag["Re_lam"],
                     "L_int": diag["L_int"], "lambda": diag["lambda"],
                     "fit_k_lo": int(k_lo), "fit_k_hi": int(k_hi),
                     "fit_slope": slope, "fit_R2": r2})
        nz = (k > 0) & (Ek_mean > 0)
        color = cmap(0.15 + 0.7 * r / max(1, len(cfg.NU_LIST) - 1))
        ax.loglog(k[nz], Ek_mean[nz], color=color, lw=1.8,
                  label=(f"nu={cfg.NU_LIST[r]}  Re_int={diag['Re_int']:.0f}  "
                         f"slope={slope:.2f}  (R^2={r2:.2f})"))
        kk = np.array([k_lo, k_hi], dtype=float)
        ax.loglog(kk, 10 ** (slope * np.log10(kk) + intercept),
                  color=color, ls=":", lw=1.2)
        if k_lo < len(Ek_mean) and Ek_mean[k_lo] > 0:
            amp_ref = Ek_mean[k_lo] * (k_lo ** 1)
            k_ref = np.linspace(k_lo, max(k_lo + 4, kvals.max().item() // 2), 30)
            ax.loglog(k_ref, amp_ref * k_ref ** (-1.0), color=color, ls="--",
                      lw=0.8, alpha=0.55,
                      label=("Kraichnan $k^{-1}$ (vorticity spectrum)" if r == 0 else None))
    ax.axvline(cfg.FORCING_K, color="orange", ls=":", lw=1.0, label=f"$k_f={cfg.FORCING_K}$")
    ax.set_xlabel("k"); ax.set_ylabel(r"$|\hat{\omega}(k)|^2$")
    ax.set_title(f"DNS Vorticity Power Spectra and Inertial-Range Slope Fits  (N={cfg.GRID})",
                 fontsize=12)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / "dns_validation.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    df = pd.DataFrame(rows)
    df.to_csv(cfg.CSV_DIR / "dns_validation.csv", index=False)
    print("[DNS]  Slope fits:\n" + df.to_string(index=False))


class TurbulenceDataset(Dataset):
    """Thin Dataset wrapping vorticity-field tensors and their regime indices (indices into
    NU_LIST identifying the Reynolds/viscosity regime of each snapshot)."""
    def __init__(self, fields, re_labels):
        """Store the field tensor and its matching regime-index tensor."""
        self.fields = fields
        self.re_labels = re_labels
    def __len__(self):
        """Number of snapshots in the dataset."""
        return self.fields.shape[0]
    def __getitem__(self, i):
        """Return the (field, regime_index) pair at index i."""
        return self.fields[i], self.re_labels[i]


def prepare_splits(full_ds, cfg):
    """Carve a fixed test holdout using a constant seed, so the test set is identical
    across every variant and seed; returns (train_indices, test_indices)."""
    N = len(full_ds)
    n_test = max(1, int(round(cfg.TEST_FRACTION * N)))
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(N, generator=g).tolist()
    test_indices  = perm[:n_test]
    train_indices = perm[n_test:]
    return train_indices, test_indices


# ============================================================================
# EDM Diffusion Schedule
# ============================================================================

class EDMSchedule:
    """Karras et al. (2022) EDM noise schedule: log-normal sigma sampling for training,
    a rho-spaced sigma grid for sampling, plus preconditioning and loss weights."""
    def __init__(self, sigma_min, sigma_max, sigma_data, rho, p_mean, p_std, device):
        """Store the sigma range, data scale (sigma_data), rho, and log-sigma sampling parameters."""
        self.sigma_min, self.sigma_max = sigma_min, sigma_max
        self.sigma_data, self.rho      = sigma_data, rho
        self.p_mean, self.p_std        = p_mean, p_std
        self.device = device

    def sample_sigma_train(self, B):
        """Draw B training noise levels from the log-normal sigma distribution,
        clamped to [sigma_min, sigma_max]."""
        eps = torch.randn(B, device=self.device)
        return (eps * self.p_std + self.p_mean).exp().clamp(self.sigma_min, self.sigma_max)

    def build_inference_schedule(self, n_steps):
        """Build the decreasing rho-spaced sigma grid for sampling, with a trailing
        zero appended (returns n_steps + 1 values)."""
        i = torch.arange(n_steps, device=self.device, dtype=torch.float64)
        t = (self.sigma_max ** (1 / self.rho) +
             i / (n_steps - 1) * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))) ** self.rho
        return torch.cat([t, torch.zeros(1, device=self.device, dtype=torch.float64)]).float()

    def preconditioning(self, sigma):
        """Return the EDM preconditioning coefficients (c_skip, c_out, c_in, c_noise)
        for the given noise level sigma."""
        sd = self.sigma_data
        c_skip  = sd ** 2 / (sigma ** 2 + sd ** 2)
        c_out   = sigma * sd / (sigma ** 2 + sd ** 2).sqrt()
        c_in    = 1.0 / (sigma ** 2 + sd ** 2).sqrt()
        c_noise = 0.25 * sigma.log()
        return c_skip, c_out, c_in, c_noise

    def loss_weight(self, sigma):
        """EDM per-sigma loss weight: (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2."""
        return (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2


# ============================================================================
# Network Components
# ============================================================================

class SinusoidalEmbedding(nn.Module):
    """Sinusoidal embedding mapping a scalar (here the EDM c_noise) to `dim` features."""
    def __init__(self, dim):
        """Store the output embedding dimension."""
        super().__init__(); self.dim = dim
    def forward(self, t):
        """Map each scalar t to interleaved sin/cos features at geometric frequencies."""
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / max(half - 1, 1))
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class ConditionEmbedding(nn.Module):
    """Fuse the noise-level embedding with a CONTINUOUS log-viscosity embedding into one
    vector. Conditioning on a continuous scalar (normalized log-nu) rather than a discrete
    regime index lets one model span all viscosity regimes on a shared physical scale."""
    def __init__(self, dim):
        """Build sinusoidal embeddings for the (noise-level, log-viscosity) scalars and the fusion MLP."""
        super().__init__()
        self.t_embed  = SinusoidalEmbedding(dim)
        self.nu_embed = SinusoidalEmbedding(dim)
        self.mlp = nn.Sequential(nn.Linear(dim * 2, dim * 4), nn.SiLU(),
                                 nn.Linear(dim * 4, dim))
    def forward(self, c_noise, nu_cond):
        """Embed (c_noise, normalized-log-nu) and return the fused conditioning vector."""
        return self.mlp(torch.cat([self.t_embed(c_noise), self.nu_embed(nu_cond)], dim=1))


class ResidualBlock(nn.Module):
    """Two-conv residual block with GroupNorm, SiLU, and additive conditioning."""
    def __init__(self, in_ch, out_ch, emb_dim):
        """Build the two conv+norm layers, the embedding projection, and the skip path."""
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x, emb):
        """Apply conv-norm-SiLU twice with injected conditioning, then add the skip."""
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.emb_proj(emb).unsqueeze(-1).unsqueeze(-1)
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class DownsampleBlock(nn.Module):
    """Strided-convolution 2x spatial downsampling."""
    def __init__(self, ch):
        """Build the stride-2 downsampling convolution."""
        super().__init__(); self.op = nn.Conv2d(ch, ch, 4, stride=2, padding=1)
    def forward(self, x):
        """Halve the spatial resolution."""
        return self.op(x)


class UpsampleBlock(nn.Module):
    """Transposed-convolution 2x spatial upsampling."""
    def __init__(self, ch):
        """Build the stride-2 transposed upsampling convolution."""
        super().__init__(); self.op = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)
    def forward(self, x):
        """Double the spatial resolution."""
        return self.op(x)


# ============================================================================
# Bottleneck Variants
# ============================================================================

class WavenumberResolvedGate(nn.Module):
    """The WRSG bottleneck: a conditioning-driven, low-rank, per-channel x
    per-radial-wavenumber-bin multiplicative gate applied in 2D Fourier space.
    The learnable `scale` starts at zero, so the module is identity at init."""
    def __init__(self, channels, emb_dim, n_radial_bins=16, rank=2):
        """Build the gate MLP that emits the low-rank factors and bias, plus the zero-init scale."""
        super().__init__()
        self.C        = channels
        self.n_bins   = n_radial_bins
        self.rank     = rank
        out_dim = rank * channels + rank * n_radial_bins + channels
        self.gate_mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim), nn.SiLU(),
            nn.Linear(2 * emb_dim, out_dim),
        )
        self.scale = nn.Parameter(torch.zeros(1))

    def _bin_indices(self, H, W, device):
        """Map each 2D wavenumber to a radial bin index in [0, n_bins-1] by normalized |k|."""
        kx = torch.fft.fftfreq(W, d=1.0 / W).to(device)
        ky = torch.fft.fftfreq(H, d=1.0 / H).to(device)
        KX, KY = torch.meshgrid(kx, ky, indexing="ij")
        K = torch.sqrt(KX ** 2 + KY ** 2)
        return (K / (K.max() + 1e-8) * (self.n_bins - 1)).round().long().clamp(0, self.n_bins - 1)

    def forward(self, x, emb):
        """Build the per-channel, per-radial-bin gate from the embedding and apply it
        multiplicatively to the field's spectrum; return the real inverse transform."""
        B, C, H, W = x.shape
        out = self.gate_mlp(emb)
        a = out[:, : self.rank * C].view(B, self.rank, C)
        b = out[:, self.rank * C : self.rank * (C + self.n_bins)].view(B, self.rank, self.n_bins)
        d = out[:, self.rank * (C + self.n_bins) :]
        gate_ck = torch.einsum("brc,brk->bck", a, b) + d.unsqueeze(-1)
        gate_ck = torch.sigmoid(gate_ck)
        gate_map = gate_ck[:, :, self._bin_indices(H, W, x.device)]
        x_hat = torch.fft.fft2(x, norm="ortho")
        x_hat = x_hat * (1.0 + self.scale * (2.0 * gate_map - 1.0))
        return torch.fft.ifft2(x_hat, norm="ortho").real


class SqueezeExciteGate(nn.Module):
    """Squeeze-and-excitation bottleneck: conditioning-aware channel reweighting."""
    def __init__(self, channels, emb_dim):
        """Build the squeeze-excite MLP over pooled channels concatenated with the embedding."""
        super().__init__()
        hidden = max(channels // 2, 8)
        self.gate_mlp = nn.Sequential(
            nn.Linear(channels + emb_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, channels), nn.Sigmoid(),
        )
    def forward(self, x, emb):
        """Rescale each channel by a gate computed from global-pooled features and the embedding."""
        z = x.mean(dim=(-1, -2))
        g = self.gate_mlp(torch.cat([z, emb], dim=1))
        return x * g.unsqueeze(-1).unsqueeze(-1)


class FNOBlock(nn.Module):
    """Fourier Neural Operator bottleneck: a learned spectral convolution over the
    lowest `modes` Fourier modes plus a pointwise (1x1) convolution residual."""
    def __init__(self, channels, modes=8):
        """Initialize the truncated complex spectral weights and the 1x1 conv."""
        super().__init__()
        self.modes = modes
        scale = 1.0 / (channels * channels)
        self.weight = nn.Parameter(scale * torch.randn(channels, channels, modes, modes, 2))
        self.conv = nn.Conv2d(channels, channels, 1)
    def _cmul(self, x_ft, w):
        """Complex channel-mixing multiply between the spectral input and the complex weight."""
        w_c = torch.complex(w[..., 0], w[..., 1])
        return torch.einsum("bixy,ioxy->boxy", x_ft, w_c)
    def forward(self, x):
        """Apply the truncated-mode spectral convolution and add the pointwise conv."""
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x.float(), norm="ortho")
        out_ft = torch.zeros(B, C, H, W // 2 + 1, device=x.device, dtype=torch.complex64)
        m = min(self.modes, H // 2, W // 2 + 1)
        out_ft[:, :,  :m, :m] = self._cmul(x_ft[:, :,  :m, :m], self.weight[:, :, :m, :m])
        out_ft[:, :, -m:, :m] = self._cmul(x_ft[:, :, -m:, :m], self.weight[:, :, :m, :m])
        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho") + self.conv(x.float())


# ============================================================================
# U-Net Denoiser
# ============================================================================

class WRSGFNOBottleneck(nn.Module):
    """Hybrid bottleneck: the WRSG spectral gate (per-shell amplitude control for
    spectral fidelity) plus an FNO spectral convolution (cross-mode mixing for the
    inter-scale transfer that governs energy/enstrophy fluxes), summed."""
    def __init__(self, channels, emb_dim, n_radial_bins=16, rank=2, fno_modes=8):
        """Build the WRSG gate and the FNO spectral-convolution block."""
        super().__init__()
        self.gate = WavenumberResolvedGate(channels, emb_dim, n_radial_bins, rank)
        self.fno  = FNOBlock(channels, modes=fno_modes)
        self.fno_scale = nn.Parameter(torch.tensor(1.0))
    def forward(self, x, emb):
        """Return the gate output plus the (learnably scaled) FNO spectral conv. NOTE: a
        small init (e.g. 0.5) tempers the forward-enstrophy overshoot but also gives back
        the inverse-cascade recovery the FNO branch provides -- the two are coupled -- so
        the scale is init neutral (1.0) and the overshoot is reported as a trade-off."""
        return self.gate(x, emb) + self.fno_scale * self.fno(x)


class TurbulenceDenoiser(nn.Module):
    """Conditional U-Net EDM denoiser for vorticity fields, with a swappable bottleneck
    selected per ablation variant: vanilla / se / fno (FNO bottleneck) / wrsg_gate (gate only) /
    wrsg_phys (losses only) / wrsg (main model: gate + losses) / wrsg_fno (gate + FNO + losses)."""
    _BOTTLENECK_OF = {
        "vanilla":   "none",
        "se":        "se",
        "fno":       "fno",
        "wrsg_gate": "wrsg",
        "wrsg_phys": "none",
        "wrsg":      "wrsg",
        "wrsg_fno":  "wrsg_fno",
    }
    VARIANTS = tuple(_BOTTLENECK_OF.keys())

    def __init__(self, variant, base_ch=48, emb_dim=128, fno_modes=8,
                 gate_bins=16, gate_rank=2):
        """Build the encoder, the variant-selected bottleneck, the skip-connected
        decoder, and the conditioning embedding."""
        super().__init__()
        assert variant in self.VARIANTS, f"unknown variant {variant!r}"
        self.variant = variant
        self._arch = self._BOTTLENECK_OF[variant]
        self.emb = ConditionEmbedding(emb_dim)
        self.in_conv = nn.Conv2d(1, base_ch, 3, padding=1)
        self.b1 = ResidualBlock(base_ch,     base_ch,     emb_dim)
        self.d1 = DownsampleBlock(base_ch)
        self.b2 = ResidualBlock(base_ch,     base_ch * 2, emb_dim)
        self.d2 = DownsampleBlock(base_ch * 2)
        self.b3 = ResidualBlock(base_ch * 2, base_ch * 4, emb_dim)
        bch = base_ch * 4
        if   self._arch == "none": self.bottleneck = nn.Identity()
        elif self._arch == "se":   self.bottleneck = SqueezeExciteGate(bch, emb_dim)
        elif self._arch == "fno":  self.bottleneck = FNOBlock(bch, modes=fno_modes)
        elif self._arch == "wrsg": self.bottleneck = WavenumberResolvedGate(bch, emb_dim, gate_bins, gate_rank)
        elif self._arch == "wrsg_fno": self.bottleneck = WRSGFNOBottleneck(bch, emb_dim, gate_bins, gate_rank, fno_modes)
        else: raise ValueError(f"unknown bottleneck arch {self._arch!r}")
        self.u2 = UpsampleBlock(base_ch * 4)
        self.b4 = ResidualBlock(base_ch * 4 + base_ch * 2, base_ch * 2, emb_dim)
        self.u1 = UpsampleBlock(base_ch * 2)
        self.b5 = ResidualBlock(base_ch * 2 + base_ch,     base_ch,     emb_dim)
        self.out_conv = nn.Conv2d(base_ch, 1, 3, padding=1)

    def forward(self, x, c_noise, nu_cond):
        """Denoise the field conditioned on (c_noise, log-viscosity): encode, apply the
        bottleneck, decode with skip connections, and project to one output channel."""
        emb = self.emb(c_noise, nu_cond)
        h1 = self.b1(self.in_conv(x), emb)
        h2 = self.b2(self.d1(h1), emb)
        h3 = self.b3(self.d2(h2), emb)
        if self._arch in ("se", "wrsg", "wrsg_fno"):
            h3 = h3 + self.bottleneck(h3, emb)
        elif self._arch == "fno":
            h3 = h3 + self.bottleneck(h3)
        elif self._arch != "none":
            raise ValueError(f"unhandled bottleneck arch {self._arch!r}")
        u2 = self.b4(torch.cat([self.u2(h3), h2], dim=1), emb)
        u1 = self.b5(torch.cat([self.u1(u2), h1], dim=1), emb)
        return self.out_conv(u1)


def denoise_preconditioned(model, x, sigma, nu_cond, sched):
    """Apply EDM preconditioning around the raw network:
    D(x; sigma) = c_skip * x + c_out * F(c_in * x), giving the denoised x0 estimate.
    nu_cond is the continuous (normalized log-viscosity) conditioning value per sample."""
    c_skip, c_out, c_in, c_noise = sched.preconditioning(sigma)
    c_skip_b = c_skip.view(-1, 1, 1, 1)
    c_out_b  = c_out .view(-1, 1, 1, 1)
    c_in_b   = c_in  .view(-1, 1, 1, 1)
    f = model(c_in_b * x, c_noise, nu_cond)
    return c_skip_b * x + c_out_b * f


def count_parameters(model):
    """Count the trainable parameters of a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_nu_cond_lut(nu_list, device):
    """Per-regime continuous conditioning values: z-scored log-viscosity over the nu grid,
    so every regime shares one fixed, normalized conditioning scale."""
    logs = torch.log(torch.tensor(list(nu_list), dtype=torch.float32))
    return ((logs - logs.mean()) / (logs.std() + 1e-8)).to(device)


class FNO2dDenoiser(nn.Module):
    """Standalone Fourier Neural Operator denoiser used as an EXTERNAL baseline: lift to
    `width` channels, apply `n_layers` spectral-convolution (FNO) layers with the
    (noise-level, log-viscosity) conditioning injected per layer, then project back to one
    channel. A proper FNO operator -- not the FNO bottleneck embedded in the U-Net."""
    def __init__(self, modes=12, width=64, n_layers=4, emb_dim=128):
        """Build the lifting conv, the stacked spectral layers with per-layer conditioning
        projections, and the projection head."""
        super().__init__()
        self.variant = "fno_operator"
        self.emb = ConditionEmbedding(emb_dim)
        self.lift = nn.Conv2d(1, width, 1)
        self.layers   = nn.ModuleList([FNOBlock(width, modes=modes) for _ in range(n_layers)])
        self.emb_proj = nn.ModuleList([nn.Linear(emb_dim, width) for _ in range(n_layers)])
        self.proj = nn.Sequential(nn.Conv2d(width, width, 1), nn.SiLU(),
                                  nn.Conv2d(width, 1, 1))
    def forward(self, x, c_noise, nu_cond):
        """Denoise via lift -> conditioned FNO layers (SiLU) -> project to one channel."""
        emb = self.emb(c_noise, nu_cond)
        h = self.lift(x)
        for layer, proj in zip(self.layers, self.emb_proj):
            h = F.silu(layer(h) + proj(emb).unsqueeze(-1).unsqueeze(-1))
        return self.proj(h)


def build_model(variant, cfg):
    """Construct the denoiser for a variant: the standalone FNO-operator baseline for
    'fno_operator', otherwise the U-Net (TurbulenceDenoiser) with the selected bottleneck."""
    if variant == "fno_operator":
        return FNO2dDenoiser(modes=cfg.FNO_OP_MODES, width=cfg.FNO_OP_WIDTH,
                             n_layers=cfg.FNO_OP_LAYERS)
    return TurbulenceDenoiser(variant, base_ch=cfg.BASE_CHANNELS,
                              gate_bins=cfg.GATE_N_BINS, gate_rank=cfg.GATE_RANK)


# ============================================================================
# Physics-Informed Losses
# ============================================================================

def loss_enstrophy(x0_pred, x0_true):
    """L1 loss between the per-sample enstrophy (mean omega^2) of prediction and target."""
    return F.l1_loss(x0_pred.float().pow(2).mean(dim=(-1, -2)),
                     x0_true.float().pow(2).mean(dim=(-1, -2)))


def loss_spectral(x0_pred, x0_true):
    """L1 loss between the Fourier amplitude spectra of prediction and target.
    FFT runs in float32 (accurate and portable under AMP autocast)."""
    Fp = torch.fft.fft2(x0_pred.float(), norm="ortho").abs()
    Ft = torch.fft.fft2(x0_true.float(), norm="ortho").abs()
    return F.l1_loss(Fp, Ft)


def loss_structure(x0_pred, x0_true, orders=(2, 3), r_max=8):
    """L1 loss in log-space between the order-`order` vorticity structure functions of
    prediction and target, over small separations r=1..r_max. Targets the small-scale
    increments (S2/S3) that the radial spectral gate tends to distort."""
    loss = 0.0
    for p in orders:
        Sp, _ = compute_vorticity_structure_function(x0_pred, p, r_max)
        St, _ = compute_vorticity_structure_function(x0_true, p, r_max)
        loss = loss + F.l1_loss(torch.log(Sp.clamp(min=1e-12)),
                                torch.log(St.clamp(min=1e-12)))
    return loss


def loss_integral_length(x0_pred, x0_true):
    """L1 loss between the integral length scales of prediction and target. A large-scale
    (low-wavenumber) constraint that counteracts the small-scale bias of the structure
    and flux losses, preventing the integral-length-scale regression."""
    return F.l1_loss(compute_integral_length(x0_pred), compute_integral_length(x0_true))


def loss_flux(x0_pred, x0_true):
    """L1 loss between the peak-normalized energy and enstrophy spectral fluxes of
    prediction and target (the flux-consistency term of the main WRSG model)."""
    xp, xt = x0_pred.float(), x0_true.float()
    Pe_p, _ = compute_energy_flux(xp)
    Pe_t, _ = compute_energy_flux(xt)
    Pz_p, _ = compute_enstrophy_flux(xp)
    Pz_t, _ = compute_enstrophy_flux(xt)
    ne = Pe_t.abs().amax(dim=1, keepdim=True).clamp(min=1e-6).detach()
    nz = Pz_t.abs().amax(dim=1, keepdim=True).clamp(min=1e-6).detach()
    return F.l1_loss(Pe_p / ne, Pe_t / ne) + F.l1_loss(Pz_p / nz, Pz_t / nz)


def loss_lowk_spectrum(x0_pred, x0_true, k_cut=6):
    """L1 loss in log-space between the low-wavenumber (|k|<=k_cut) radial energy spectra of
    prediction and target. A dense constraint over the energy-containing range, where the
    integral length scale lives, supplying the large-scale gradient signal that the
    single-scalar integral-length loss cannot."""
    Ep, kk = compute_radial_spectrum(x0_pred.squeeze(1))
    Et, _  = compute_radial_spectrum(x0_true.squeeze(1))
    band = kk <= k_cut
    return F.l1_loss(torch.log(Ep[:, band].clamp(min=1e-12)),
                     torch.log(Et[:, band].clamp(min=1e-12)))


# ============================================================================
# EMA and Checkpointing
# ============================================================================

class EMA:
    """Exponential moving average of the model's trainable parameters, used as the
    weights for evaluation and sampling."""
    def __init__(self, model, decay):
        """Initialize the shadow parameters as a detached clone of the model's trainable parameters."""
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    @torch.no_grad()
    def update(self, model):
        """Move each shadow parameter toward the current model parameter by (1 - decay)."""
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
    def state_dict(self):
        """Return the shadow (EMA) parameters on CPU for checkpointing."""
        return {n: v.cpu() for n, v in self.shadow.items()}
    def load_state_dict(self, state):
        """Load shadow parameters from a checkpoint, warning about any missing or unexpected keys."""
        missing    = [n for n in self.shadow if n not in state]
        unexpected = [n for n in state    if n not in self.shadow]
        if missing:
            print(f"[EMA]  Warning: {len(missing)} shadow params missing from checkpoint "
                  f"(first 3: {missing[:3]}).")
        if unexpected:
            print(f"[EMA]  Warning: {len(unexpected)} checkpoint keys not in shadow "
                  f"(first 3: {unexpected[:3]}).")
        for n in self.shadow:
            if n in state:
                self.shadow[n] = state[n].to(self.shadow[n].device)
    @torch.no_grad()
    def copy_to(self, model):
        """Copy the shadow (EMA) parameters into the live model in place."""
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n].to(p.device))


def adaptive_ema_decay(total_steps, decay_max=0.999, fraction=0.2):
    """Choose an EMA decay so the averaging window spans `fraction` of training,
    capped at decay_max."""
    n_eff = max(50, int(fraction * total_steps))
    decay = 1.0 - 1.0 / n_eff
    return min(decay, decay_max)


def ckpt_path(cfg, variant, seed):
    """Return the checkpoint path for a given variant and seed under the current config."""
    return cfg.CKPT_DIR / f"{variant}_seed{seed}_r{cfg.SOLVER_REV}.pt"


def save_ckpt(path, model, optim, ema, epoch, history, compute):
    """Save model, optimizer, EMA, epoch, history, and compute stats to a checkpoint."""
    torch.save({"model": model.state_dict(), "optim": optim.state_dict(),
                "ema": ema.state_dict(), "epoch": epoch, "history": history,
                "compute": compute}, path)


def load_ckpt(path, model, optim, ema):
    """Load a checkpoint into model/optimizer/EMA if present; return (epoch, history, compute).
    A truncated or otherwise unreadable checkpoint (e.g. an interrupted write) is treated as
    absent so training restarts from scratch rather than crashing the job."""
    if not path.exists():
        return 0, [], {}
    try:
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state["model"])
        optim.load_state_dict(state["optim"])
        if "ema" in state:
            ema.load_state_dict(state["ema"])
        return state["epoch"], state.get("history", []), state.get("compute", {})
    except Exception as e:
        print(f"[Ckpt] Warning: {path.name} unreadable ({type(e).__name__}); "
              f"restarting this run from scratch.")
        return 0, [], {}


# ============================================================================
# Training Loop
# ============================================================================

def train_one_model(variant, seed, cfg, train_pool, device, use_physics, use_flux=False):
    """Train one denoiser variant at one seed: carve an inner train/val split, run EDM
    diffusion training (optionally with enstrophy/spectral/flux physics losses), track an
    EMA, apply warmup+cosine LR, and resume from any checkpoint. Returns the EMA-weighted
    model, the diffusion schedule, the epoch history, and compute stats."""
    print(f"\n[Train] === variant={variant.upper()}  seed={seed}  "
          f"phys={'on' if use_physics else 'off'} ===")
    torch.manual_seed(seed); np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    N = len(train_pool)
    n_val = max(1, int(round(cfg.INNER_VAL_FRACTION * N)))
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g).tolist()
    inner_val_idx = perm[:n_val]
    inner_tr_idx  = perm[n_val:]
    train_ds = Subset(train_pool, inner_tr_idx)
    val_ds   = Subset(train_pool, inner_val_idx)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.NUM_WORKERS, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False,
                              num_workers=cfg.NUM_WORKERS, pin_memory=(device.type == "cuda"))

    model = build_model(variant, cfg).to(device)
    nu_lut = make_nu_cond_lut(cfg.NU_LIST, device)
    n_params = count_parameters(model)
    print(f"[Train] Parameters: {n_params/1e6:.3f} M")

    steps_per_epoch = max(1, len(train_loader))
    total_steps     = cfg.EPOCHS * steps_per_epoch
    ema_decay       = adaptive_ema_decay(total_steps, decay_max=cfg.EMA_DECAY_MAX)
    n_eff           = int(1.0 / (1.0 - ema_decay))
    print(f"[Train] Total steps: {total_steps}  EMA decay: {ema_decay:.5f}  "
          f"(n_eff = {n_eff} = {100.0*n_eff/total_steps:.0f}% of training)")
    ema = EMA(model, decay=ema_decay)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    warmup = max(1, cfg.EPOCHS // 20)
    def lr_lambda(epoch):
        """Linear LR warmup for the first `warmup` epochs, then cosine decay toward zero."""
        if epoch < warmup:
            return (epoch + 1) / warmup
        prog = (epoch - warmup) / max(1, cfg.EPOCHS - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * prog))
    lr_sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    diff_sched = EDMSchedule(cfg.SIGMA_MIN, cfg.SIGMA_MAX, cfg.SIGMA_DATA, cfg.RHO,
                             cfg.P_MEAN, cfg.P_STD, device)
    scaler = GradScaler("cuda", enabled=cfg.USE_AMP and device.type == "cuda")

    cpath = ckpt_path(cfg, variant, seed)
    start_epoch, history, prior_compute = load_ckpt(cpath, model, optim, ema)
    if start_epoch > 0:
        for _ in range(start_epoch):
            lr_sched.step()
        print(f"[Train] Resumed from epoch {start_epoch}.")

    t0 = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(start_epoch, cfg.EPOCHS):
        model.train()
        ep_loss = ep_mse = ep_phys = 0.0; nb = 0
        for x, re in train_loader:
            x  = x .to(device, non_blocking=True)
            re = re.to(device, non_blocking=True)
            nu_cond = nu_lut[re]
            B = x.shape[0]
            sigma = diff_sched.sample_sigma_train(B)
            noise = torch.randn_like(x) * sigma.view(-1, 1, 1, 1)
            x_noisy = x + noise
            optim.zero_grad(set_to_none=True)
            ctx = autocast("cuda") if (cfg.USE_AMP and device.type == "cuda") else nullcontext()
            with ctx:
                x0_pred = denoise_preconditioned(model, x_noisy, sigma, nu_cond, diff_sched)
                w = diff_sched.loss_weight(sigma).view(-1, 1, 1, 1)
                mse = (w * (x0_pred - x) ** 2).mean()
                if use_physics:
                    L_ens    = loss_enstrophy(x0_pred, x)
                    L_spec   = loss_spectral (x0_pred, x)
                    L_struct = loss_structure(x0_pred, x, r_max=cfg.STRUCT_R_MAX)
                    loss = (mse + cfg.LAMBDA_ENSTROPHY * L_ens
                                + cfg.LAMBDA_SPECTRAL  * L_spec
                                + cfg.LAMBDA_STRUCT    * L_struct)
                    phys = (L_ens + L_spec + L_struct).item()
                    low = sigma < cfg.FLUX_SIGMA_MAX
                    if low.any():
                        L_intlen = loss_integral_length(x0_pred[low], x[low])
                        loss = loss + cfg.LAMBDA_INTLEN * L_intlen
                        phys = phys + L_intlen.item()
                        if cfg.LAMBDA_LOWK > 0:
                            L_lowk = loss_lowk_spectrum(x0_pred[low], x[low], cfg.LOWK_KCUT)
                            loss = loss + cfg.LAMBDA_LOWK * L_lowk
                            phys = phys + L_lowk.item()
                        if use_flux:
                            L_flx = loss_flux(x0_pred[low], x[low])
                            loss = loss + cfg.LAMBDA_FLUX * L_flx
                            phys = phys + L_flx.item()
                else:
                    loss = mse; phys = 0.0
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim); scaler.update()
            ema.update(model)
            ep_loss += loss.item(); ep_mse += mse.item(); ep_phys += phys; nb += 1

        model.eval()
        v_mse, v_n = 0.0, 0
        with torch.no_grad():
            for x, re in val_loader:
                x, re = x.to(device), re.to(device); B = x.shape[0]
                nu_cond = nu_lut[re]
                sigma = diff_sched.sample_sigma_train(B)
                noise = torch.randn_like(x) * sigma.view(-1, 1, 1, 1)
                x0_pred = denoise_preconditioned(model, x + noise, sigma, nu_cond, diff_sched)
                v_mse += ((x0_pred - x) ** 2).mean().item(); v_n += 1

        rec = {"epoch": epoch + 1, "train_loss": ep_loss / nb,
               "train_mse": ep_mse / nb, "train_phys": ep_phys / nb,
               "val_mse":   v_mse / max(v_n, 1), "lr": optim.param_groups[0]["lr"]}
        history.append(rec)
        print(f"[Train] ep {epoch+1:03d}/{cfg.EPOCHS}  loss={rec['train_loss']:.4f}  "
              f"val_mse={rec['val_mse']:.4f}  phys={rec['train_phys']:.4f}  "
              f"lr={rec['lr']:.2e}")
        sess_secs = time.time() - t0
        sess_mem  = (torch.cuda.max_memory_allocated() / 1e6) if device.type == "cuda" else 0.0
        compute = {
            "n_params":     n_params,
            "train_time_s": prior_compute.get("train_time_s", 0.0) + sess_secs,
            "peak_mem_mb":  max(prior_compute.get("peak_mem_mb", 0.0), sess_mem),
        }
        save_ckpt(cpath, model, optim, ema, epoch + 1, history, compute)
        lr_sched.step()

    sess_secs = time.time() - t0
    sess_mem  = (torch.cuda.max_memory_allocated() / 1e6) if device.type == "cuda" else 0.0
    if start_epoch >= cfg.EPOCHS and prior_compute:
        train_time = float(prior_compute.get("train_time_s", sess_secs))
        peak_mem   = float(prior_compute.get("peak_mem_mb",  sess_mem))
    else:
        train_time = prior_compute.get("train_time_s", 0.0) + sess_secs
        peak_mem   = max(prior_compute.get("peak_mem_mb",   0.0), sess_mem)
    print(f"[Train] done in {train_time:.1f}s  peak_mem={peak_mem:.1f} MB")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    ema.copy_to(model)
    return model, diff_sched, history, {"n_params": n_params, "train_time_s": train_time,
                                         "peak_mem_mb": peak_mem}


# ============================================================================
# DPM-Solver++ 2M Sampler
# ============================================================================

@torch.no_grad()
def sample_dpmpp_2m(model, sched, n, nu_cond, image_size, device,
                    n_steps=50, clip_x0=None, noise_seed=None):
    """Sample fields with the DPM-Solver++(2M) ODE solver on the EDM sigma schedule,
    conditioned on a continuous log-viscosity value nu_cond; supports optional x0 clipping.
    noise_seed: None -> stochastic latent noise (varies per run); an int -> deterministic,
    reproducible sampling (used by evaluation so the reported metrics are reproducible)."""
    model.eval()
    sigmas = sched.build_inference_schedule(n_steps)
    if noise_seed is None:
        x0 = torch.randn(n, 1, image_size, image_size, device=device)
    else:
        g = torch.Generator(device=device).manual_seed(int(noise_seed))
        x0 = torch.randn(n, 1, image_size, image_size, device=device, generator=g)
    x = x0 * sigmas[0]
    cond = torch.full((n,), float(nu_cond), device=device)
    old = None
    for i in range(n_steps):
        sigma      = sigmas[i]
        sigma_next = sigmas[i + 1]
        sigma_b = torch.full((n,), sigma.item(), device=device)
        denoised = denoise_preconditioned(model, x, sigma_b, cond, sched)
        if clip_x0 is not None:
            denoised = denoised.clamp(-clip_x0, clip_x0)
        if sigma_next == 0:
            x = denoised
        else:
            t      = -sigma.log()
            t_next = -sigma_next.log()
            h = t_next - t
            if old is None:
                x = (sigma_next / sigma) * x - (-h).expm1() * denoised
            else:
                h_prev = t - (-sigmas[i - 1].log())
                r = h_prev / h
                D = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old
                x = (sigma_next / sigma) * x - (-h).expm1() * D
            old = denoised
    return x


# ============================================================================
# Evaluation Metrics
# ============================================================================

def metric_lsd_aggregate(pred, true):
    """Log-spectral distance between the batch-mean dealiased vorticity spectra of prediction and truth."""
    Ep, _ = compute_dealiased_spectrum(pred.squeeze(1))
    Et, _ = compute_dealiased_spectrum(true.squeeze(1))
    Ep = Ep.mean(0).clamp(min=1e-12)
    Et = Et.mean(0).clamp(min=1e-12)
    return (torch.log10(Ep) - torch.log10(Et)).abs().mean().item()


def metric_lsd_per_snapshot_diag(pred_one, true_one):
    """Per-snapshot log-spectral distance between a single predicted and true field (diagnostic)."""
    Ep, _ = compute_dealiased_spectrum(pred_one.squeeze(1))
    Et, _ = compute_dealiased_spectrum(true_one.squeeze(1))
    Ep = Ep.squeeze(0).clamp(min=1e-12)
    Et = Et.squeeze(0).clamp(min=1e-12)
    return (torch.log10(Ep) - torch.log10(Et)).abs().mean().item()


def metric_vorticity_structure_log_rmse(pred, true, order):
    """Log-space RMSE between the order-`order` vorticity structure functions of prediction and truth."""
    Sp, _ = compute_vorticity_structure_function(pred, order)
    St, _ = compute_vorticity_structure_function(true, order)
    Sp_m = Sp.mean(0).clamp(min=1e-12)
    St_m = St.mean(0).clamp(min=1e-12)
    return ((torch.log10(Sp_m) - torch.log10(St_m)) ** 2).mean().sqrt().item()


def metric_integral_length_rel_err(pred, true):
    """Relative error of the mean integral length scale, prediction vs truth."""
    Lp = compute_integral_length(pred).mean()
    Lt = compute_integral_length(true).mean()
    return ((Lp - Lt).abs() / (Lt + 1e-8)).item()


def metric_energy_flux_rmse(pred, true):
    """Peak-normalized RMSE between the mean energy spectral fluxes of prediction and truth."""
    Pp, _ = compute_energy_flux(pred)
    Pt, _ = compute_energy_flux(true)
    Pp_m, Pt_m = Pp.mean(0), Pt.mean(0)
    norm = Pt_m.abs().max().clamp(min=1e-8)
    return ((Pp_m - Pt_m) ** 2).mean().sqrt().item() / norm.item()


def metric_enstrophy_flux_rmse(pred, true):
    """Peak-normalized RMSE between the mean enstrophy spectral fluxes of prediction and truth."""
    Pp, _ = compute_enstrophy_flux(pred)
    Pt, _ = compute_enstrophy_flux(true)
    Pp_m, Pt_m = Pp.mean(0), Pt.mean(0)
    norm = Pt_m.abs().max().clamp(min=1e-8)
    return ((Pp_m - Pt_m) ** 2).mean().sqrt().item() / norm.item()


def metric_inverse_energy_cascade_recovery(pred, true, kf=4):
    """Percent recovery of the inverse energy cascade: predicted vs true negative
    energy flux below the forcing wavenumber (clamped to [0, 200]%)."""
    Pp, _ = compute_energy_flux(pred)
    Pt, kt = compute_energy_flux(true)
    Pp_m, Pt_m = Pp.mean(0), Pt.mean(0)
    mask = (kt < kf) & (kt > 0)
    if mask.sum() == 0 or Pt_m[mask].clamp(max=0).abs().sum() < 1e-8:
        return 0.0
    inv_p = (-Pp_m[mask]).clamp(min=0).sum()
    inv_t = (-Pt_m[mask]).clamp(min=0).sum()
    return 100.0 * (inv_p / inv_t.clamp(min=1e-8)).clamp(min=0.0, max=2.0).item()


def metric_forward_enstrophy_cascade_recovery(pred, true, kf=4):
    """Percent recovery of the forward enstrophy cascade: predicted vs true positive
    enstrophy flux above the forcing wavenumber (clamped to [0, 200]%)."""
    Pp, _ = compute_enstrophy_flux(pred)
    Pt, kt = compute_enstrophy_flux(true)
    Pp_m, Pt_m = Pp.mean(0), Pt.mean(0)
    mask = (kt > kf)
    if mask.sum() == 0 or Pt_m[mask].clamp(min=0).sum() < 1e-8:
        return 0.0
    fwd_p = Pp_m[mask].clamp(min=0).sum()
    fwd_t = Pt_m[mask].clamp(min=0).sum()
    return 100.0 * (fwd_p / fwd_t.clamp(min=1e-8)).clamp(min=0.0, max=2.0).item()


def metric_crps_ensemble(ens, target):
    """Empirical CRPS of an ensemble forecast against the target (energy-form estimator),
    averaged over pixels."""
    M = ens.shape[0]
    e_flat = ens.reshape(M, -1).to(torch.float32)
    y_flat = target.reshape(-1).to(torch.float32)
    N = e_flat.shape[1]
    t1 = (e_flat - y_flat.unsqueeze(0)).abs().mean(dim=0)
    t2 = torch.zeros(N, device=ens.device, dtype=torch.float32)
    for i in range(M):
        t2 += (e_flat[i:i+1] - e_flat).abs().mean(dim=0)
    t2 *= 0.5 / M
    return float((t1 - t2).mean().item())


def metric_coverage_band(ens, target, lo_pct=5, hi_pct=95):
    """Fraction of target pixels falling within the ensemble's [lo_pct, hi_pct] percentile band."""
    lo = torch.quantile(ens, lo_pct / 100.0, dim=0)
    hi = torch.quantile(ens, hi_pct / 100.0, dim=0)
    return float(((target >= lo) & (target <= hi)).float().mean().item())


def calibrate_alpha(ens_cal, y_cal, target_cov=0.90, grid=None):
    """Grid-search the interval-scaling factor alpha on calibration data so the
    median +/- alpha*spread band attains the target coverage."""
    device = ens_cal.device
    if grid is None:
        grid = torch.linspace(0.1, 4.0, 79, device=device)
    elif not torch.is_tensor(grid):
        grid = torch.as_tensor(grid, device=device, dtype=torch.float32)
    lo  = torch.quantile(ens_cal,  0.05, dim=0)
    hi  = torch.quantile(ens_cal,  0.95, dim=0)
    med = torch.quantile(ens_cal,  0.50, dim=0)
    half_lo = (med - lo).clamp(min=0.0)
    half_hi = (hi - med).clamp(min=0.0)
    best_a, best_gap = 1.0, float("inf")
    for a in grid.tolist():
        cov = ((y_cal >= med - a * half_lo) & (y_cal <= med + a * half_hi)).float().mean().item()
        gap = abs(cov - target_cov)
        if gap < best_gap:
            best_a, best_gap = float(a), float(gap)
    return best_a


def coverage_calibrated(ens, target, alpha):
    """Empirical coverage of the alpha-scaled median +/- spread prediction band."""
    lo  = torch.quantile(ens, 0.05, dim=0)
    hi  = torch.quantile(ens, 0.95, dim=0)
    med = torch.quantile(ens, 0.50, dim=0)
    half_lo = (med - lo).clamp(min=0.0)
    half_hi = (hi - med).clamp(min=0.0)
    return float(((target >= med - alpha * half_lo) &
                  (target <= med + alpha * half_hi)).float().mean().item())


# ============================================================================
# Evaluation Routine
# ============================================================================

def evaluate(model, sched, test_ds, cfg, device, variant, seed):
    """Generate samples per Reynolds regime on the fixed test set, compute aggregate,
    per-sample, and per-regime physics metrics, and run ensemble UQ (CRPS plus raw and
    calibrated 90% coverage). Returns (aggregates, per_sample, per_regime, pred, true, re)."""
    print(f"[Eval] {variant}/seed{seed}")
    nu_lut = make_nu_cond_lut(cfg.NU_LIST, device)
    loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                        pin_memory=(device.type == "cuda"))
    all_pred, all_true, all_re = [], [], []
    with torch.no_grad():
        for x, re in loader:
            x  = x .to(device, non_blocking=True)
            re = re.to(device, non_blocking=True)
            for r in range(len(cfg.NU_LIST)):
                idx = (re == r)
                if idx.sum() == 0: continue
                n = int(idx.sum().item())
                gens = sample_dpmpp_2m(model, sched, n, float(nu_lut[r]), cfg.GRID, device,
                                       n_steps=cfg.N_SAMPLE_STEPS, noise_seed=seed * 1000 + r)
                all_pred.append(gens)
                all_true.append(x[idx])
                all_re.append(torch.full((n,), r, device=device))
            if sum(p.shape[0] for p in all_pred) >= cfg.N_EVAL_SAMPLES:
                break
    pred   = torch.cat(all_pred, dim=0)[:cfg.N_EVAL_SAMPLES]
    true   = torch.cat(all_true, dim=0)[:cfg.N_EVAL_SAMPLES]
    re_arr = torch.cat(all_re,   dim=0)[:cfg.N_EVAL_SAMPLES]

    per_sample = []
    for i in range(pred.shape[0]):
        per_sample.append({
            "variant": variant, "seed": seed, "re_idx": int(re_arr[i].item()),
            "mse_diag":                       float(F.mse_loss(pred[i], true[i]).item()),
            "lsd_per_snapshot_diag":          metric_lsd_per_snapshot_diag(pred[i:i+1], true[i:i+1]),
            "vorticity_S2_log_rmse_diag":     metric_vorticity_structure_log_rmse(pred[i:i+1], true[i:i+1], order=2),
            "vorticity_S3_log_rmse_diag":     metric_vorticity_structure_log_rmse(pred[i:i+1], true[i:i+1], order=3),
        })

    aggregates = {
        "lsd_aggregate":                metric_lsd_aggregate(pred, true),
        "vorticity_S2_log_rmse":        metric_vorticity_structure_log_rmse(pred, true, order=2),
        "vorticity_S3_log_rmse":        metric_vorticity_structure_log_rmse(pred, true, order=3),
        "integral_length_rel_err":      metric_integral_length_rel_err(pred, true),
        "energy_flux_rmse":             metric_energy_flux_rmse(pred, true),
        "enstrophy_flux_rmse":          metric_enstrophy_flux_rmse(pred, true),
        "inverse_energy_cascade_recovery_pct":    metric_inverse_energy_cascade_recovery(pred, true, kf=cfg.FORCING_K),
        "forward_enstrophy_cascade_recovery_pct": metric_forward_enstrophy_cascade_recovery(pred, true, kf=cfg.FORCING_K),
    }

    per_regime = []
    for r in range(len(cfg.NU_LIST)):
        mask = (re_arr == r)
        if mask.sum() < 8:
            continue
        pr = pred[mask]; tr = true[mask]
        per_regime.append({
            "variant": variant, "seed": seed, "re_idx": r, "nu": cfg.NU_LIST[r],
            "n_samples":                              int(mask.sum().item()),
            "lsd_aggregate":                          metric_lsd_aggregate(pr, tr),
            "vorticity_S2_log_rmse":                  metric_vorticity_structure_log_rmse(pr, tr, order=2),
            "vorticity_S3_log_rmse":                  metric_vorticity_structure_log_rmse(pr, tr, order=3),
            "integral_length_rel_err":                metric_integral_length_rel_err(pr, tr),
            "energy_flux_rmse":                       metric_energy_flux_rmse(pr, tr),
            "enstrophy_flux_rmse":                    metric_enstrophy_flux_rmse(pr, tr),
            "inverse_energy_cascade_recovery_pct":    metric_inverse_energy_cascade_recovery(pr, tr, kf=cfg.FORCING_K),
            "forward_enstrophy_cascade_recovery_pct": metric_forward_enstrophy_cascade_recovery(pr, tr, kf=cfg.FORCING_K),
        })

    print(f"[Eval] LSD_agg={aggregates['lsd_aggregate']:.4f}  "
          f"S2={aggregates['vorticity_S2_log_rmse']:.4f}  "
          f"EnergyFluxRMSE={aggregates['energy_flux_rmse']:.4f}  "
          f"EnstrophyFluxRMSE={aggregates['enstrophy_flux_rmse']:.4f}  "
          f"InvE={aggregates['inverse_energy_cascade_recovery_pct']:.1f}%  "
          f"FwdZ={aggregates['forward_enstrophy_cascade_recovery_pct']:.1f}%")

    n_uq = min(64, pred.shape[0])
    true_uq = true[:n_uq]
    re_uq   = re_arr[:n_uq]
    ens = torch.zeros(cfg.N_ENSEMBLE, n_uq, 1, cfg.GRID, cfg.GRID, device=device)
    with torch.no_grad():
        for m in range(cfg.N_ENSEMBLE):
            for r in range(len(cfg.NU_LIST)):
                idx = (re_uq == r)
                if idx.sum() == 0: continue
                gens = sample_dpmpp_2m(model, sched, int(idx.sum().item()), float(nu_lut[r]),
                                       cfg.GRID, device, n_steps=cfg.N_SAMPLE_STEPS,
                                       noise_seed=seed * 100000 + m * 1000 + r)
                ens[m][idx.nonzero(as_tuple=False).flatten()] = gens

    n_cal = max(2, int(cfg.UQ_CAL_FRACTION * n_uq))
    if n_uq - n_cal >= 2:
        perm = torch.randperm(n_uq, generator=torch.Generator().manual_seed(seed)).to(device)
        ens_p, y_p = ens[:, perm], true_uq[perm]
        ens_cal, ens_tst = ens_p[:, :n_cal], ens_p[:, n_cal:]
        y_cal,   y_tst   = y_p[:n_cal], y_p[n_cal:]
        alpha = calibrate_alpha(ens_cal, y_cal, target_cov=0.90)
        aggregates.update({
            "crps_test":              metric_crps_ensemble(ens_tst, y_tst),
            "coverage_90_raw_test":   metric_coverage_band(ens_tst, y_tst),
            "coverage_90_calibrated": coverage_calibrated(ens_tst, y_tst, alpha),
            "coverage_alpha":         alpha,
        })
        print(f"[Eval]  UQ: alpha={alpha:.3f}  raw_cov={aggregates['coverage_90_raw_test']:.3f}  "
              f"cal_cov={aggregates['coverage_90_calibrated']:.3f}  "
              f"CRPS={aggregates['crps_test']:.4f}")
    else:
        for k in ("crps_test", "coverage_90_raw_test", "coverage_90_calibrated", "coverage_alpha"):
            aggregates[k] = float("nan")

    del ens
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return aggregates, per_sample, per_regime, pred.cpu(), true.cpu(), re_arr.cpu()


# ============================================================================
# Seed-Level Statistical Analysis
# ============================================================================

def bootstrap_ci(values, n_resamples=2000, ci=0.95, seed=0):
    """Percentile bootstrap of the mean: return (mean, ci_lo, ci_hi) for a 1-D sample."""
    rng = np.random.default_rng(seed)
    vals = np.asarray(values, dtype=float)
    if len(vals) < 2:
        return float(vals.mean() if len(vals) else float("nan")), float("nan"), float("nan")
    samples = rng.choice(vals, size=(n_resamples, len(vals)), replace=True).mean(axis=1)
    lo = np.percentile(samples, 100 * (1 - ci) / 2)
    hi = np.percentile(samples, 100 * (1 + ci) / 2)
    return float(vals.mean()), float(lo), float(hi)


def seed_level_paired_tests(summary_df, target, baseline, metrics, cfg):
    """Seed-paired comparison of target vs baseline per metric: bootstrap CI of the
    difference, paired t-test, Wilcoxon signed-rank, and Cohen's d; returns a DataFrame."""
    rows = []
    for m in metrics:
        if m not in summary_df.columns:
            continue
        a_df = summary_df[summary_df["variant"] == target  ].set_index("seed")[m]
        b_df = summary_df[summary_df["variant"] == baseline].set_index("seed")[m]
        common = sorted(set(a_df.index) & set(b_df.index))
        if len(common) < 3:
            continue
        a = a_df.loc[common].values.astype(float)
        b = b_df.loc[common].values.astype(float)
        diff = a - b
        mean_d, lo_d, hi_d = bootstrap_ci(diff, cfg.BOOTSTRAP_RESAMPLES, cfg.BOOTSTRAP_CI, seed=12345)
        try:
            _, t_p = stats.ttest_rel(a, b)
        except Exception:
            t_p = float("nan")
        try:
            _, w_p = stats.wilcoxon(a, b)
        except ValueError:
            w_p = float("nan")
        cohen_d = float(diff.mean() / (diff.std(ddof=1) + 1e-12)) if len(diff) > 1 else float("nan")
        rows.append({
            "comparison":    f"{target}_vs_{baseline}",
            "metric":        m,
            "n_seeds":       int(len(common)),
            "target_mean":   float(a.mean()),
            "baseline_mean": float(b.mean()),
            "diff_mean":     mean_d,
            "diff_ci_lo":    lo_d,
            "diff_ci_hi":    hi_d,
            "t_p":           float(t_p),
            "wilcoxon_p":    float(w_p),
            "cohen_d":       cohen_d,
        })
    return pd.DataFrame(rows)


def aggregate_with_bootstrap_ci(summary_df, metrics, cfg):
    """Per-variant bootstrap mean and 95% CI for each metric across seeds; returns a DataFrame."""
    rows = []
    for variant, grp in summary_df.groupby("variant"):
        row = {"variant": variant, "n_seeds": int(len(grp))}
        for m in metrics:
            if m not in grp.columns:
                continue
            mean, lo, hi = bootstrap_ci(
                grp[m].dropna().values, cfg.BOOTSTRAP_RESAMPLES, cfg.BOOTSTRAP_CI,
                seed=int(hashlib.sha1(f"{variant}_{m}".encode()).hexdigest(), 16) & 0xFFFF)
            row[f"{m}_mean"]  = mean
            row[f"{m}_ci_lo"] = lo
            row[f"{m}_ci_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows).sort_values("variant").reset_index(drop=True)


# ============================================================================
# Visualization
# ============================================================================

VARIANT_COLORS = {
    "vanilla":      "#888888",
    "se":           "#7B7BC6",
    "fno":          "#5BAF6A",
    "fno_operator": "#2E7D43",
    "wrsg_gate":    "#F0A040",
    "wrsg_phys":    "#7050C0",
    "wrsg":         "#C9534D",
    "wrsg_fno":     "#1F77B4",
}
VARIANT_ORDER = ["vanilla", "se", "fno", "fno_operator",
                 "wrsg_gate", "wrsg_phys", "wrsg", "wrsg_fno"]
ALL_VARIANTS = tuple(TurbulenceDenoiser.VARIANTS) + ("fno_operator",)

VARIANT_LABELS = {
    "vanilla":      "Vanilla U-Net",
    "se":           "SE",
    "fno":          "FNO-bottleneck",
    "fno_operator": "FNO (standalone)",
    "wrsg_gate":    "WRSG-gate",
    "wrsg_phys":    "WRSG-phys",
    "wrsg":         "WRSG",
    "wrsg_fno":     "WRSG+FNO",
}
def vlabel(v):
    """Paper-ready display label for a variant key (falls back to the key itself)."""
    return VARIANT_LABELS.get(v, v)


def plot_training_curves(all_histories, cfg):
    """Plot mean +/- std training-loss and validation-MSE curves per variant across seeds."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    by_variant = {}
    for (variant, _), hist in all_histories.items():
        by_variant.setdefault(variant, []).append(pd.DataFrame(hist))
    for variant, dfs in by_variant.items():
        if variant not in VARIANT_COLORS:
            continue
        min_e = min(len(d) for d in dfs)
        epochs = dfs[0]["epoch"].values[:min_e]
        tl = np.stack([d["train_loss"].values[:min_e] for d in dfs], axis=0)
        vm = np.stack([d["val_mse"   ].values[:min_e] for d in dfs], axis=0)
        for ax, arr in [(axes[0], tl), (axes[1], vm)]:
            m  = arr.mean(0); sd = arr.std(0)
            color = VARIANT_COLORS.get(variant, "#999")
            ax.plot(epochs, m, color=color, lw=1.8, label=vlabel(variant))
            ax.fill_between(epochs, m - sd, m + sd, color=color, alpha=0.18)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Train loss")
    axes[0].set_title(f"Training loss (mean +/- std over {len(cfg.SEEDS)} seeds)")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Val MSE")
    axes[1].set_title(f"Validation MSE (mean +/- std over {len(cfg.SEEDS)} seeds)")
    axes[0].legend(fontsize=9); axes[1].legend(fontsize=9)
    fig.tight_layout(); fig.savefig(cfg.FIG_DIR / "training_curves.png", dpi=140, bbox_inches="tight"); plt.close(fig)


def plot_dealiased_spectra(samples_per_variant, true_samples, cfg):
    """Plot generated vs DNS dealiased vorticity power spectra, with the inertial range shaded."""
    fig, ax = plt.subplots(figsize=(7, 5))
    Et, kvals = compute_dealiased_spectrum(true_samples.squeeze(1))
    k = kvals.cpu().numpy()
    ax.loglog(k[1:], Et.mean(0).cpu().numpy()[1:], "k-", lw=2.6, label="DNS (dealiased)")
    for v in VARIANT_ORDER:
        if v not in samples_per_variant: continue
        Ep, _ = compute_dealiased_spectrum(samples_per_variant[v].squeeze(1))
        ax.loglog(k[1:], Ep.mean(0).cpu().numpy()[1:], "--", lw=1.6,
                  color=VARIANT_COLORS[v], alpha=0.9, label=vlabel(v))
    ax.axvspan(cfg.FORCING_K, max(cfg.GRID // 4, cfg.FORCING_K + 4), alpha=0.15,
               color="orange", label="inertial range")
    ax.set_xlabel("k"); ax.set_ylabel(r"$|\hat{\omega}(k)|^2$")
    ax.set_title("Dealiased Vorticity Power Spectra: Generated vs DNS")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(cfg.FIG_DIR / "spectra_dealiased.png", dpi=140, bbox_inches="tight"); plt.close(fig)


def plot_structure_functions(samples_per_variant, true_samples, cfg):
    """Plot generated vs DNS 2nd- and 3rd-order vorticity structure functions."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    S2_t, rs = compute_vorticity_structure_function(true_samples, order=2)
    S3_t, _  = compute_vorticity_structure_function(true_samples, order=3)
    r_np = rs.cpu().numpy()
    axes[0].loglog(r_np, S2_t.mean(0).cpu().numpy(), "k-", lw=2.6, label="DNS")
    axes[1].loglog(r_np, S3_t.mean(0).cpu().numpy(), "k-", lw=2.6, label="DNS")
    for v in VARIANT_ORDER:
        if v not in samples_per_variant: continue
        s = samples_per_variant[v]
        S2, _ = compute_vorticity_structure_function(s, order=2)
        S3, _ = compute_vorticity_structure_function(s, order=3)
        axes[0].loglog(r_np, S2.mean(0).cpu().numpy(), "--", lw=1.5,
                       color=VARIANT_COLORS[v], alpha=0.9, label=vlabel(v))
        axes[1].loglog(r_np, S3.mean(0).cpu().numpy(), "--", lw=1.5,
                       color=VARIANT_COLORS[v], alpha=0.9, label=vlabel(v))
    axes[0].set_xlabel("r"); axes[0].set_ylabel(r"$S_2^{\omega}(r)$")
    axes[0].set_title("Vorticity 2nd-order structure function")
    axes[1].set_xlabel("r"); axes[1].set_ylabel(r"$|S_3^{\omega}(r)|$")
    axes[1].set_title("Vorticity 3rd-order structure function")
    axes[0].legend(fontsize=9); axes[1].legend(fontsize=9)
    fig.tight_layout(); fig.savefig(cfg.FIG_DIR / "structure_functions.png", dpi=140, bbox_inches="tight"); plt.close(fig)


FLUX_SHOW_VARIANTS = ["vanilla", "fno", "wrsg", "wrsg_fno"]


def plot_flux_panels(samples_by_variant_regime, true_by_regime, cfg):
    """Plot normalized energy (top) and enstrophy (bottom) spectral fluxes, generated vs
    DNS, for three representative regimes and a curated variant set, with one shared legend."""
    all_re = sorted(true_by_regime.keys())
    if len(all_re) >= 3:
        show_re = [all_re[0], all_re[len(all_re) // 2], all_re[-1]]
    else:
        show_re = all_re
    show_vars = [v for v in FLUX_SHOW_VARIANTS if v in samples_by_variant_regime]
    nre = len(show_re)
    fig, axes = plt.subplots(2, nre, figsize=(4.8 * nre, 7.6), squeeze=False)
    for row, (flux_fn, label, sym) in enumerate([(compute_energy_flux,    "Energy",    "E"),
                                                 (compute_enstrophy_flux, "Enstrophy", "Z")]):
        for c, r_idx in enumerate(show_re):
            ax = axes[row, c]
            true = true_by_regime.get(r_idx)
            if true is None or len(true) == 0:
                continue
            Pt, k = flux_fn(true)
            Pt_m = Pt.mean(0)
            norm = Pt_m.abs().max().clamp(min=1e-8)
            ax.semilogx(k.cpu().numpy(), (Pt_m / norm).cpu().numpy(), "k-", lw=2.8, label="DNS")
            for v in show_vars:
                sr = samples_by_variant_regime[v].get(r_idx)
                if sr is None or len(sr) == 0:
                    continue
                Pp, _ = flux_fn(sr)
                ax.semilogx(k.cpu().numpy(), (Pp.mean(0) / norm).cpu().numpy(),
                            "--", lw=2.0, color=VARIANT_COLORS[v], alpha=0.95, label=vlabel(v))
            ax.axvline(cfg.FORCING_K, color="orange", ls=":", lw=1.2)
            ax.axhline(0, color="gray", lw=0.7)
            ax.tick_params(labelsize=11)
            if row == 1:
                ax.set_xlabel("$k$", fontsize=13)
            if c == 0:
                ax.set_ylabel(rf"$\Pi_{{{sym}}}(k)\,/\,|\Pi_{{DNS}}|_{{\max}}$", fontsize=13)
            ax.set_title(rf"{label} flux,  $\nu={cfg.NU_LIST[r_idx]}$", fontsize=12)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(handles), fontsize=12,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Normalized energy (top) and enstrophy (bottom) spectral fluxes "
                 "($k_f=4$ dotted)", y=1.0, fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(cfg.FIG_DIR / "fluxes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


METRIC_TITLES = {
    "lsd_aggregate":                          "LSD aggregate $\\downarrow$",
    "vorticity_S2_log_rmse":                  "Vort. $S_2$ log-RMSE $\\downarrow$",
    "vorticity_S3_log_rmse":                  "Vort. $S_3$ log-RMSE $\\downarrow$",
    "integral_length_rel_err":                "Integral length rel. err. $\\downarrow$",
    "energy_flux_rmse":                       "Energy flux RMSE $\\downarrow$",
    "enstrophy_flux_rmse":                    "Enstrophy flux RMSE $\\downarrow$",
    "inverse_energy_cascade_recovery_pct":    "Inv. energy cascade rec. % $\\uparrow$",
    "forward_enstrophy_cascade_recovery_pct": "Fwd. enstrophy cascade rec. % $\\uparrow$",
}


def plot_headline_bars_with_ci(agg_ci_df, cfg):
    """Bar chart of per-variant headline metrics with 95% bootstrap confidence intervals.
    One shared legend keys colour to variant, so the per-panel x-axis carries no repeated
    labels; metric titles state the direction of improvement."""
    metrics = ["lsd_aggregate", "vorticity_S2_log_rmse", "vorticity_S3_log_rmse",
               "integral_length_rel_err", "energy_flux_rmse", "enstrophy_flux_rmse",
               "inverse_energy_cascade_recovery_pct", "forward_enstrophy_cascade_recovery_pct"]
    metrics = [m for m in metrics if f"{m}_mean" in agg_ci_df.columns]
    n = len(metrics)
    ncol = (n + 1) // 2
    fig, axes = plt.subplots(2, ncol, figsize=(3.3 * ncol, 7.0))
    axes = axes.flatten()
    df = agg_ci_df.set_index("variant").reindex(
        [v for v in VARIANT_ORDER if v in agg_ci_df["variant"].values])
    variants = list(df.index)
    colors = [VARIANT_COLORS.get(v, "#999") for v in variants]
    xpos = np.arange(len(variants))
    for i, m in enumerate(metrics):
        means = df[f"{m}_mean"].values
        err_lo = np.maximum(means - df[f"{m}_ci_lo"].values, 0)
        err_hi = np.maximum(df[f"{m}_ci_hi"].values - means, 0)
        axes[i].bar(xpos, means, yerr=[err_lo, err_hi], capsize=3, color=colors)
        axes[i].set_title(METRIC_TITLES.get(m, m), fontsize=12)
        axes[i].set_xticks([])
        axes[i].tick_params(axis="y", labelsize=11)
        axes[i].margins(x=0.02)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=VARIANT_COLORS.get(v, "#999"))
               for v in variants]
    fig.legend(handles, [vlabel(v) for v in variants], loc="lower center",
               ncol=len(variants), fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Headline metrics: per-variant mean with 95% bootstrap CI "
                 f"(n={len(cfg.SEEDS)} seeds)", y=1.0, fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(cfg.FIG_DIR / "headline_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pareto(summary_df, compute_df, agg_ci_df, cfg):
    """Scatter the parameter-count vs performance Pareto frontier per variant, with CIs."""
    metrics = ["lsd_aggregate", "vorticity_S2_log_rmse", "energy_flux_rmse"]
    merged = summary_df.merge(compute_df[["variant", "seed", "n_params"]],
                              on=["variant", "seed"])
    params_per_v = merged.groupby("variant")["n_params"].mean()
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 4.0))
    for i, m in enumerate(metrics):
        ax = axes[i]
        for variant in [v for v in VARIANT_ORDER if v in agg_ci_df["variant"].values]:
            row = agg_ci_df[agg_ci_df["variant"] == variant].iloc[0]
            if f"{m}_mean" not in row: continue
            mean = row[f"{m}_mean"]
            lo, hi = row[f"{m}_ci_lo"], row[f"{m}_ci_hi"]
            err_lo = max(mean - lo, 0); err_hi = max(hi - mean, 0)
            ax.errorbar(params_per_v[variant] / 1e6, mean,
                        yerr=[[err_lo], [err_hi]], fmt="o", markersize=10, capsize=5,
                        color=VARIANT_COLORS.get(variant, "#999"), label=vlabel(variant))
        ax.set_xlabel("Parameters (M)"); ax.set_ylabel(m)
        ax.set_title(m, fontsize=10)
        if i == 0:
            ax.legend(fontsize=9)
    fig.suptitle("Parameter / Performance Pareto Frontier (95% bootstrap CIs)", y=1.02)
    fig.tight_layout(); fig.savefig(cfg.FIG_DIR / "pareto.png", dpi=140, bbox_inches="tight"); plt.close(fig)


def plot_uq_panel(summary_df, cfg):
    """Bar chart of CRPS and raw/calibrated 90% coverage per variant against the 0.90 target."""
    cols = [c for c in ["crps_test", "coverage_90_raw_test",
                        "coverage_90_calibrated", "coverage_alpha"]
            if c in summary_df.columns]
    if not cols:
        return
    agg = summary_df.groupby("variant")[cols].agg(["mean", "std"])
    variants = [v for v in VARIANT_ORDER if v in agg.index]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(variants)); w = 0.27
    crps = agg.loc[variants, ("crps_test",              "mean")].values
    rawc = agg.loc[variants, ("coverage_90_raw_test",   "mean")].values
    calc = agg.loc[variants, ("coverage_90_calibrated", "mean")].values
    crps_e = agg.loc[variants, ("crps_test",              "std")].fillna(0).values
    rawc_e = agg.loc[variants, ("coverage_90_raw_test",   "std")].fillna(0).values
    calc_e = agg.loc[variants, ("coverage_90_calibrated", "std")].fillna(0).values
    ax.bar(x - w, crps, w, yerr=crps_e, capsize=4, label="CRPS (lower better)", color="#C9534D")
    ax.bar(x,     rawc, w, yerr=rawc_e, capsize=4, label="Coverage@90% (raw)",  color="#5BAF6A")
    ax.bar(x + w, calc, w, yerr=calc_e, capsize=4, label="Coverage@90% (calibrated)", color="#2E7D43")
    ax.axhline(0.90, color="k", ls="--", lw=0.8, label="target = 0.90")
    ax.set_xticks(x); ax.set_xticklabels([vlabel(v) for v in variants], rotation=15)
    ax.set_title(f"Uncertainty Quantification (M={cfg.N_ENSEMBLE} ensemble, 50/50 cal/test)")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(cfg.FIG_DIR / "uq_panel.png", dpi=140, bbox_inches="tight"); plt.close(fig)


def plot_vorticity_panel(paired_samples_per_variant, true_sample, cfg, nu=None):
    """Plot paired vorticity samples (shared latent noise) across variants beside the
    DNS reference, each with a zoom inset. `nu` (the viscosity of the single regime the
    paired samples are drawn at) is shown in the title so the figure is self-documenting."""
    H = true_sample.shape[-1]
    crop = H // 4
    cx, cy = H // 2, H // 2
    variants = [v for v in VARIANT_ORDER if v in paired_samples_per_variant]
    n_cols = 1 + len(variants)
    fig, axes = plt.subplots(2, n_cols, figsize=(2.4 * n_cols, 5.0))
    vmax = float(true_sample.abs().max().item())
    titles = ["DNS reference"] + [f"{vlabel(v)}\n(shared noise)" for v in variants]
    fields_top = [true_sample] + [paired_samples_per_variant[v] for v in variants]
    for col, (t, f) in enumerate(zip(titles, fields_top)):
        ax = axes[0, col]
        ax.imshow(f[0].cpu().numpy() if f.ndim == 3 else f[0, 0].cpu().numpy(),
                  cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(t, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        rect = plt.Rectangle((cy - crop // 2, cx - crop // 2), crop, crop,
                             linewidth=1.2, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)
    for col, (t, f) in enumerate(zip(titles, fields_top)):
        ax = axes[1, col]
        arr = f[0].cpu().numpy() if f.ndim == 3 else f[0, 0].cpu().numpy()
        zoom = arr[cx - crop // 2:cx + crop // 2, cy - crop // 2:cy + crop // 2]
        ax.imshow(zoom, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title("zoom", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    title = "Paired vorticity samples (shared latent noise across variants)"
    if nu is not None:
        title += f"  --  single regime, $\\nu$ = {nu:g}"
    fig.suptitle(title, y=1.02)
    fig.tight_layout(); fig.savefig(cfg.FIG_DIR / "vorticity_samples.png", dpi=140, bbox_inches="tight"); plt.close(fig)


# ============================================================================
# Main
# ============================================================================

HEADLINE_PHYSICS = ["lsd_aggregate", "vorticity_S2_log_rmse", "vorticity_S3_log_rmse",
                    "integral_length_rel_err", "energy_flux_rmse", "enstrophy_flux_rmse",
                    "inverse_energy_cascade_recovery_pct",
                    "forward_enstrophy_cascade_recovery_pct"]
HEADLINE_UQ      = ["crps_test", "coverage_90_raw_test",
                    "coverage_90_calibrated", "coverage_alpha"]
ALL_HEADLINE     = HEADLINE_PHYSICS + HEADLINE_UQ
def run_gate_sensitivity(cfg, train_pool, test_set, device):
    """Sweep the WRSG gate's radial-bin count and rank on a seed subset, training and
    evaluating wrsg per configuration; writes a CSV of LSD mean/std vs config."""
    configs = [(b, 2) for b in cfg.GATE_SWEEP_BINS] + \
              [(16, k) for k in cfg.GATE_SWEEP_RANKS if k != 2]
    base_ckpt, base_seeds = cfg.CKPT_DIR, cfg.SEEDS
    base_bins, base_rank  = cfg.GATE_N_BINS, cfg.GATE_RANK
    cfg.SEEDS = cfg.SWEEP_SEEDS
    rows = []
    print(f"[Sweep] gate sensitivity over {len(configs)} configs x "
          f"{len(cfg.SWEEP_SEEDS)} seeds (variant=wrsg).")
    for (bins, rank) in configs:
        cfg.GATE_N_BINS, cfg.GATE_RANK = bins, rank
        cfg.CKPT_DIR = base_ckpt / f"gate_sweep_b{bins}_k{rank}"
        cfg.CKPT_DIR.mkdir(parents=True, exist_ok=True)
        lsds = []
        for seed in cfg.SEEDS:
            model, sched, _, _ = train_one_model(
                "wrsg", seed, cfg, train_pool, device,
                use_physics=True, use_flux=True)
            agg = evaluate(model, sched, test_set, cfg, device, "wrsg", seed)[0]
            lsds.append(agg["lsd_aggregate"])
        rows.append({"gate_bins": bins, "gate_rank": rank, "n_seeds": len(lsds),
                     "lsd_mean": float(np.mean(lsds)), "lsd_std": float(np.std(lsds))})
        print(f"[Sweep] bins={bins:2d} rank={rank}: "
              f"LSD {np.mean(lsds):.4f} +/- {np.std(lsds):.4f}")
    cfg.CKPT_DIR, cfg.SEEDS = base_ckpt, base_seeds
    cfg.GATE_N_BINS, cfg.GATE_RANK = base_bins, base_rank
    df = pd.DataFrame(rows)
    cfg.CSV_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.CSV_DIR / "gate_sensitivity.csv", index=False)
    print("[Sweep] wrote gate_sensitivity.csv")
    return df


PHYSICS_VARIANTS = {"wrsg_phys", "wrsg", "wrsg_fno"}
FLUX_VARIANTS    = {"wrsg", "wrsg_fno"}


def parse_args():
    """Parse command-line arguments: smoke test, variant/seed selection, sharding,
    the claim queue, the gate sweep, and the aggregate-only / data-only modes."""
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="~2 minute end-to-end smoke test")
    p.add_argument("--variants", nargs="+", default=list(ALL_VARIANTS))
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--grid", type=int, default=None)
    p.add_argument("--snapshots", type=int, default=None)
    p.add_argument("--sample_steps", type=int, default=None)
    p.add_argument("--sweep_gate", action="store_true",
                   help="run the gate bin/rank sensitivity sweep, then exit")
    p.add_argument("--shard_index", type=int, default=0,
                   help="index of this worker in a multi-process run")
    p.add_argument("--shard_count", type=int, default=1,
                   help="total number of workers; each handles jobs[index::count]")
    p.add_argument("--claim_queue", action="store_true",
                   help="pull-based dynamic scheduler: each worker atomically "
                        "claims jobs via lock files, so fast workers take more "
                        "work and no GPU idles at the tail. Enables many workers "
                        "per GPU. Ignores --shard_index/--shard_count.")
    p.add_argument("--worker_tag", type=str, default=None,
                   help="label for this worker in claim-queue logs (e.g. gpu3.1)")
    p.add_argument("--aggregate_only", action="store_true",
                   help="build tables and figures from cached per-job files, then exit")
    p.add_argument("--prepare_data_only", action="store_true",
                   help="generate and cache the dataset (and DNS validation), then exit")
    return p.parse_args()


def apply_cli_overrides(cfg, args):
    """Apply CLI overrides to the config, including the fast end-to-end smoke-test preset."""
    if args.smoke:
        cfg.GRID = 32
        cfg.EPOCHS = 1; cfg.SEEDS = [29, 47]
        cfg.SNAPSHOTS_PER_REGIME = 80
        cfg.SPINUP_STEPS = 200; cfg.SNAPSHOT_INTERVAL = 20
        cfg.BATCH_SIZE = 8
        cfg.N_EVAL_SAMPLES = 24; cfg.N_ENSEMBLE = 4; cfg.N_SAMPLE_STEPS = 20
        cfg.BOOTSTRAP_RESAMPLES = 200
        cfg.USE_DEALIASED_JACOBIAN = False
    if args.grid is not None:         cfg.GRID = args.grid
    if args.seeds is not None:        cfg.SEEDS = args.seeds
    if args.epochs is not None:       cfg.EPOCHS = args.epochs
    if args.snapshots is not None:    cfg.SNAPSHOTS_PER_REGIME = args.snapshots
    if args.sample_steps is not None: cfg.N_SAMPLE_STEPS = args.sample_steps


_VARIANT_COST_RANK = {"wrsg_fno": 0, "fno_operator": 1, "wrsg": 2, "wrsg_gate": 3,
                      "fno": 4, "wrsg_phys": 5, "se": 6, "vanilla": 7}


def run_claim_queue(jobs, jobs_dir, args, cfg, train_pool, test_set, device):
    """Pull-based dynamic scheduler.

    Every worker walks the (cost-sorted) job list and atomically claims each job
    by creating an exclusive lock file. Faster workers naturally take more jobs,
    so no GPU sits idle while one straggler finishes the tail -- the failure mode
    of static ``jobs[i::N]`` striping. Safe to launch many workers per GPU.

    Claims live in ``jobs_dir/.claims`` and are cleared by the launcher's
    single-process prepare step before each fresh sweep.
    """
    tag = args.worker_tag or str(os.getpid())
    claims = jobs_dir / ".claims"
    claims.mkdir(parents=True, exist_ok=True)
    n = len(jobs)
    start = (hash(tag) % n) if n else 0
    rotated = jobs[start:] + jobs[:start]

    done = []
    for (variant, seed) in rotated:
        lock = claims / f"{variant}_seed{seed}.lock"
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        os.write(fd, f"{tag}\n".encode())
        os.close(fd)
        print(f"[Queue:{tag}] claim {variant} seed{seed} "
              f"(#{len(done) + 1} for this worker)", flush=True)
        res = run_single_job(variant, seed, cfg, train_pool, test_set, device)
        torch.save(res, jobs_dir / f"{variant}_seed{seed}.pt")
        done.append((variant, seed))
    return done


def run_single_job(variant, seed, cfg, train_pool, test_set, device):
    """Train and evaluate one (variant, seed) job, returning its metrics, history, and
    compute stats, plus (for the first seed only) sample fields cached for plotting."""
    use_phys = variant in PHYSICS_VARIANTS
    use_flx  = variant in FLUX_VARIANTS
    model, sched, history, compute = train_one_model(
        variant, seed, cfg, train_pool, device,
        use_physics=use_phys, use_flux=use_flx)
    agg, per_samp, per_reg, gen, true, re_eval = evaluate(
        model, sched, test_set, cfg, device, variant, seed)
    result = {"variant": variant, "seed": seed, "agg": agg,
              "per_samp": per_samp, "per_reg": per_reg,
              "compute": {**compute, "n_epochs": cfg.EPOCHS},
              "history": history}
    if seed == cfg.SEEDS[0]:
        result["gen64"]  = gen[:64].cpu()
        result["true64"] = true[:64].cpu()
        gen_by_regime, true_by_regime = {}, {}
        for r in range(len(cfg.NU_LIST)):
            mask = (re_eval == r)
            if mask.sum() > 0:
                gen_by_regime[r]  = gen[mask].cpu()
                true_by_regime[r] = true[mask].cpu()
        result["gen_by_regime"]  = gen_by_regime
        result["true_by_regime"] = true_by_regime
        r_target = int(re_eval[0].item()) if len(re_eval) > 0 else 0
        nu_lut = make_nu_cond_lut(cfg.NU_LIST, device)
        paired = sample_dpmpp_2m(model, sched, 1, float(nu_lut[r_target]), cfg.GRID, device,
                                 n_steps=cfg.N_SAMPLE_STEPS, noise_seed=20260521)
        result["paired"] = paired[0].cpu()
        result["paired_regime"] = r_target
        result["paired_nu"] = float(cfg.NU_LIST[r_target])
        rt = (re_eval == r_target).nonzero(as_tuple=False).flatten()
        result["paired_true"] = true[rt[0]].cpu() if len(rt) > 0 else None
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def assemble_results(results, variants, cfg):
    """Collate per-job results into flat records (summary, per-sample, per-regime,
    compute) plus the sample collections used for the figures."""
    summary_records, per_sample_all, per_regime_all, compute_records = [], [], [], []
    all_histories, last_samples_per_variant, last_true_samples = {}, {}, None
    samples_by_variant_regime = {v: {} for v in variants}
    true_by_regime, paired_samples_per_variant, paired_true_sample = {}, {}, None
    paired_nu = None
    for r in results:
        v, s = r["variant"], r["seed"]
        summary_records.append({"variant": v, "seed": s, **r["agg"]})
        per_sample_all.extend(r["per_samp"])
        per_regime_all.extend(r["per_reg"])
        compute_records.append({"variant": v, "seed": s, **r["compute"]})
        all_histories[(v, s)] = r["history"]
        if "gen64" in r:
            last_samples_per_variant[v] = r["gen64"]
            last_true_samples = r["true64"]
            samples_by_variant_regime[v] = r.get("gen_by_regime", {})
            for rk, tv in r.get("true_by_regime", {}).items():
                true_by_regime.setdefault(rk, tv)
            paired_samples_per_variant[v] = r["paired"]
            if paired_true_sample is None and r.get("paired_true") is not None:
                paired_true_sample = r["paired_true"]
                paired_nu = r.get("paired_nu")
    if paired_nu is None and paired_true_sample is not None:
        for rk, tv in true_by_regime.items():
            if tv is not None and len(tv) > 0 and torch.equal(tv[0], paired_true_sample):
                paired_nu = float(cfg.NU_LIST[rk]); break
    return {"summary_records": summary_records, "per_sample_all": per_sample_all,
            "per_regime_all": per_regime_all, "compute_records": compute_records,
            "all_histories": all_histories,
            "last_samples_per_variant": last_samples_per_variant,
            "last_true_samples": last_true_samples,
            "samples_by_variant_regime": samples_by_variant_regime,
            "true_by_regime": true_by_regime,
            "paired_samples_per_variant": paired_samples_per_variant,
            "paired_true_sample": paired_true_sample,
            "paired_nu": paired_nu}


def finalize_outputs(a, cfg):
    """Persist all CSV tables, run the seed-level statistics, render every figure, and
    print the headline and aggregate summary report."""
    print("\n[Phase 3] Persisting results and seed-level statistics.")
    summary_df    = pd.DataFrame(a["summary_records"])
    per_sample_df = pd.DataFrame(a["per_sample_all"])
    per_regime_df = pd.DataFrame(a["per_regime_all"])
    compute_df    = pd.DataFrame(a["compute_records"])
    summary_df   .to_csv(cfg.CSV_DIR / "summary_metrics.csv",        index=False)
    per_sample_df.to_csv(cfg.CSV_DIR / "per_sample_diagnostics.csv", index=False)
    per_regime_df.to_csv(cfg.CSV_DIR / "per_regime_metrics.csv",     index=False)
    compute_df   .to_csv(cfg.CSV_DIR / "compute_profile.csv",        index=False)

    agg_ci_df = aggregate_with_bootstrap_ci(summary_df, ALL_HEADLINE, cfg)
    agg_ci_df.to_csv(cfg.CSV_DIR / "headline_table_with_ci.csv", index=False)
    print("\n[Headline] Per-variant means with 95% bootstrap CI over seeds:")
    print(agg_ci_df.round(4).to_string(index=False))

    print("\n[Stats] Seed-level paired tests (each main model vs the others):")
    present = set(summary_df["variant"].unique())
    for target in [t for t in ["wrsg", "wrsg_fno"] if t in present]:
        for baseline in [b for b in VARIANT_ORDER if b in present and b != target]:
            tdf = seed_level_paired_tests(summary_df, target, baseline, HEADLINE_PHYSICS, cfg)
            tdf.to_csv(cfg.CSV_DIR / f"seed_level_tests_{target}_vs_{baseline}.csv", index=False)
            if len(tdf):
                print(f"\n  {target} vs {baseline}:")
                print(tdf[["metric", "n_seeds", "target_mean", "baseline_mean",
                           "diff_mean", "diff_ci_lo", "diff_ci_hi", "wilcoxon_p",
                           "cohen_d"]].round(4).to_string(index=False))

    print("\n[Phase 4] Visualization.")
    plot_training_curves(a["all_histories"], cfg)
    if a["last_true_samples"] is not None and a["last_samples_per_variant"]:
        plot_dealiased_spectra(a["last_samples_per_variant"], a["last_true_samples"], cfg)
        plot_structure_functions(a["last_samples_per_variant"], a["last_true_samples"], cfg)
        if a["true_by_regime"]:
            plot_flux_panels(a["samples_by_variant_regime"], a["true_by_regime"], cfg)
    if a["paired_true_sample"] is not None and a["paired_samples_per_variant"]:
        plot_vorticity_panel(a["paired_samples_per_variant"], a["paired_true_sample"], cfg,
                             nu=a.get("paired_nu"))
    plot_headline_bars_with_ci(agg_ci_df, cfg)
    plot_pareto(summary_df, compute_df, agg_ci_df, cfg)
    plot_uq_panel(summary_df, cfg)

    print("\n[Phase 5] Aggregate report (seed means):")
    print(summary_df.groupby("variant").mean(numeric_only=True).round(4).to_string())
    print(f"\n[Done] All artifacts under {cfg.RESULTS_DIR}")


def main():
    """Entry point: build/cache the dataset, hold out the fixed test set, then train and
    evaluate jobs (single-process, sharded, or claim-queue) and aggregate the outputs."""
    args = parse_args()
    Config.setup()
    apply_cli_overrides(Config, args)

    print("=" * 80)
    print(" Wavenumber-Resolved Spectral Diffusion (WRSG) for 2D Turbulence")
    print(f" Grid={Config.GRID}^2  Regimes (nu)={Config.NU_LIST}  Seeds={Config.SEEDS}")
    print(f" Epochs={Config.EPOCHS}  Variants={args.variants}")
    print("=" * 80)

    device = configure_device()
    jobs_dir = Config.RESULTS_DIR / "_jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        files = sorted(jobs_dir.glob("*.pt"))
        if not files:
            print(f"[Aggregate] No job files in {jobs_dir}; nothing to do.")
            return
        in_scope, skipped = [], 0
        for f in files:
            r = torch.load(f, map_location="cpu")
            if r["variant"] in VARIANT_ORDER and r["seed"] in Config.SEEDS:
                in_scope.append(r)
            else:
                skipped += 1
        if not in_scope:
            print(f"[Aggregate] No in-scope job files in {jobs_dir}; nothing to do.")
            return
        present = {r["variant"] for r in in_scope}
        variants = [v for v in VARIANT_ORDER if v in present]
        print(f"[Aggregate] Loaded {len(in_scope)} in-scope job files "
              f"({skipped} stale skipped); variants={variants}.")
        finalize_outputs(assemble_results(in_scope, variants, Config), Config)
        return

    print("\n[Phase 1] Dataset generation, DNS validation, test holdout.")
    payload = generate_dataset(Config, device)
    full_ds = TurbulenceDataset(payload["fields"], payload["re_labels"])
    train_indices, test_indices = prepare_splits(full_ds, Config)
    train_pool = Subset(full_ds, train_indices)
    test_set   = Subset(full_ds, test_indices)
    print(f"[Splits] train_pool={len(train_pool)}  test={len(test_set)}  "
          f"(test seed=0, fixed across all variants and seeds)")
    Config.N_EVAL_SAMPLES = min(Config.N_EVAL_SAMPLES, len(test_set))

    if args.shard_count == 1 or args.shard_index == 0 or args.prepare_data_only:
        validate_dns(Config, payload, device)
    if args.prepare_data_only:
        print("[Done] Dataset prepared and cached.")
        return

    if args.sweep_gate:
        run_gate_sensitivity(Config, train_pool, test_set, device)
        print("[Done] Gate sensitivity sweep complete.")
        return

    jobs = [(v, s) for v in args.variants for s in Config.SEEDS]
    n_total = len(jobs)

    if args.claim_queue:
        jobs.sort(key=lambda vs: (_VARIANT_COST_RANK.get(vs[0], 99), vs[1]))
        print("\n[Phase 2] Training/eval via dynamic claim queue "
              f"({n_total} jobs, longest-first).")
        done = run_claim_queue(jobs, jobs_dir, args, Config,
                               train_pool, test_set, device)
        print(f"[Queue] worker {args.worker_tag or os.getpid()} ran "
              f"{len(done)} job(s); run --aggregate_only to build outputs.")
        return

    if args.shard_count > 1:
        jobs = jobs[args.shard_index::args.shard_count]
        print(f"[Shard] index {args.shard_index}/{args.shard_count}: "
              f"{len(jobs)} of {n_total} jobs.")

    print("\n[Phase 2] Training and evaluation on FIXED test holdout.")
    results = []
    for (variant, seed) in jobs:
        res = run_single_job(variant, seed, Config, train_pool, test_set, device)
        torch.save(res, jobs_dir / f"{variant}_seed{seed}.pt")
        if args.shard_count == 1:
            results.append(res)

    if args.shard_count > 1:
        print(f"[Shard] index {args.shard_index} complete; "
              f"run with --aggregate_only to build tables and figures.")
        return

    finalize_outputs(assemble_results(results, args.variants, Config), Config)


# ============================================================================
# Reviewer-Response Experiments (OOD, ablations, mechanism, speed, UQ)
# ============================================================================

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def device_from_env():
    """Pick the CUDA device (respecting CUDA_VISIBLE_DEVICES) or fall back to CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def load_payload(cfg):
    """Load the cached training dataset payload (fields, normalization, diagnostics)."""
    cache = cfg.DATA_DIR / f"kolmogorov_{cfg.dataset_hash()}.pt"
    if not cache.exists():
        raise FileNotFoundError(f"dataset cache {cache} not found; run WRSG.py --prepare_data_only")
    return torch.load(cache, map_location="cpu")


def nu_cond_value(nu, nu_list):
    """Continuous conditioning value for a (possibly unseen) viscosity nu: the z-scored
    log-nu under the TRAINING grid's statistics, matching make_nu_cond_lut exactly."""
    logs = torch.log(torch.tensor(list(nu_list), dtype=torch.float32))
    mean, std = logs.mean(), logs.std()
    return float(((math.log(nu) - mean) / (std + 1e-8)).item())


def build_sched(cfg, device):
    """Construct the EDM schedule with the paper's settings."""
    return EDMSchedule(cfg.SIGMA_MIN, cfg.SIGMA_MAX, cfg.SIGMA_DATA, cfg.RHO,
                       cfg.P_MEAN, cfg.P_STD, device)


def load_eval_model(variant, seed, cfg, device, ckpt_dir=None):
    """Rebuild a trained model with its EMA (evaluation) weights, exactly as the paper
    evaluates: load the raw weights, then copy the EMA shadow into the live model."""
    ckpt_dir = Path(ckpt_dir) if ckpt_dir is not None else cfg.CKPT_DIR
    path = ckpt_dir / f"{variant}_seed{seed}_r{cfg.SOLVER_REV}.pt"
    if not path.exists():
        raise FileNotFoundError(f"checkpoint {path} not found")
    ckpt = torch.load(path, map_location="cpu")
    model = build_model(variant, cfg).to(device)
    model.load_state_dict(ckpt["model"])
    ema = EMA(model, decay=0.999)
    ema.load_state_dict(ckpt["ema"])
    ema.copy_to(model)
    model.eval()
    return model


def get_gate(model):
    """Return the WavenumberResolvedGate inside a wrsg / wrsg_fno denoiser (or None)."""
    bn = getattr(model, "bottleneck", None)
    if bn is None:
        return None
    if isinstance(bn, WavenumberResolvedGate):
        return bn
    if isinstance(bn, WRSGFNOBottleneck):
        return bn.gate
    return None


def physics_metrics(pred, true, kf):
    """All eight headline physics metrics for a (pred, true) pair of field batches."""
    return {
        "lsd_aggregate":            metric_lsd_aggregate(pred, true),
        "vorticity_S2_log_rmse":    metric_vorticity_structure_log_rmse(pred, true, order=2),
        "vorticity_S3_log_rmse":    metric_vorticity_structure_log_rmse(pred, true, order=3),
        "integral_length_rel_err":  metric_integral_length_rel_err(pred, true),
        "energy_flux_rmse":         metric_energy_flux_rmse(pred, true),
        "enstrophy_flux_rmse":      metric_enstrophy_flux_rmse(pred, true),
        "inverse_energy_cascade_recovery_pct": metric_inverse_energy_cascade_recovery(pred, true, kf=kf),
        "forward_enstrophy_cascade_recovery_pct": metric_forward_enstrophy_cascade_recovery(pred, true, kf=kf),
    }


def dns_snapshots(cfg, nu, n_snapshots, device, seed):
    """Run DNS at one viscosity and return n_snapshots real-space vorticity fields,
    using the same solver settings as the training data."""
    dns = KolmogorovDNS(cfg.GRID, cfg.DOMAIN, nu, cfg.FORCING_K, cfg.ALPHA_DRAG,
                        cfg.DT, device, dealias_jacobian=cfg.USE_DEALIASED_JACOBIAN)
    return dns.simulate(n_snapshots, cfg.SNAPSHOT_INTERVAL, cfg.SPINUP_STEPS, seed=seed)


def sample_n(model, sched, n, nu_cond, cfg, device, n_steps, base_seed, batch=128):
    """Draw n samples at a given conditioning value in batches (seeded, reproducible)."""
    outs = []
    done = 0
    while done < n:
        b = min(batch, n - done)
        g = base_seed + done
        outs.append(sample_dpmpp_2m(model, sched, b, nu_cond, cfg.GRID, device,
                                    n_steps=n_steps, noise_seed=g))
        done += b
    return torch.cat(outs, dim=0)


# ---------------------------------------------------------------------------
# Experiment 1: out-of-distribution viscosity
# ---------------------------------------------------------------------------

OOD_INTERP = [0.0085, 0.0115, 0.0155, 0.0210, 0.0275]
OOD_EXTRAP = [0.0045, 0.0400]
OOD_VARIANTS = ["vanilla", "fno", "wrsg", "wrsg_fno"]


def run_ood(cfg, device, variants=None, seeds=None, n_snap=192, n_steps=50):
    """Out-of-distribution viscosity test: generate DNS at unseen interior and
    out-of-range viscosities, sample each trained model at the matching continuous
    conditioning value, and score the eight physics metrics. Writes ood_metrics.csv."""
    variants = variants or OOD_VARIANTS
    seeds = seeds or cfg.SEEDS
    payload = load_payload(cfg)
    mean, std = payload["raw_mean"], payload["raw_std"]
    sched = build_sched(cfg, device)
    rows = []
    nus = [(nu, "interp") for nu in OOD_INTERP] + [(nu, "extrap") for nu in OOD_EXTRAP]
    for nu, kind in nus:
        in_range = cfg.NU_LIST[0] <= nu <= cfg.NU_LIST[-1]
        print(f"[OOD] nu={nu} ({kind}, {'in-range' if in_range else 'out-of-range'}) "
              f"-- DNS {n_snap} snapshots ...", flush=True)
        t0 = time.time()
        raw = dns_snapshots(cfg, nu, n_snap, device, seed=2000 + int(round(nu * 1e4)))
        true = ((raw - mean) / (std + 1e-8)).unsqueeze(1)
        print(f"[OOD]   DNS done in {time.time()-t0:.1f}s", flush=True)
        ncond = nu_cond_value(nu, cfg.NU_LIST)
        for variant in variants:
            for seed in seeds:
                model = load_eval_model(variant, seed, cfg, device)
                pred = sample_n(model, sched, n_snap, ncond, cfg, device, n_steps,
                                base_seed=seed * 1000 + int(round(nu * 1e4)))
                m = physics_metrics(pred, true, cfg.FORCING_K)
                m.update({"nu": nu, "kind": kind, "in_range": in_range,
                          "nu_cond": ncond, "variant": variant, "seed": seed,
                          "n_snap": n_snap})
                rows.append(m)
                del model, pred
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print(f"[OOD]   {variant:9s} seed{seed}: LSD={m['lsd_aggregate']:.3f} "
                      f"Efl={m['energy_flux_rmse']:.3f} FwdZ={m['forward_enstrophy_cascade_recovery_pct']:.0f}%",
                      flush=True)
    df = pd.DataFrame(rows)
    out = cfg.CSV_DIR / "ood_metrics.csv"
    df.to_csv(out, index=False)
    print(f"[OOD] wrote {out}  ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# Experiment 2: sampling-step sensitivity
# ---------------------------------------------------------------------------

STEP_GRID = [10, 15, 20, 30, 50, 75, 100]


def run_step_ablation(cfg, device, variants=("vanilla", "wrsg", "wrsg_fno"),
                      seeds=None, step_grid=None, n_eval=192):
    """Re-sample existing checkpoints on the in-distribution test set at several solver
    step counts and record LSD and flux error. Writes sampling_steps.csv."""
    seeds = seeds or cfg.SEEDS
    step_grid = step_grid or STEP_GRID
    payload = load_payload(cfg)
    fields = payload["fields"]; re = payload["re_labels"]
    full = TurbulenceDataset(fields, re)
    _, test_idx = prepare_splits(full, cfg)
    test_fields = fields[test_idx][:n_eval].to(device)
    test_re = re[test_idx][:n_eval].to(device)
    nu_lut = make_nu_cond_lut(cfg.NU_LIST, device)
    sched = build_sched(cfg, device)
    rows = []
    for variant in variants:
        for seed in seeds:
            model = load_eval_model(variant, seed, cfg, device)
            for ns in step_grid:
                preds, trues = [], []
                for r in range(len(cfg.NU_LIST)):
                    idx = (test_re == r).nonzero(as_tuple=False).flatten()
                    if idx.numel() == 0:
                        continue
                    gens = sample_dpmpp_2m(model, sched, int(idx.numel()), float(nu_lut[r]),
                                           cfg.GRID, device, n_steps=ns,
                                           noise_seed=seed * 1000 + r)
                    preds.append(gens); trues.append(test_fields[idx])
                pred = torch.cat(preds, 0); true = torch.cat(trues, 0)
                rows.append({"variant": variant, "seed": seed, "n_steps": ns,
                             "lsd_aggregate": metric_lsd_aggregate(pred, true),
                             "energy_flux_rmse": metric_energy_flux_rmse(pred, true),
                             "enstrophy_flux_rmse": metric_enstrophy_flux_rmse(pred, true)})
                print(f"[Steps] {variant:9s} seed{seed} steps={ns:3d}: "
                      f"LSD={rows[-1]['lsd_aggregate']:.3f}", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    out = cfg.CSV_DIR / "sampling_steps.csv"
    df.to_csv(out, index=False)
    print(f"[Steps] wrote {out}")
    return df


# ---------------------------------------------------------------------------
# Experiment 3: inference speed / memory vs DNS
# ---------------------------------------------------------------------------

def run_speed(cfg, device, variants=None, n_steps=50, n_fields=64, reps=3):
    """Measure sampling throughput and peak inference memory per variant on the current
    device, plus the DNS wall-clock to produce one decorrelated field, for comparison.
    Writes inference_speed.csv."""
    variants = variants or list(VARIANT_ORDER)
    sched = build_sched(cfg, device)
    ncond = nu_cond_value(cfg.NU_LIST[0], cfg.NU_LIST)
    rows = []
    for variant in variants:
        try:
            model = load_eval_model(variant, cfg.SEEDS[0], cfg, device)
        except FileNotFoundError:
            print(f"[Speed] no checkpoint for {variant}; skipping")
            continue
        n_params = sum(p.numel() for p in model.parameters())
        _ = sample_dpmpp_2m(model, sched, n_fields, ncond, cfg.GRID, device,
                            n_steps=n_steps, noise_seed=0)
        if device.type == "cuda":
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        ts = []
        for rep in range(reps):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            _ = sample_dpmpp_2m(model, sched, n_fields, ncond, cfg.GRID, device,
                                n_steps=n_steps, noise_seed=rep + 1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            ts.append(time.time() - t0)
        peak_mb = (torch.cuda.max_memory_allocated() / 1e6) if device.type == "cuda" else float("nan")
        per_field_ms = 1000.0 * float(np.mean(ts)) / n_fields
        rows.append({"variant": variant, "device": str(device), "n_params": n_params,
                     "n_steps": n_steps, "n_fields": n_fields,
                     "sec_total_mean": float(np.mean(ts)), "sec_total_std": float(np.std(ts)),
                     "ms_per_field": per_field_ms,
                     "fields_per_sec": n_fields / float(np.mean(ts)),
                     "peak_mem_mb": peak_mb})
        print(f"[Speed] {variant:12s} {per_field_ms:7.1f} ms/field  "
              f"peak={peak_mb:.0f}MB  params={n_params/1e6:.2f}M", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    dns = KolmogorovDNS(cfg.GRID, cfg.DOMAIN, cfg.NU_LIST[0], cfg.FORCING_K,
                        cfg.ALPHA_DRAG, cfg.DT, device,
                        dealias_jacobian=cfg.USE_DEALIASED_JACOBIAN)
    w_hat = dns._init_field(0)
    for _ in range(50):
        w_hat = dns.step_rk4(w_hat)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(cfg.SNAPSHOT_INTERVAL):
        w_hat = dns.step_rk4(w_hat)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dns_interval_s = time.time() - t0
    rows.append({"variant": "DNS_solver", "device": str(device), "n_params": 0,
                 "n_steps": cfg.SNAPSHOT_INTERVAL, "n_fields": 1,
                 "sec_total_mean": dns_interval_s, "sec_total_std": 0.0,
                 "ms_per_field": dns_interval_s * 1000.0,
                 "fields_per_sec": 1.0 / dns_interval_s, "peak_mem_mb": float("nan")})
    print(f"[Speed] DNS one decorrelated field ({cfg.SNAPSHOT_INTERVAL} RK4 steps): "
          f"{dns_interval_s*1000:.1f} ms", flush=True)
    df = pd.DataFrame(rows)
    out = cfg.CSV_DIR / f"inference_speed_{device.type}.csv"
    df.to_csv(out, index=False)
    print(f"[Speed] wrote {out}")
    return df


# ---------------------------------------------------------------------------
# Experiment 4: gate mechanism -- learned multiplier vs noise level and band
# ---------------------------------------------------------------------------

@torch.no_grad()
def gate_multiplier_grid(model, sched, sigmas, nu_cond, device):
    """For each sigma, evaluate the gate's per-band spectral multiplier
    m_b(sigma) = 1 + scale * (2*sigmoid(g_{c,b}) - 1) of the gate equation, averaged over
    channels. Returns an array of shape (len(sigmas), n_bins) plus the learned scale."""
    gate = get_gate(model)
    if gate is None:
        return None, None
    scale = float(gate.scale.item())
    rows = []
    for s in sigmas:
        sig = torch.tensor([s], device=device, dtype=torch.float32)
        _, _, _, c_noise = sched.preconditioning(sig)
        emb = model.emb(c_noise, torch.tensor([nu_cond], device=device))
        out = gate.gate_mlp(emb)
        C, nb, rk = gate.C, gate.n_bins, gate.rank
        a = out[:, :rk * C].view(1, rk, C)
        b = out[:, rk * C:rk * (C + nb)].view(1, rk, nb)
        d = out[:, rk * (C + nb):]
        g = torch.einsum("brc,brk->bck", a, b) + d.unsqueeze(-1)
        gck = torch.sigmoid(g)
        mult = 1.0 + scale * (2.0 * gck - 1.0)
        rows.append(mult.mean(dim=1).squeeze(0).cpu().numpy())
    return np.stack(rows, axis=0), scale


def run_mechanism(cfg, device, variant="wrsg", seeds=None, n_sigma=40):
    """Probe the learned gate across noise levels and wavenumber bands at the mid viscosity,
    averaged over seeds. Writes gate_mechanism.csv and gate_mechanism.png."""
    seeds = seeds or cfg.SEEDS
    sched = build_sched(cfg, device)
    sigmas = np.logspace(np.log10(cfg.SIGMA_MIN), np.log10(cfg.SIGMA_MAX), n_sigma)
    nu_mid = cfg.NU_LIST[len(cfg.NU_LIST) // 2]
    grids, scales = [], []
    for seed in seeds:
        model = load_eval_model(variant, seed, cfg, device)
        ncond = nu_cond_value(nu_mid, cfg.NU_LIST)
        grid, scale = gate_multiplier_grid(model, sched, sigmas, ncond, device)
        if grid is None:
            print(f"[Mech] {variant} has no gate; aborting")
            return None
        grids.append(grid); scales.append(scale)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    grid = np.mean(grids, axis=0)
    scale_mean = float(np.mean(scales))
    rows = []
    for i, s in enumerate(sigmas):
        for bnum in range(grid.shape[1]):
            rows.append({"variant": variant, "sigma": s, "bin": bnum,
                         "multiplier": float(grid[i, bnum]), "nu": nu_mid,
                         "scale_mean": scale_mean})
    df = pd.DataFrame(rows)
    out = cfg.CSV_DIR / "gate_mechanism.csv"
    df.to_csv(out, index=False)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    dev = max(abs(grid - 1.0).max(), 1e-3)
    im = ax.pcolormesh(np.arange(grid.shape[1]), sigmas, grid, cmap="RdBu_r",
                       shading="nearest", vmin=1.0 - dev, vmax=1.0 + dev)
    ax.set_yscale("log")
    ax.axhline(cfg.FLUX_SIGMA_MAX, color="k", ls="--", lw=1.2)
    ax.set_xlabel("radial wavenumber bin (low $k$ $\\rightarrow$ high $k$)", fontsize=12)
    ax.set_ylabel("diffusion noise level $\\sigma$", fontsize=12)
    ax.set_title(f"Learned gate multiplier (mean over channels & {len(seeds)} seeds), "
                 f"$s={scale_mean:.3f}$", fontsize=11)
    cb = fig.colorbar(im, ax=ax); cb.set_label("per-band multiplier", fontsize=11)
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / "gate_mechanism.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Mech] wrote {out} and gate_mechanism.png  (scale={scale_mean:.3f})")
    return df


# ---------------------------------------------------------------------------
# Experiment 5: root cause -- per-band denoising error vs noise level
# ---------------------------------------------------------------------------

@torch.no_grad()
def band_logerror_map(model, sched, x0, sigmas, nu_cond, device):
    """Single-step denoising error in LOG-spectral space: for each sigma, add noise to
    clean fields x0, denoise once, and return the per-band |log10 E_pred - log10 E_true|
    of the batch-mean spectra. Log space is bounded (unlike a ratio with tiny high-k
    denominators) and is the per-(k,sigma) contribution to the LSD. Returns (n_sigma, n_k), k."""
    Et_full, kk = compute_dealiased_spectrum(x0.squeeze(1))
    Et = Et_full.mean(0).clamp(min=1e-12)
    out = np.zeros((len(sigmas), Et.shape[0]))
    B = x0.shape[0]
    cond = torch.full((B,), float(nu_cond), device=device)
    for i, s in enumerate(sigmas):
        sig = torch.full((B,), float(s), device=device)
        g = torch.Generator(device=device).manual_seed(12345 + i)
        x0p = denoise_preconditioned(model, x0 + torch.randn(x0.shape, device=device,
                                       generator=g) * s, sig, cond, sched)
        Ep, _ = compute_dealiased_spectrum(x0p.squeeze(1))
        Ep = Ep.mean(0).clamp(min=1e-12)
        out[i] = (torch.log10(Ep) - torch.log10(Et)).abs().cpu().numpy()
    return out, kk.cpu().numpy()


@torch.no_grad()
def generated_band_ratio(model, sched, true_norm, nu_cond, cfg, device, n, seed):
    """End-to-end: sample n fields and return the ratio of the batch-mean generated
    spectrum to the DNS spectrum, per band (the actual distortion the model produces)."""
    pred = sample_n(model, sched, n, nu_cond, cfg, device, cfg.N_SAMPLE_STEPS, base_seed=seed)
    Ep, kk = compute_dealiased_spectrum(pred.squeeze(1))
    Et, _ = compute_dealiased_spectrum(true_norm.squeeze(1))
    ratio = (Ep.mean(0).clamp(min=1e-12) / Et.mean(0).clamp(min=1e-12)).cpu().numpy()
    return ratio, kk.cpu().numpy()


def run_rootcause(cfg, device, variants=("vanilla", "wrsg"), seed=None,
                  n_sigma=24, n_fields=128, regime=0):
    """Locate the origin of the spectral distortion two ways: (i) the single-step
    log-band denoising error vs noise level (where in (k, sigma) the bias lives), and
    (ii) the end-to-end generated/DNS spectrum ratio (the distortion that survives to the
    samples). Writes rootcause_band_error.csv, rootcause_gen_ratio.csv, and figures."""
    seed = seed or cfg.SEEDS[0]
    payload = load_payload(cfg)
    fields = payload["fields"]; re = payload["re_labels"]
    full = TurbulenceDataset(fields, re)
    _, test_idx = prepare_splits(full, cfg)
    sel = [i for i in test_idx if int(re[i]) == regime][:n_fields]
    x0 = fields[sel].to(device)
    sched = build_sched(cfg, device)
    sigmas = np.logspace(np.log10(cfg.SIGMA_MIN), np.log10(cfg.SIGMA_MAX), n_sigma)
    ncond = nu_cond_value(cfg.NU_LIST[regime], cfg.NU_LIST)
    maps, ratios, rows, rrows = {}, {}, [], []
    for variant in variants:
        model = load_eval_model(variant, seed, cfg, device)
        m, kk = band_logerror_map(model, sched, x0, sigmas, ncond, device)
        ratio, kkr = generated_band_ratio(model, sched, x0, ncond, cfg, device,
                                          n=x0.shape[0], seed=seed * 7 + regime)
        maps[variant] = (m, kk); ratios[variant] = (ratio, kkr)
        for i, s in enumerate(sigmas):
            for j, kval in enumerate(kk):
                rows.append({"variant": variant, "sigma": float(s), "k": float(kval),
                             "log_band_err": float(m[i, j]), "regime": regime})
        for j, kval in enumerate(kkr):
            rrows.append({"variant": variant, "k": float(kval),
                          "gen_dns_ratio": float(ratio[j]), "regime": regime})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(cfg.CSV_DIR / "rootcause_band_error.csv", index=False)
    pd.DataFrame(rrows).to_csv(cfg.CSV_DIR / "rootcause_gen_ratio.csv", index=False)

    fig, axes = plt.subplots(1, len(variants) + 1,
                             figsize=(4.6 * (len(variants) + 1), 4.2))
    vmax = max(maps[v][0][:, 1:].max() for v in variants)
    for ax, variant in zip(axes[:-1], variants):
        m, kk = maps[variant]
        im = ax.pcolormesh(kk[1:], sigmas, m[:, 1:], cmap="magma", shading="nearest",
                           vmin=0, vmax=vmax)
        ax.set_yscale("log")
        ax.set_xlabel("wavenumber $k$", fontsize=12)
        ax.set_title(f"{vlabel(variant)}: single-step log-band error", fontsize=11)
        ax.tick_params(labelsize=11)
    axes[0].set_ylabel("noise level $\\sigma$", fontsize=12)
    cb = fig.colorbar(im, ax=axes[:-1].tolist(), fraction=0.046)
    cb.set_label("$|\\log_{10} E_{pred}-\\log_{10} E_{true}|$", fontsize=10)
    axr = axes[-1]
    for variant in variants:
        ratio, kkr = ratios[variant]
        axr.semilogy(kkr[1:], ratio[1:], lw=2, color=VARIANT_COLORS.get(variant),
                     label=vlabel(variant))
    axr.axhline(1.0, color="k", ls="--", lw=1)
    axr.set_xlabel("wavenumber $k$", fontsize=12)
    axr.set_ylabel("generated / DNS spectrum", fontsize=12)
    axr.set_title("End-to-end spectral distortion", fontsize=11)
    axr.legend(fontsize=10); axr.tick_params(labelsize=11)
    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / "rootcause_band_error.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[Root] wrote rootcause_band_error.csv, rootcause_gen_ratio.csv, figure")
    return pd.DataFrame(rrows)


# ---------------------------------------------------------------------------
# Gate sweep / loss-weight cells (each cell = one shardable training+eval job)
# ---------------------------------------------------------------------------

def _physics_loss(x0_pred, x0, sigma, cfg):
    """The training physics loss (weighted), with the integral-length term active only at
    low noise, matching the schedule used during training."""
    L = (cfg.LAMBDA_ENSTROPHY * loss_enstrophy(x0_pred, x0)
         + cfg.LAMBDA_SPECTRAL * loss_spectral(x0_pred, x0)
         + cfg.LAMBDA_STRUCT * loss_structure(x0_pred, x0, r_max=cfg.STRUCT_R_MAX))
    if sigma < cfg.FLUX_SIGMA_MAX:
        L = L + cfg.LAMBDA_INTLEN * loss_integral_length(x0_pred, x0)
    return L


def gradflow_at_sigma(model, sched, x0, sigma, nu_cond, cfg, device):
    """At one noise level, backpropagate the physics loss through the denoiser and return
    the gradient norm reaching the deepest encoder feature (input to the bottleneck) and,
    for a gated model, the fraction of the total parameter-gradient norm carried by the gate."""
    captured = {}
    def hook(_m, _i, out):
        out.retain_grad()
        captured["h"] = out
    handle = model.b3.register_forward_hook(hook)
    B = x0.shape[0]
    cond = torch.full((B,), float(nu_cond), device=device)
    sig = torch.full((B,), float(sigma), device=device)
    g = torch.Generator(device=device).manual_seed(2024)
    x_noisy = x0 + torch.randn(x0.shape, device=device, generator=g) * sigma
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        x0_pred = denoise_preconditioned(model, x_noisy, sig, cond, sched)
        _physics_loss(x0_pred, x0, sigma, cfg).backward()
    grad_h = float(captured["h"].grad.flatten(1).norm(dim=1).mean().item())
    handle.remove()
    total_sq = sum(p.grad.pow(2).sum().item() for p in model.parameters() if p.grad is not None)
    gate = get_gate(model)
    gate_frac = float("nan")
    if gate is not None:
        gate_sq = sum(p.grad.pow(2).sum().item() for p in gate.parameters() if p.grad is not None)
        gate_frac = gate_sq / max(total_sq, 1e-12)
    return grad_h, gate_frac


def run_gradflow(cfg, device, seeds=None, n_sigma=24, n_fields=64):
    """Measure how the physics-loss gradient flows across diffusion noise levels, with the
    gate (wrsg) and without it (wrsg_phys). Reports the gradient norm reaching the deepest
    encoder feature vs sigma for both, and the gate's share of the parameter gradient for
    wrsg. Writes gradflow.csv and gradflow.png."""
    seeds = seeds or cfg.SEEDS
    payload = load_payload(cfg)
    fields = payload["fields"]; re = payload["re_labels"]
    full = TurbulenceDataset(fields, re)
    _, test_idx = prepare_splits(full, cfg)
    regime = len(cfg.NU_LIST) // 2
    sel = [i for i in test_idx if int(re[i]) == regime][:n_fields]
    x0 = fields[sel].to(device)
    ncond = nu_cond_value(cfg.NU_LIST[regime], cfg.NU_LIST)
    sched = build_sched(cfg, device)
    sigmas = np.logspace(np.log10(cfg.SIGMA_MIN), np.log10(cfg.SIGMA_MAX), n_sigma)
    rows = []
    for variant in ("wrsg", "wrsg_phys"):
        per_seed_gh, per_seed_gf = [], []
        for seed in seeds:
            model = load_eval_model(variant, seed, cfg, device)
            gh, gf = [], []
            for s in sigmas:
                a, b = gradflow_at_sigma(model, sched, x0, float(s), ncond, cfg, device)
                gh.append(a); gf.append(b)
            per_seed_gh.append(gh); per_seed_gf.append(gf)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        gh_m = np.mean(per_seed_gh, axis=0); gf_m = np.mean(per_seed_gf, axis=0)
        for i, s in enumerate(sigmas):
            rows.append({"variant": variant, "sigma": float(s),
                         "grad_into_backbone": float(gh_m[i]),
                         "gate_grad_fraction": float(gf_m[i])})
    df = pd.DataFrame(rows)
    df.to_csv(cfg.CSV_DIR / "gradflow.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for variant in ("wrsg", "wrsg_phys"):
        d = df[df.variant == variant]
        axes[0].loglog(d.sigma, d.grad_into_backbone, lw=2, marker="o", ms=3,
                       label=vlabel(variant))
    axes[0].axvline(cfg.FLUX_SIGMA_MAX, color="k", ls="--", lw=1)
    axes[0].set_xlabel("noise level $\\sigma$", fontsize=12)
    axes[0].set_ylabel("$\\|\\partial \\mathcal{L}_{phys}/\\partial h\\|$ (encoder feature)", fontsize=12)
    axes[0].set_title("Physics-loss gradient reaching the backbone", fontsize=12)
    axes[0].legend(fontsize=11); axes[0].tick_params(labelsize=11)
    dw = df[df.variant == "wrsg"]
    axes[1].semilogx(dw.sigma, dw.gate_grad_fraction, lw=2, marker="o", ms=3, color="#C9534D")
    axes[1].axvline(cfg.FLUX_SIGMA_MAX, color="k", ls="--", lw=1)
    axes[1].set_xlabel("noise level $\\sigma$", fontsize=12)
    axes[1].set_ylabel("gate share of physics-loss gradient", fontsize=12)
    axes[1].set_title("Fraction of the gradient routed through the gate (WRSG)", fontsize=12)
    axes[1].tick_params(labelsize=11)
    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / "gradflow.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[GradFlow] wrote gradflow.csv and gradflow.png")
    return df


@torch.no_grad()
def run_uq_depth(cfg, device, variant="wrsg", seeds=None, n_cond=20, n_ens=None):
    """Resolve the calibrated uncertainty by viscosity regime and by wavenumber, to show
    where the under-dispersion lives. Per regime: raw and per-regime-calibrated 90% pixel
    coverage and CRPS. Per wavenumber: coverage of the DNS band power by the ensemble's
    5-95 spectral band. Writes uq_depth_regime.csv, uq_depth_spectral.csv, uq_depth.png."""
    seeds = seeds or [cfg.SEEDS[0]]
    n_ens = n_ens or cfg.N_ENSEMBLE
    payload = load_payload(cfg)
    fields = payload["fields"]; re = payload["re_labels"]
    full = TurbulenceDataset(fields, re)
    _, test_idx = prepare_splits(full, cfg)
    nu_lut = make_nu_cond_lut(cfg.NU_LIST, device)
    sched = build_sched(cfg, device)
    seed = seeds[0]
    reg_rows, spec_rows = [], []
    for r in range(len(cfg.NU_LIST)):
        sel = [i for i in test_idx if int(re[i]) == r][:n_cond]
        if len(sel) < 4:
            continue
        truth = fields[sel].to(device)
        m = len(sel)
        ens = torch.zeros(n_ens, m, 1, cfg.GRID, cfg.GRID, device=device)
        for j in range(n_ens):
            ens[j] = sample_dpmpp_2m(model_cache(variant, seed, cfg, device), sched, m,
                                     float(nu_lut[r]), cfg.GRID, device,
                                     n_steps=cfg.N_SAMPLE_STEPS,
                                     noise_seed=seed * 100000 + j * 1000 + r)
        n_cal = max(2, m // 2)
        alpha = calibrate_alpha(ens[:, :n_cal], truth[:n_cal], target_cov=0.90)
        reg_rows.append({
            "regime": r, "nu": cfg.NU_LIST[r], "n_cond": m,
            "coverage_raw": metric_coverage_band(ens[:, n_cal:], truth[n_cal:]),
            "coverage_calibrated": coverage_calibrated(ens[:, n_cal:], truth[n_cal:], alpha),
            "crps": metric_crps_ensemble(ens[:, n_cal:], truth[n_cal:]),
            "alpha": alpha})
        E_ens_flat, kk = compute_radial_spectrum(ens.reshape(n_ens * m, cfg.GRID, cfg.GRID))
        E_ens = E_ens_flat.reshape(n_ens, m, -1)
        E_true, _ = compute_radial_spectrum(truth.squeeze(1))
        n_cal_s = max(2, m // 2)
        for j, kval in enumerate(kk.tolist()):
            ens_cal_k = E_ens[:, :n_cal_s, j]
            ens_tst_k = E_ens[:, n_cal_s:, j]
            a_k = calibrate_alpha(ens_cal_k, E_true[:n_cal_s, j], target_cov=0.90)
            spec_rows.append({
                "regime": r, "k": kval,
                "spectral_coverage_raw": coverage_calibrated(ens_tst_k, E_true[n_cal_s:, j], 1.0),
                "spectral_coverage_calibrated": coverage_calibrated(ens_tst_k, E_true[n_cal_s:, j], a_k),
                "alpha_k": a_k})
        print(f"[UQdepth] regime {r} nu={cfg.NU_LIST[r]}: raw={reg_rows[-1]['coverage_raw']:.3f} "
              f"cal={reg_rows[-1]['coverage_calibrated']:.3f} crps={reg_rows[-1]['crps']:.3f}", flush=True)
    reg = pd.DataFrame(reg_rows); spec = pd.DataFrame(spec_rows)
    reg.to_csv(cfg.CSV_DIR / "uq_depth_regime.csv", index=False)
    spec.to_csv(cfg.CSV_DIR / "uq_depth_spectral.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].plot(reg.nu, reg.coverage_raw, "o-", label="raw", lw=2)
    axes[0].plot(reg.nu, reg.coverage_calibrated, "s-", label="calibrated", lw=2)
    axes[0].axhline(0.90, color="k", ls="--", lw=1)
    axes[0].set_xlabel("viscosity $\\nu$", fontsize=12)
    axes[0].set_ylabel("90% pixel coverage", fontsize=12)
    axes[0].set_title(f"Coverage by regime ({vlabel(variant)})", fontsize=12)
    axes[0].legend(fontsize=11); axes[0].tick_params(labelsize=11)
    spec_raw = spec.groupby("k").spectral_coverage_raw.mean()
    spec_cal = spec.groupby("k").spectral_coverage_calibrated.mean()
    axes[1].plot(spec_raw.index, spec_raw.values, "o-", lw=2, color="#7050C0", label="raw")
    axes[1].plot(spec_cal.index, spec_cal.values, "s-", lw=2, color="#2E7D43",
                 label="per-wavenumber calibrated")
    axes[1].axhline(0.90, color="k", ls="--", lw=1)
    axes[1].set_xlabel("wavenumber $k$", fontsize=12)
    axes[1].set_ylabel("ensemble band coverage of DNS power", fontsize=12)
    axes[1].set_title("Spectral coverage by wavenumber", fontsize=12)
    axes[1].legend(fontsize=10); axes[1].tick_params(labelsize=11)
    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / "uq_depth.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[UQdepth] wrote uq_depth_regime.csv, uq_depth_spectral.csv, uq_depth.png")
    return reg


_MODEL_CACHE = {}
def model_cache(variant, seed, cfg, device):
    """Load and cache one evaluation model so repeated ensemble draws reuse it."""
    key = (variant, seed)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = load_eval_model(variant, seed, cfg, device)
    return _MODEL_CACHE[key]


def run_sweep_cell(cfg, device, train_pool, test_set, bins, rank, seed):
    """Train+eval one WRSG model at a given (gate_bins, gate_rank, seed), writing the
    checkpoint under a per-cell directory so parallel workers do not collide."""
    cfg.GATE_N_BINS, cfg.GATE_RANK = bins, rank
    cfg.CKPT_DIR = cfg.CKPT_DIR if "gate_sweep" in str(cfg.CKPT_DIR) else \
        cfg.CKPT_DIR / f"gate_sweep_b{bins}_k{rank}"
    cfg.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    model, sched, _, _ = train_one_model("wrsg", seed, cfg, train_pool, device,
                                            use_physics=True, use_flux=True)
    agg = evaluate(model, sched, test_set, cfg, device, "wrsg", seed)[0]
    return {"gate_bins": bins, "gate_rank": rank, "seed": seed,
            "lsd_aggregate": agg["lsd_aggregate"],
            "energy_flux_rmse": agg["energy_flux_rmse"],
            "integral_length_rel_err": agg["integral_length_rel_err"]}


def run_lambda_cell(cfg, device, train_pool, test_set, scale, seed):
    """Train+eval one WRSG model with all physics-loss weights multiplied by `scale`,
    for the loss-weight sensitivity. scale=1.0 reproduces the paper setting."""
    base = dict(LAMBDA_ENSTROPHY=cfg.LAMBDA_ENSTROPHY, LAMBDA_SPECTRAL=cfg.LAMBDA_SPECTRAL,
                LAMBDA_STRUCT=cfg.LAMBDA_STRUCT, LAMBDA_INTLEN=cfg.LAMBDA_INTLEN,
                LAMBDA_FLUX=cfg.LAMBDA_FLUX)
    for k, v in base.items():
        setattr(cfg, k, v * scale)
    cfg.CKPT_DIR = cfg.CKPT_DIR if "lambda_sweep" in str(cfg.CKPT_DIR) else \
        cfg.CKPT_DIR / f"lambda_sweep_s{scale:g}"
    cfg.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    model, sched, _, _ = train_one_model("wrsg", seed, cfg, train_pool, device,
                                            use_physics=True, use_flux=True)
    agg = evaluate(model, sched, test_set, cfg, device, "wrsg", seed)[0]
    for k, v in base.items():
        setattr(cfg, k, v)
    return {"lambda_scale": scale, "seed": seed,
            "lsd_aggregate": agg["lsd_aggregate"],
            "vorticity_S2_log_rmse": agg["vorticity_S2_log_rmse"],
            "integral_length_rel_err": agg["integral_length_rel_err"],
            "energy_flux_rmse": agg["energy_flux_rmse"]}


def run_lambdaL_cell(cfg, device, train_pool, test_set, lamL_scale, seed):
    """Train+eval one WRSG model with only the integral-length loss weight multiplied by
    lamL_scale, every other weight fixed, to test whether up-weighting the large-scale term
    recovers the integral length the default weights trade away. lamL_scale=1.0 reproduces
    the paper setting."""
    base = cfg.LAMBDA_INTLEN
    cfg.LAMBDA_INTLEN = base * lamL_scale
    cfg.CKPT_DIR = cfg.CKPT_DIR if "lambdaL_sweep" in str(cfg.CKPT_DIR) else \
        cfg.CKPT_DIR / f"lambdaL_sweep_s{lamL_scale:g}"
    cfg.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    model, sched, _, _ = train_one_model("wrsg", seed, cfg, train_pool, device,
                                            use_physics=True, use_flux=True)
    agg = evaluate(model, sched, test_set, cfg, device, "wrsg", seed)[0]
    cfg.LAMBDA_INTLEN = base
    return {"lambdaL_scale": lamL_scale, "seed": seed,
            "lsd_aggregate": agg["lsd_aggregate"],
            "vorticity_S2_log_rmse": agg["vorticity_S2_log_rmse"],
            "integral_length_rel_err": agg["integral_length_rel_err"],
            "energy_flux_rmse": agg["energy_flux_rmse"]}


def run_lowk_cell(cfg, device, train_pool, test_set, lam_lowk, seed):
    """Train+eval one WRSG model with the dense low-wavenumber spectral loss added at weight
    lam_lowk, the full physics losses kept, to test whether a large-scale spectral constraint
    recovers the integral length the default losses trade away without losing the cascade."""
    base = cfg.LAMBDA_LOWK
    cfg.LAMBDA_LOWK = lam_lowk
    cfg.CKPT_DIR = cfg.CKPT_DIR if "lowk_sweep" in str(cfg.CKPT_DIR) else \
        cfg.CKPT_DIR / f"lowk_sweep_s{lam_lowk:g}"
    cfg.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    model, sched, _, _ = train_one_model("wrsg", seed, cfg, train_pool, device,
                                            use_physics=True, use_flux=True)
    agg = evaluate(model, sched, test_set, cfg, device, "wrsg", seed)[0]
    cfg.LAMBDA_LOWK = base
    return {"lam_lowk": lam_lowk, "seed": seed,
            "lsd_aggregate": agg["lsd_aggregate"],
            "vorticity_S2_log_rmse": agg["vorticity_S2_log_rmse"],
            "integral_length_rel_err": agg["integral_length_rel_err"],
            "energy_flux_rmse": agg["energy_flux_rmse"],
            "inverse_energy_cascade_recovery_pct": agg["inverse_energy_cascade_recovery_pct"]}


def run_losscell(cfg, device, train_pool, test_set, tag, seed,
                 struct_scale=1.0, flux_scale=1.0, logspec_lam=0.0, logspec_kcut=64):
    """Train+eval one WRSG model with selected physics-loss terms rescaled, to probe a
    specific trade-off. struct_scale and flux_scale multiply those weights; logspec_lam>0
    adds a full-band log-spectral term (the low-k machinery with a wide cut) that targets the
    log-spectral distance directly. Records the aggregate metrics plus the integral length,
    cascade recovery, and the most-viscous-regime LSD."""
    keep = dict(STRUCT=cfg.LAMBDA_STRUCT, FLUX=cfg.LAMBDA_FLUX,
                LOWK=cfg.LAMBDA_LOWK, KCUT=cfg.LOWK_KCUT, CKPT=cfg.CKPT_DIR)
    cfg.LAMBDA_STRUCT = keep["STRUCT"] * struct_scale
    cfg.LAMBDA_FLUX = keep["FLUX"] * flux_scale
    cfg.LAMBDA_LOWK = logspec_lam
    cfg.LOWK_KCUT = logspec_kcut
    cfg.CKPT_DIR = keep["CKPT"] / f"loss_{tag}"
    cfg.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    model, sched, _, _ = train_one_model("wrsg", seed, cfg, train_pool, device,
                                            use_physics=True, use_flux=True)
    agg, _, per_regime, _, _, _ = evaluate(model, sched, test_set, cfg, device, "wrsg", seed)
    hi = max(per_regime, key=lambda r: r["nu"]) if per_regime else \
        {"nu": float("nan"), "lsd_aggregate": agg["lsd_aggregate"]}
    cfg.LAMBDA_STRUCT, cfg.LAMBDA_FLUX = keep["STRUCT"], keep["FLUX"]
    cfg.LAMBDA_LOWK, cfg.LOWK_KCUT, cfg.CKPT_DIR = keep["LOWK"], keep["KCUT"], keep["CKPT"]
    return {"tag": tag, "seed": seed,
            "lsd_aggregate": agg["lsd_aggregate"],
            "integral_length_rel_err": agg["integral_length_rel_err"],
            "energy_flux_rmse": agg["energy_flux_rmse"],
            "inverse_energy_cascade_recovery_pct": agg["inverse_energy_cascade_recovery_pct"],
            "forward_enstrophy_cascade_recovery_pct": agg["forward_enstrophy_cascade_recovery_pct"],
            "high_nu": hi["nu"], "high_nu_lsd": hi["lsd_aggregate"]}


def run_sigma_cell(cfg, device, train_pool, test_set, sigma_max, seed):
    """Train+eval one WRSG model at a given maximum diffusion noise level, with the rest
    of the schedule fixed. sigma_max=20 reproduces the paper setting."""
    base = cfg.SIGMA_MAX
    cfg.SIGMA_MAX = sigma_max
    cfg.CKPT_DIR = cfg.CKPT_DIR if "sigma_sweep" in str(cfg.CKPT_DIR) else \
        cfg.CKPT_DIR / f"sigma_sweep_{sigma_max:g}"
    cfg.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    model, sched, _, _ = train_one_model("wrsg", seed, cfg, train_pool, device,
                                            use_physics=True, use_flux=True)
    agg = evaluate(model, sched, test_set, cfg, device, "wrsg", seed)[0]
    cfg.SIGMA_MAX = base
    return {"sigma_max": sigma_max, "seed": seed,
            "lsd_aggregate": agg["lsd_aggregate"],
            "vorticity_S2_log_rmse": agg["vorticity_S2_log_rmse"],
            "energy_flux_rmse": agg["energy_flux_rmse"]}


def regenerate_figures(cfg, device, n_per_regime=40):
    """Re-sample the trained checkpoints (all seeds, pooled) and rebuild the sample-based
    figures (spectra, structure functions, fluxes) with the de-cluttered layouts, plus the
    headline-bar figure from the cached CI table. No retraining."""
    payload = load_payload(cfg)
    fields = payload["fields"]; re = payload["re_labels"]
    full = TurbulenceDataset(fields, re)
    _, test_idx = prepare_splits(full, cfg)
    sched = build_sched(cfg, device)
    nu_lut = make_nu_cond_lut(cfg.NU_LIST, device)
    true_by_regime, idx_by_regime = {}, {}
    for r in range(len(cfg.NU_LIST)):
        sel = [i for i in test_idx if int(re[i]) == r][:n_per_regime]
        if sel:
            true_by_regime[r] = fields[sel].to(device)
            idx_by_regime[r] = len(sel)
    samples_by_vr = {}
    for v in VARIANT_ORDER:
        per_regime = {}
        for seed in cfg.SEEDS:
            try:
                model = load_eval_model(v, seed, cfg, device)
            except FileNotFoundError:
                continue
            for r, n in idx_by_regime.items():
                s = sample_dpmpp_2m(model, sched, n, float(nu_lut[r]),
                                    cfg.GRID, device, n_steps=cfg.N_SAMPLE_STEPS,
                                    noise_seed=seed * 1000 + r).cpu()
                per_regime.setdefault(r, []).append(s)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if per_regime:
            samples_by_vr[v] = {r: torch.cat(chunks, 0) for r, chunks in per_regime.items()}
            print(f"[Figs] sampled {v} over {len(cfg.SEEDS)} seeds", flush=True)
    samples_per_variant = {v: torch.cat([samples_by_vr[v][r] for r in idx_by_regime], 0)
                           for v in samples_by_vr}
    true_pooled = torch.cat([true_by_regime[r].cpu() for r in idx_by_regime], 0)
    true_by_regime_cpu = {r: t.cpu() for r, t in true_by_regime.items()}
    plot_dealiased_spectra(samples_per_variant, true_pooled, cfg)
    plot_structure_functions(samples_per_variant, true_pooled, cfg)
    plot_flux_panels(samples_by_vr, true_by_regime_cpu, cfg)
    ci_csv = cfg.CSV_DIR / "headline_table_with_ci.csv"
    if ci_csv.exists():
        plot_headline_bars_with_ci(pd.read_csv(ci_csv), cfg)
    print("[Figs] regenerated spectra, structure_functions, fluxes, headline_bars")


def aggregate_ablations(cfg):
    """Collect gate (bins/rank) and physics-loss-weight ablation cells into summary CSVs,
    folding in the bins=16/rank=2 and scale=1.0 baselines from the main WRSG runs (seeds
    29/47/89). Writes gate_sensitivity.csv and lambda_sensitivity.csv."""
    SWEEP_SEEDS = [29, 47, 89]
    summ = pd.read_csv(cfg.CSV_DIR / "summary_metrics.csv")
    base = summ[(summ.variant == "wrsg") & (summ.seed.isin(SWEEP_SEEDS))]

    rows = []
    for f in sorted((cfg.RESULTS_DIR / "_sweep_cells").glob("*.pt")):
        rows.append(torch.load(f, map_location="cpu"))
    for _, r in base.iterrows():
        rows.append({"gate_bins": 16, "gate_rank": 2, "seed": int(r.seed),
                     "lsd_aggregate": r.lsd_aggregate, "energy_flux_rmse": r.energy_flux_rmse,
                     "integral_length_rel_err": r.integral_length_rel_err})
    if rows:
        gdf = pd.DataFrame(rows)
        g = gdf.groupby(["gate_bins", "gate_rank"]).agg(
            n=("seed", "count"),
            lsd_mean=("lsd_aggregate", "mean"), lsd_std=("lsd_aggregate", "std"),
            eflux_mean=("energy_flux_rmse", "mean"),
            intlen_mean=("integral_length_rel_err", "mean")).reset_index()
        g.to_csv(cfg.CSV_DIR / "gate_sensitivity.csv", index=False)
        print("[Agg] gate_sensitivity.csv\n", g.to_string(index=False))

    lrows = []
    for f in sorted((cfg.RESULTS_DIR / "_lambda_cells").glob("*.pt")):
        lrows.append(torch.load(f, map_location="cpu"))
    for _, r in base.iterrows():
        lrows.append({"lambda_scale": 1.0, "seed": int(r.seed),
                      "lsd_aggregate": r.lsd_aggregate,
                      "vorticity_S2_log_rmse": r.vorticity_S2_log_rmse,
                      "integral_length_rel_err": r.integral_length_rel_err,
                      "energy_flux_rmse": r.energy_flux_rmse})
    if lrows:
        ldf = pd.DataFrame(lrows)
        l = ldf.groupby("lambda_scale").agg(
            n=("seed", "count"),
            lsd_mean=("lsd_aggregate", "mean"), lsd_std=("lsd_aggregate", "std"),
            S2_mean=("vorticity_S2_log_rmse", "mean"),
            intlen_mean=("integral_length_rel_err", "mean"),
            eflux_mean=("energy_flux_rmse", "mean")).reset_index()
        l.to_csv(cfg.CSV_DIR / "lambda_sensitivity.csv", index=False)
        print("[Agg] lambda_sensitivity.csv\n", l.to_string(index=False))

    llrows = []
    for f in sorted((cfg.RESULTS_DIR / "_lambdaL_cells").glob("*.pt")):
        llrows.append(torch.load(f, map_location="cpu"))
    for _, r in base.iterrows():
        llrows.append({"lambdaL_scale": 1.0, "seed": int(r.seed),
                       "lsd_aggregate": r.lsd_aggregate,
                       "vorticity_S2_log_rmse": r.vorticity_S2_log_rmse,
                       "integral_length_rel_err": r.integral_length_rel_err,
                       "energy_flux_rmse": r.energy_flux_rmse})
    if any("_lambdaL_cells" in str(f) for f in (cfg.RESULTS_DIR / "_lambdaL_cells").glob("*.pt")):
        lldf = pd.DataFrame(llrows)
        ll = lldf.groupby("lambdaL_scale").agg(
            n=("seed", "count"),
            lsd_mean=("lsd_aggregate", "mean"), lsd_std=("lsd_aggregate", "std"),
            intlen_mean=("integral_length_rel_err", "mean"),
            intlen_std=("integral_length_rel_err", "std"),
            eflux_mean=("energy_flux_rmse", "mean")).reset_index()
        ll.to_csv(cfg.CSV_DIR / "lambdaL_sensitivity.csv", index=False)
        print("[Agg] lambdaL_sensitivity.csv\n", ll.to_string(index=False))

    lkrows = []
    for f in sorted((cfg.RESULTS_DIR / "_lowk_cells").glob("*.pt")):
        lkrows.append(torch.load(f, map_location="cpu"))
    if lkrows:
        for _, r in base.iterrows():
            lkrows.append({"lam_lowk": 0.0, "seed": int(r.seed),
                           "lsd_aggregate": r.lsd_aggregate,
                           "integral_length_rel_err": r.integral_length_rel_err,
                           "energy_flux_rmse": r.energy_flux_rmse,
                           "inverse_energy_cascade_recovery_pct":
                               r.inverse_energy_cascade_recovery_pct})
        lkdf = pd.DataFrame(lkrows)
        lk = lkdf.groupby("lam_lowk").agg(
            n=("seed", "count"),
            lsd_mean=("lsd_aggregate", "mean"), lsd_std=("lsd_aggregate", "std"),
            intlen_mean=("integral_length_rel_err", "mean"),
            intlen_std=("integral_length_rel_err", "std"),
            eflux_mean=("energy_flux_rmse", "mean"),
            invE_mean=("inverse_energy_cascade_recovery_pct", "mean")).reset_index()
        lk.to_csv(cfg.CSV_DIR / "lowk_sensitivity.csv", index=False)
        print("[Agg] lowk_sensitivity.csv\n", lk.to_string(index=False))

    srows = []
    for f in sorted((cfg.RESULTS_DIR / "_sigma_cells").glob("*.pt")):
        srows.append(torch.load(f, map_location="cpu"))
    for _, r in base.iterrows():
        srows.append({"sigma_max": 20.0, "seed": int(r.seed),
                      "lsd_aggregate": r.lsd_aggregate,
                      "vorticity_S2_log_rmse": r.vorticity_S2_log_rmse,
                      "energy_flux_rmse": r.energy_flux_rmse})
    if srows:
        sdf = pd.DataFrame(srows)
        s = sdf.groupby("sigma_max").agg(
            n=("seed", "count"),
            lsd_mean=("lsd_aggregate", "mean"), lsd_std=("lsd_aggregate", "std"),
            S2_mean=("vorticity_S2_log_rmse", "mean"),
            eflux_mean=("energy_flux_rmse", "mean")).reset_index()
        s.to_csv(cfg.CSV_DIR / "sigma_sensitivity.csv", index=False)
        print("[Agg] sigma_sensitivity.csv\n", s.to_string(index=False))


def aggregate_loss_cells(cfg):
    """Collect the per-term loss-probe cells (_loss_cells) into loss_probe.csv: per tag, the
    mean aggregate LSD, integral length, energy-flux error, cascade recovery, and the
    most-viscous-regime LSD, for the targeted integral-length/cascade/high-viscosity probes."""
    rows = []
    for f in sorted((cfg.RESULTS_DIR / "_loss_cells").glob("*.pt")):
        rows.append(torch.load(f, map_location="cpu"))
    if not rows:
        print("[LossAgg] no cells found"); return None
    df = pd.DataFrame(rows)
    summ = df.groupby("tag").agg(
        n=("seed", "count"),
        lsd_mean=("lsd_aggregate", "mean"), lsd_std=("lsd_aggregate", "std"),
        intlen_mean=("integral_length_rel_err", "mean"),
        eflux_mean=("energy_flux_rmse", "mean"),
        invE_mean=("inverse_energy_cascade_recovery_pct", "mean"),
        fwdZ_mean=("forward_enstrophy_cascade_recovery_pct", "mean"),
        high_nu_lsd_mean=("high_nu_lsd", "mean")).reset_index()
    summ.to_csv(cfg.CSV_DIR / "loss_probe.csv", index=False)
    print(summ.to_string(index=False))
    print("[LossAgg] wrote loss_probe.csv")
    return summ


def _load_train_test(cfg, device):
    """Reconstruct the train pool and fixed test holdout from the cached dataset."""
    payload = load_payload(cfg)
    full = TurbulenceDataset(payload["fields"], payload["re_labels"])
    tr_idx, te_idx = prepare_splits(full, cfg)
    return Subset(full, tr_idx), Subset(full, te_idx)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _reviews_main():
    p = argparse.ArgumentParser(description="WRSG reviewer-response experiments")
    p.add_argument("cmd", choices=["ood", "steps", "speed", "mechanism", "rootcause",
                                   "gradflow", "uqdepth", "sweepcell", "lambdacell",
                                   "lambdaLcell", "lowkcell", "losscell", "sigmacell",
                                   "figs", "aggregate", "lossagg", "smoke"])
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--variants", nargs="+", default=None)
    p.add_argument("--bins", type=int, default=16)
    p.add_argument("--rank", type=int, default=2)
    p.add_argument("--seed", type=int, default=29)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--lamL", type=float, default=1.0)
    p.add_argument("--lamlowk", type=float, default=0.3)
    p.add_argument("--kcut", type=int, default=6)
    p.add_argument("--struct_scale", type=float, default=1.0)
    p.add_argument("--flux_scale", type=float, default=1.0)
    p.add_argument("--logspec", type=float, default=0.0)
    p.add_argument("--tag", default="cell")
    p.add_argument("--sigma_max", type=float, default=20.0)
    p.add_argument("--cpu", action="store_true", help="force CPU (for speed on second hardware)")
    p.add_argument("--n_steps", type=int, default=50)
    p.add_argument("--epochs", type=int, default=None, help="override training epochs (testing)")
    args = p.parse_args()

    Config.setup()
    if args.epochs is not None:
        Config.EPOCHS = args.epochs
    device = torch.device("cpu") if args.cpu else device_from_env()
    print(f"[Reviews] cmd={args.cmd} device={device}")

    if args.cmd == "ood":
        run_ood(Config, device, variants=args.variants, seeds=args.seeds)
    elif args.cmd == "steps":
        run_step_ablation(Config, device,
                          variants=tuple(args.variants) if args.variants else
                          ("vanilla", "wrsg", "wrsg_fno"), seeds=args.seeds)
    elif args.cmd == "speed":
        run_speed(Config, device, variants=args.variants, n_steps=args.n_steps)
    elif args.cmd == "mechanism":
        run_mechanism(Config, device, seeds=args.seeds)
    elif args.cmd == "rootcause":
        run_rootcause(Config, device,
                      variants=tuple(args.variants) if args.variants else ("vanilla", "wrsg"))
    elif args.cmd == "sweepcell":
        train_pool, test_set = _load_train_test(Config, device)
        res = run_sweep_cell(Config, device, train_pool, test_set,
                             args.bins, args.rank, args.seed)
        outdir = Config.RESULTS_DIR / "_sweep_cells"; outdir.mkdir(exist_ok=True)
        torch.save(res, outdir / f"gate_b{args.bins}_k{args.rank}_seed{args.seed}.pt")
        print(f"[SweepCell] {res}")
    elif args.cmd == "lambdacell":
        train_pool, test_set = _load_train_test(Config, device)
        res = run_lambda_cell(Config, device, train_pool, test_set, args.scale, args.seed)
        outdir = Config.RESULTS_DIR / "_lambda_cells"; outdir.mkdir(exist_ok=True)
        torch.save(res, outdir / f"lambda_s{args.scale:g}_seed{args.seed}.pt")
        print(f"[LambdaCell] {res}")
    elif args.cmd == "lambdaLcell":
        train_pool, test_set = _load_train_test(Config, device)
        res = run_lambdaL_cell(Config, device, train_pool, test_set, args.lamL, args.seed)
        outdir = Config.RESULTS_DIR / "_lambdaL_cells"; outdir.mkdir(exist_ok=True)
        torch.save(res, outdir / f"lambdaL_s{args.lamL:g}_seed{args.seed}.pt")
        print(f"[LambdaLCell] {res}")
    elif args.cmd == "lowkcell":
        train_pool, test_set = _load_train_test(Config, device)
        res = run_lowk_cell(Config, device, train_pool, test_set, args.lamlowk, args.seed)
        outdir = Config.RESULTS_DIR / "_lowk_cells"; outdir.mkdir(exist_ok=True)
        torch.save(res, outdir / f"lowk_s{args.lamlowk:g}_seed{args.seed}.pt")
        print(f"[LowkCell] {res}")
    elif args.cmd == "losscell":
        train_pool, test_set = _load_train_test(Config, device)
        res = run_losscell(Config, device, train_pool, test_set, args.tag, args.seed,
                           struct_scale=args.struct_scale, flux_scale=args.flux_scale,
                           logspec_lam=args.logspec, logspec_kcut=args.kcut)
        outdir = Config.RESULTS_DIR / "_loss_cells"; outdir.mkdir(exist_ok=True)
        torch.save(res, outdir / f"{args.tag}_seed{args.seed}.pt")
        print(f"[LossCell] {res}")
    elif args.cmd == "lossagg":
        aggregate_loss_cells(Config)
    elif args.cmd == "sigmacell":
        train_pool, test_set = _load_train_test(Config, device)
        res = run_sigma_cell(Config, device, train_pool, test_set, args.sigma_max, args.seed)
        outdir = Config.RESULTS_DIR / "_sigma_cells"; outdir.mkdir(exist_ok=True)
        torch.save(res, outdir / f"sigma_{args.sigma_max:g}_seed{args.seed}.pt")
        print(f"[SigmaCell] {res}")
    elif args.cmd == "gradflow":
        run_gradflow(Config, device, seeds=args.seeds)
    elif args.cmd == "uqdepth":
        run_uq_depth(Config, device, seeds=args.seeds)
    elif args.cmd == "figs":
        regenerate_figures(Config, device)
    elif args.cmd == "aggregate":
        aggregate_ablations(Config)
    elif args.cmd == "smoke":
        run_smoke(Config, device)


def run_smoke(cfg, device):
    """Fast end-to-end check of the no-training experiments on a single seed/variant."""
    print("[Smoke] mechanism ...")
    run_mechanism(cfg, device, seeds=[cfg.SEEDS[0]], n_sigma=8)
    print("[Smoke] rootcause ...")
    run_rootcause(cfg, device, variants=("vanilla", "wrsg"), n_sigma=6, n_fields=16)
    print("[Smoke] speed ...")
    run_speed(cfg, device, variants=["vanilla", "wrsg"], n_steps=10, n_fields=8, reps=1)
    print("[Smoke] steps ...")
    run_step_ablation(cfg, device, variants=("wrsg",), seeds=[cfg.SEEDS[0]],
                      step_grid=[10, 20], n_eval=24)
    print("[Smoke] ood (1 nu, 1 variant) ...")
    global OOD_INTERP, OOD_EXTRAP
    OOD_INTERP, OOD_EXTRAP = [0.0085], []
    run_ood(cfg, device, variants=["wrsg"], seeds=[cfg.SEEDS[0]], n_snap=24, n_steps=10)
    print("[Smoke] done.")


# ============================================================================
# Second Flow: Passive Scalar Transport
# ============================================================================

KAPPA_LIST = [0.005, 0.010, 0.020, 0.040]
SCALAR_NU = 0.010
SCALAR_G = 1.0
SCALAR_SNAPSHOTS = 500
SCALAR_SPINUP = 8000


# ---------------------------------------------------------------------------
# Passive scalar DNS
# ---------------------------------------------------------------------------

class PassiveScalarKolmogorov(KolmogorovDNS):
    """Kolmogorov velocity field (inherited) plus a passive scalar with an imposed
    mean gradient, integrated jointly with RK4."""

    def __init__(self, N, L, nu, kf, alpha, dt, device, kappa, grad_G=1.0,
                 dealias_jacobian=True):
        super().__init__(N, L, nu, kf, alpha, dt, device, dealias_jacobian)
        self.kappa = kappa
        self.grad_G = grad_G

    def _velocity_hat(self, w_hat):
        """Spectral velocity (u_hat, v_hat) from vorticity via the streamfunction."""
        psi_hat = w_hat * self.inv_k2
        return 1j * self.ky * psi_hat, -1j * self.kx * psi_hat

    def _rhs_scalar(self, w_hat, th_hat):
        """RHS of the scalar fluctuation equation: advection, diffusion, gradient source."""
        u_hat, v_hat = self._velocity_hat(w_hat)
        thx_hat = 1j * self.kx * th_hat
        thy_hat = 1j * self.ky * th_hat
        if self.dealias_jacobian:
            adv = (self._product_dealiased(u_hat, thx_hat) +
                   self._product_dealiased(v_hat, thy_hat)) * self.dealias
        else:
            u = torch.fft.ifft2(u_hat).real
            v = torch.fft.ifft2(v_hat).real
            thx = torch.fft.ifft2(thx_hat).real
            thy = torch.fft.ifft2(thy_hat).real
            adv = torch.fft.fft2(u * thx + v * thy) * self.dealias
        return -adv - self.kappa * self.k2 * th_hat - self.grad_G * v_hat

    def step_rk4_joint(self, w_hat, th_hat):
        """One explicit RK4 step of the coupled (vorticity, scalar) state."""
        def deriv(w, th):
            return self._rhs(w), self._rhs_scalar(w, th)
        k1w, k1t = deriv(w_hat, th_hat)
        k2w, k2t = deriv(w_hat + 0.5 * self.dt * k1w, th_hat + 0.5 * self.dt * k1t)
        k3w, k3t = deriv(w_hat + 0.5 * self.dt * k2w, th_hat + 0.5 * self.dt * k2t)
        k4w, k4t = deriv(w_hat + self.dt * k3w, th_hat + self.dt * k3t)
        w_new = w_hat + (self.dt / 6.0) * (k1w + 2 * k2w + 2 * k3w + k4w)
        th_new = th_hat + (self.dt / 6.0) * (k1t + 2 * k2t + 2 * k3t + k4t)
        return w_new, th_new

    def simulate_scalar(self, n_snapshots, snapshot_interval, spinup_steps, seed,
                        trace_var=False):
        """Spin up the coupled flow, then collect n_snapshots real-space scalar fields."""
        w_hat = self._init_field(seed)
        th_hat = torch.zeros_like(w_hat)
        var_trace = []
        for s in range(spinup_steps):
            w_hat, th_hat = self.step_rk4_joint(w_hat, th_hat)
            if trace_var and (s % 200 == 0):
                th = torch.fft.ifft2(th_hat).real
                var_trace.append(float((th ** 2).mean().item()))
        snaps = torch.zeros(n_snapshots, self.N, self.N, device=self.device)
        for i in range(n_snapshots):
            for _ in range(snapshot_interval):
                w_hat, th_hat = self.step_rk4_joint(w_hat, th_hat)
            snaps[i] = torch.fft.ifft2(th_hat).real
        return (snaps, var_trace) if trace_var else snaps


# ---------------------------------------------------------------------------
# Scalar config (reuses the wrsg training loop unchanged)
# ---------------------------------------------------------------------------

class ScalarConfig(Config):
    """Config for the passive-scalar PoC: kappa regimes replace the viscosity grid, with
    a separate dataset cache and checkpoint/output directories."""
    NU_LIST = KAPPA_LIST
    CKPT_DIR = Config.CKPT_DIR / "scalar"
    CSV_DIR = Config.CSV_DIR
    FIG_DIR = Config.FIG_DIR
    SCALAR = True

    @classmethod
    def dataset_hash(cls):
        import hashlib
        payload = repr({"scalar": True, "nu": SCALAR_NU, "kappa": KAPPA_LIST,
                        "G": SCALAR_G, "grid": cls.GRID, "snaps": SCALAR_SNAPSHOTS,
                        "spinup": SCALAR_SPINUP, "interval": cls.SNAPSHOT_INTERVAL})
        return "scalar_" + hashlib.sha1(payload.encode()).hexdigest()[:10]


def scalar_cache_path(cfg):
    return cfg.DATA_DIR / f"{cfg.dataset_hash()}.pt"


# ---------------------------------------------------------------------------
# Dataset generation + validation
# ---------------------------------------------------------------------------

def validate_scalar(device):
    """Quick physics check: run a short coupled DNS at one kappa, confirm the scalar
    variance saturates, and plot the scalar spectrum. Writes scalar_validation.png."""
    cfg = ScalarConfig
    cfg.setup()
    print(f"[ScalarVal] nu={SCALAR_NU} kappa-grid={KAPPA_LIST} G={SCALAR_G}")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for kappa in KAPPA_LIST:
        dns = PassiveScalarKolmogorov(cfg.GRID, cfg.DOMAIN, SCALAR_NU, cfg.FORCING_K,
                                      cfg.ALPHA_DRAG, cfg.DT, device, kappa, SCALAR_G,
                                      dealias_jacobian=cfg.USE_DEALIASED_JACOBIAN)
        t0 = time.time()
        snaps, var_trace = dns.simulate_scalar(48, cfg.SNAPSHOT_INTERVAL, 6000,
                                               seed=777, trace_var=True)
        Ek, kk = compute_dealiased_spectrum(snaps)
        Ek_m = Ek.mean(0).cpu().numpy(); k = kk.cpu().numpy()
        k_lo, k_hi = cfg.FORCING_K + 2, 20
        slope, intc, r2 = fit_inertial_slope(Ek.mean(0), k_lo, k_hi)
        nz = (k > 0) & (Ek_m > 0)
        axes[0].plot(np.arange(len(var_trace)) * 200, var_trace, label=f"kappa={kappa}")
        axes[1].loglog(k[nz], Ek_m[nz], label=f"kappa={kappa}  slope={slope:.2f} (R2={r2:.2f})")
        print(f"[ScalarVal] kappa={kappa}: var_final={var_trace[-1]:.3f} "
              f"slope={slope:.2f} R2={r2:.2f} ({time.time()-t0:.0f}s)")
    axes[0].set_xlabel("step"); axes[0].set_ylabel("scalar variance <theta^2>")
    axes[0].set_title("Variance saturation (spinup)"); axes[0].legend(fontsize=8)
    axes[1].set_xlabel("k"); axes[1].set_ylabel("scalar spectrum")
    axes[1].set_title("Passive scalar spectra"); axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / "scalar_validation.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[ScalarVal] wrote {cfg.FIG_DIR/'scalar_validation.png'}")


def generate_scalar_dataset(device):
    """Generate (or load) the passive-scalar dataset across kappa regimes; normalize and cache."""
    cfg = ScalarConfig
    cfg.setup()
    cache = scalar_cache_path(cfg)
    if cache.exists():
        print(f"[ScalarData] cached: {cache.name}")
        return torch.load(cache, map_location="cpu")
    all_fields, all_re, diags = [], [], []
    for r, kappa in enumerate(KAPPA_LIST):
        print(f"[ScalarData] regime {r+1}/{len(KAPPA_LIST)} kappa={kappa}", flush=True)
        dns = PassiveScalarKolmogorov(cfg.GRID, cfg.DOMAIN, SCALAR_NU, cfg.FORCING_K,
                                      cfg.ALPHA_DRAG, cfg.DT, device, kappa, SCALAR_G,
                                      dealias_jacobian=cfg.USE_DEALIASED_JACOBIAN)
        t0 = time.time()
        snaps = dns.simulate_scalar(SCALAR_SNAPSHOTS, cfg.SNAPSHOT_INTERVAL,
                                    SCALAR_SPINUP, seed=3000 + r)
        var = float((snaps ** 2).mean().item())
        diags.append({"regime": r, "kappa": kappa, "variance": var})
        all_fields.append(snaps.cpu())
        all_re.append(torch.full((SCALAR_SNAPSHOTS,), float(r)))
        print(f"[ScalarData]   variance={var:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    fields = torch.cat(all_fields, 0)
    re_labels = torch.cat(all_re, 0).long()
    mean, std = fields.mean(), fields.std()
    payload = {"fields": ((fields - mean) / (std + 1e-8)).unsqueeze(1).float(),
               "re_labels": re_labels, "raw_mean": mean.item(), "raw_std": std.item(),
               "kappa_list": KAPPA_LIST, "grid": cfg.GRID, "diagnostics": diags,
               "hash": cfg.dataset_hash()}
    torch.save(payload, cache)
    print(f"[ScalarData] saved -> {cache.name}  ({fields.shape[0]} fields)")
    return payload


def scalar_splits(device):
    """Build the scalar train pool and fixed test holdout (same protocol as the paper)."""
    cfg = ScalarConfig
    payload = generate_scalar_dataset(device)
    full = TurbulenceDataset(payload["fields"], payload["re_labels"])
    tr_idx, te_idx = prepare_splits(full, cfg)
    return Subset(full, tr_idx), Subset(full, te_idx), payload


# ---------------------------------------------------------------------------
# Scalar evaluation (LSD, structure functions, integral length -- no fluxes)
# ---------------------------------------------------------------------------

SCALAR_VARIANTS = ["vanilla", "wrsg_gate", "wrsg"]


def evaluate_scalar(model, sched, test_set, cfg, device, variant, seed, n_eval=192):
    """Sample per kappa-regime and score the field-agnostic metrics."""
    nu_lut = make_nu_cond_lut(cfg.NU_LIST, device)
    loader = DataLoader(test_set, batch_size=64, shuffle=False)
    preds, trues, res = [], [], []
    with torch.no_grad():
        for x, re in loader:
            x = x.to(device); re = re.to(device)
            for r in range(len(cfg.NU_LIST)):
                idx = (re == r)
                if idx.sum() == 0:
                    continue
                n = int(idx.sum().item())
                gens = sample_dpmpp_2m(model, sched, n, float(nu_lut[r]), cfg.GRID, device,
                                       n_steps=cfg.N_SAMPLE_STEPS, noise_seed=seed * 1000 + r)
                preds.append(gens); trues.append(x[idx])
                res.append(torch.full((n,), r, device=device))
            if sum(p.shape[0] for p in preds) >= n_eval:
                break
    pred = torch.cat(preds, 0)[:n_eval]; true = torch.cat(trues, 0)[:n_eval]
    agg = {"lsd_aggregate": metric_lsd_aggregate(pred, true),
           "vorticity_S2_log_rmse": metric_vorticity_structure_log_rmse(pred, true, 2),
           "vorticity_S3_log_rmse": metric_vorticity_structure_log_rmse(pred, true, 3),
           "integral_length_rel_err": metric_integral_length_rel_err(pred, true)}
    return agg


def run_scalar_cell(variant, seed, device, epochs=None, lam_scale=1.0, profile="full"):
    """Train+eval one scalar PoC model (variant, seed) and save its metrics. lam_scale
    multiplies every physics-loss weight (the default weights were tuned on the vorticity
    field). profile selects which physics losses are active for the scalar: "full" keeps all,
    "spectral_only" keeps only the field-agnostic spectral term (dropping the vorticity
    structure/enstrophy/integral-length terms that over-sharpen the scalar), and
    "spectral_lowk" adds the dense low-wavenumber spectral term. lam_scale=1.0 with the full
    profile is the paper setting; other cells are saved under _scalar_retune."""
    cfg = ScalarConfig
    cfg.setup()
    cfg.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    if epochs is not None:
        cfg.EPOCHS = epochs
    base = dict(LAMBDA_ENSTROPHY=cfg.LAMBDA_ENSTROPHY, LAMBDA_SPECTRAL=cfg.LAMBDA_SPECTRAL,
                LAMBDA_STRUCT=cfg.LAMBDA_STRUCT, LAMBDA_INTLEN=cfg.LAMBDA_INTLEN,
                LAMBDA_LOWK=cfg.LAMBDA_LOWK)
    base_ckpt = cfg.CKPT_DIR
    tag = ""
    if profile in ("spectral_only", "spectral_lowk"):
        cfg.LAMBDA_ENSTROPHY = cfg.LAMBDA_STRUCT = cfg.LAMBDA_INTLEN = 0.0
        cfg.LAMBDA_SPECTRAL = base["LAMBDA_SPECTRAL"] * lam_scale
        cfg.LAMBDA_LOWK = 0.3 * lam_scale if profile == "spectral_lowk" else 0.0
        tag = f"_{profile}_s{lam_scale:g}"
    elif lam_scale != 1.0:
        for k in ("LAMBDA_ENSTROPHY", "LAMBDA_SPECTRAL", "LAMBDA_STRUCT", "LAMBDA_INTLEN"):
            setattr(cfg, k, base[k] * lam_scale)
        tag = f"_s{lam_scale:g}"
    if tag:
        cfg.CKPT_DIR = base_ckpt / f"retune{tag}"
        cfg.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    train_pool, test_set, _ = scalar_splits(device)
    use_phys = variant in ("wrsg_phys", "wrsg", "wrsg_fno")
    sched = EDMSchedule(cfg.SIGMA_MIN, cfg.SIGMA_MAX, cfg.SIGMA_DATA, cfg.RHO,
                          cfg.P_MEAN, cfg.P_STD, device)
    model, sched, _, compute = train_one_model(variant, seed, cfg, train_pool, device,
                                               use_physics=use_phys, use_flux=False)
    agg = evaluate_scalar(model, sched, test_set, cfg, device, variant, seed)
    agg.update({"variant": variant, "seed": seed, "lam_scale": lam_scale,
                "profile": profile, "n_params": compute["n_params"]})
    for k, v in base.items():
        setattr(cfg, k, v)
    cfg.CKPT_DIR = base_ckpt
    if tag:
        outdir = cfg.RESULTS_DIR / "_scalar_retune"; outdir.mkdir(parents=True, exist_ok=True)
        torch.save(agg, outdir / f"{variant}{tag}_seed{seed}.pt")
    else:
        outdir = cfg.RESULTS_DIR / "_scalar_cells"; outdir.mkdir(parents=True, exist_ok=True)
        torch.save(agg, outdir / f"{variant}_seed{seed}.pt")
    print(f"[ScalarCell] {variant} seed{seed} lam={lam_scale:g} prof={profile}: "
          f"LSD={agg['lsd_aggregate']:.3f} S2={agg['vorticity_S2_log_rmse']:.3f} "
          f"intL={agg['integral_length_rel_err']:.3f}")
    return agg


def aggregate_scalar():
    """Collect all scalar-cell results into scalar_poc_metrics.csv with seed means/CIs."""
    cfg = ScalarConfig
    outdir = cfg.RESULTS_DIR / "_scalar_cells"
    files = sorted(outdir.glob("*.pt"))
    rows = [torch.load(f, map_location="cpu") for f in files]
    df = pd.DataFrame(rows)
    df.to_csv(cfg.CSV_DIR / "scalar_poc_raw.csv", index=False)
    metrics = ["lsd_aggregate", "vorticity_S2_log_rmse", "vorticity_S3_log_rmse",
               "integral_length_rel_err"]
    summ = df.groupby("variant")[metrics].agg(["mean", "std"]).reset_index()
    summ.columns = ["_".join(c).strip("_") for c in summ.columns]
    summ.to_csv(cfg.CSV_DIR / "scalar_poc_metrics.csv", index=False)
    print(summ.to_string(index=False))
    print("[ScalarAgg] wrote scalar_poc_metrics.csv")
    return summ


def aggregate_scalar_retune():
    """Collect the scalar retune/profile cells against the wrsg_gate (no losses) and wrsg
    (full vorticity weights) baselines into scalar_retune.csv: scalar LSD per loss
    configuration. The full-profile scale sweep shows the vorticity weights are field-specific
    (less is better); the spectral-only and spectral+low-k profiles test whether a
    field-appropriate loss beats the gate-only transfer."""
    cfg = ScalarConfig
    def label(profile, lam):
        if profile in ("spectral_only", "spectral_lowk"):
            return f"{profile}_s{lam:g}"
        return f"full_lam{lam:g}"
    rows = []
    for f in sorted((cfg.RESULTS_DIR / "_scalar_retune").glob("*.pt")):
        d = torch.load(f, map_location="cpu")
        prof = d.get("profile", "full")
        rows.append({"config": label(prof, float(d["lam_scale"])), "profile": prof,
                     "lam_scale": float(d["lam_scale"]), "seed": int(d["seed"]),
                     "lsd_aggregate": d["lsd_aggregate"],
                     "integral_length_rel_err": d["integral_length_rel_err"]})
    bdir = cfg.RESULTS_DIR / "_scalar_cells"
    for f in sorted(bdir.glob("wrsg_gate_seed*.pt")):
        d = torch.load(f, map_location="cpu")
        rows.append({"config": "gate_only", "profile": "none", "lam_scale": 0.0,
                     "seed": int(d["seed"]), "lsd_aggregate": d["lsd_aggregate"],
                     "integral_length_rel_err": d["integral_length_rel_err"]})
    for f in sorted(bdir.glob("wrsg_seed*.pt")):
        d = torch.load(f, map_location="cpu")
        rows.append({"config": "full_lam1", "profile": "full", "lam_scale": 1.0,
                     "seed": int(d["seed"]), "lsd_aggregate": d["lsd_aggregate"],
                     "integral_length_rel_err": d["integral_length_rel_err"]})
    df = pd.DataFrame(rows)
    df.to_csv(cfg.CSV_DIR / "scalar_retune_raw.csv", index=False)
    summ = df.groupby("config").agg(
        n=("seed", "count"),
        lsd_mean=("lsd_aggregate", "mean"), lsd_std=("lsd_aggregate", "std"),
        intlen_mean=("integral_length_rel_err", "mean")).reset_index().sort_values("lsd_mean")
    summ.to_csv(cfg.CSV_DIR / "scalar_retune.csv", index=False)
    print(summ.to_string(index=False))
    print("[ScalarRetune] wrote scalar_retune.csv")
    return summ


def _secondflow_main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["validate", "gendata", "traincell", "aggregate",
                                   "retuneagg"])
    p.add_argument("--variant", default="wrsg")
    p.add_argument("--seed", type=int, default=29)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--profile", default="full",
                   choices=["full", "spectral_only", "spectral_lowk"])
    p.add_argument("--epochs", type=int, default=None)
    args = p.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[SecondFlow] cmd={args.cmd} device={device}")
    if args.cmd == "validate":
        validate_scalar(device)
    elif args.cmd == "gendata":
        generate_scalar_dataset(device)
    elif args.cmd == "traincell":
        run_scalar_cell(args.variant, args.seed, device, epochs=args.epochs,
                        lam_scale=args.scale, profile=args.profile)
    elif args.cmd == "aggregate":
        aggregate_scalar()
    elif args.cmd == "retuneagg":
        aggregate_scalar_retune()


# ============================================================================
# Third Flow: 2D Cellular Forcing
# ============================================================================

CELL_NU_LIST = [0.005, 0.010, 0.020]
CELL_FORCING_K = 6
CELL_SNAPSHOTS = 800
CELL_VARIANTS = ["vanilla", "wrsg_gate", "wrsg"]


class CellularForcedDNS(KolmogorovDNS):
    """2D Navier-Stokes vorticity solver with a 2D cellular forcing
    f = k_f^2 (cos(k_f x) + cos(k_f y)) replacing the 1D Kolmogorov shear forcing."""

    def __init__(self, N, L, nu, kf, alpha, dt, device, dealias_jacobian=True):
        super().__init__(N, L, nu, kf, alpha, dt, device, dealias_jacobian)
        x = torch.linspace(0, L, N + 1, device=device)[:-1]
        X, Y = torch.meshgrid(x, x, indexing="ij")
        self.forcing = (kf ** 2) * (torch.cos(kf * X) + torch.cos(kf * Y))
        self.forcing_hat = torch.fft.fft2(self.forcing)


class CellConfig(Config):
    """Config for the cellular-forcing flow: a different forcing wavenumber and viscosity
    grid, with separate dataset cache and checkpoint directories."""
    NU_LIST = CELL_NU_LIST
    FORCING_K = CELL_FORCING_K
    CKPT_DIR = Config.CKPT_DIR / "cellular"

    @classmethod
    def dataset_hash(cls):
        import hashlib
        payload = repr({"cellular": True, "nu": CELL_NU_LIST, "kf": CELL_FORCING_K,
                        "grid": cls.GRID, "snaps": CELL_SNAPSHOTS,
                        "spinup": cls.SPINUP_STEPS, "interval": cls.SNAPSHOT_INTERVAL})
        return "cellular_" + hashlib.sha1(payload.encode()).hexdigest()[:10]


def cell_cache_path(cfg):
    return cfg.DATA_DIR / f"{cfg.dataset_hash()}.pt"


def validate_cell(device):
    """Run a short DNS at each viscosity, confirm the flow develops, and plot the
    vorticity spectra. Writes cellular_validation.png."""
    cfg = CellConfig
    cfg.setup()
    print(f"[CellVal] cellular forcing kf={CELL_FORCING_K}, nu grid={CELL_NU_LIST}")
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for nu in CELL_NU_LIST:
        dns = CellularForcedDNS(cfg.GRID, cfg.DOMAIN, nu, CELL_FORCING_K, cfg.ALPHA_DRAG,
                                cfg.DT, device, dealias_jacobian=cfg.USE_DEALIASED_JACOBIAN)
        t0 = time.time()
        snaps = dns.simulate(48, cfg.SNAPSHOT_INTERVAL, cfg.SPINUP_STEPS, seed=555)
        Ek, kk = compute_dealiased_spectrum(snaps)
        slope, _, r2 = fit_inertial_slope(Ek.mean(0), CELL_FORCING_K + 2, 20)
        diag = dns.diagnostics(snaps[-32:])
        k = kk.cpu().numpy(); Em = Ek.mean(0).cpu().numpy(); nz = (k > 0) & (Em > 0)
        ax.loglog(k[nz], Em[nz], lw=1.8,
                  label=f"nu={nu} Re_int={diag['Re_int']:.0f} slope={slope:.2f} (R2={r2:.2f})")
        print(f"[CellVal] nu={nu}: Re_int={diag['Re_int']:.0f} slope={slope:.2f} "
              f"R2={r2:.2f} enstrophy={diag['enstrophy']:.2f} ({time.time()-t0:.0f}s)", flush=True)
    ax.axvline(CELL_FORCING_K, color="orange", ls=":", lw=1.0, label=f"$k_f={CELL_FORCING_K}$")
    ax.set_xlabel("k"); ax.set_ylabel(r"$|\hat{\omega}(k)|^2$")
    ax.set_title(f"Cellular-forcing DNS spectra (N={cfg.GRID}, $k_f$={CELL_FORCING_K})")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / "cellular_validation.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[CellVal] wrote {cfg.FIG_DIR/'cellular_validation.png'}")


def generate_cell_dataset(device):
    """Generate (or load) the cellular-forcing dataset across viscosity regimes."""
    cfg = CellConfig
    cfg.setup()
    cache = cell_cache_path(cfg)
    if cache.exists():
        print(f"[CellData] cached: {cache.name}")
        return torch.load(cache, map_location="cpu")
    all_fields, all_re, diags = [], [], []
    for r, nu in enumerate(CELL_NU_LIST):
        print(f"[CellData] regime {r+1}/{len(CELL_NU_LIST)} nu={nu}", flush=True)
        dns = CellularForcedDNS(cfg.GRID, cfg.DOMAIN, nu, CELL_FORCING_K, cfg.ALPHA_DRAG,
                                cfg.DT, device, dealias_jacobian=cfg.USE_DEALIASED_JACOBIAN)
        t0 = time.time()
        snaps = dns.simulate(CELL_SNAPSHOTS, cfg.SNAPSHOT_INTERVAL, cfg.SPINUP_STEPS,
                             seed=4000 + r)
        diag = dns.diagnostics(snaps[-256:]); diag["nu"] = nu; diag["regime"] = r
        diags.append(diag)
        all_fields.append(snaps.cpu())
        all_re.append(torch.full((CELL_SNAPSHOTS,), float(r)))
        print(f"[CellData]   Re_int={diag['Re_int']:.0f} ({time.time()-t0:.0f}s)", flush=True)
    fields = torch.cat(all_fields, 0); re_labels = torch.cat(all_re, 0).long()
    mean, std = fields.mean(), fields.std()
    payload = {"fields": ((fields - mean) / (std + 1e-8)).unsqueeze(1).float(),
               "re_labels": re_labels, "raw_mean": mean.item(), "raw_std": std.item(),
               "nu_list": CELL_NU_LIST, "grid": cfg.GRID, "diagnostics": diags,
               "hash": cfg.dataset_hash()}
    torch.save(payload, cache)
    print(f"[CellData] saved -> {cache.name} ({fields.shape[0]} fields)")
    return payload


def cell_splits(device):
    cfg = CellConfig
    payload = generate_cell_dataset(device)
    full = TurbulenceDataset(payload["fields"], payload["re_labels"])
    tr_idx, te_idx = prepare_splits(full, cfg)
    return Subset(full, tr_idx), Subset(full, te_idx)


def evaluate_cell(model, sched, test_set, cfg, device, seed, n_eval=192):
    """Score the eight forced-turbulence physics metrics (cascade split at the new k_f)."""
    nu_lut = make_nu_cond_lut(cfg.NU_LIST, device)
    loader = DataLoader(test_set, batch_size=64, shuffle=False)
    preds, trues, res = [], [], []
    with torch.no_grad():
        for x, re in loader:
            x = x.to(device); re = re.to(device)
            for r in range(len(cfg.NU_LIST)):
                idx = (re == r)
                if idx.sum() == 0:
                    continue
                n = int(idx.sum().item())
                gens = sample_dpmpp_2m(model, sched, n, float(nu_lut[r]), cfg.GRID, device,
                                       n_steps=cfg.N_SAMPLE_STEPS, noise_seed=seed * 1000 + r)
                preds.append(gens); trues.append(x[idx]); res.append(torch.full((n,), r, device=device))
            if sum(p.shape[0] for p in preds) >= n_eval:
                break
    pred = torch.cat(preds, 0)[:n_eval]; true = torch.cat(trues, 0)[:n_eval]
    return {"lsd_aggregate": metric_lsd_aggregate(pred, true),
            "vorticity_S2_log_rmse": metric_vorticity_structure_log_rmse(pred, true, 2),
            "vorticity_S3_log_rmse": metric_vorticity_structure_log_rmse(pred, true, 3),
            "integral_length_rel_err": metric_integral_length_rel_err(pred, true),
            "energy_flux_rmse": metric_energy_flux_rmse(pred, true),
            "enstrophy_flux_rmse": metric_enstrophy_flux_rmse(pred, true),
            "inverse_energy_cascade_recovery_pct": metric_inverse_energy_cascade_recovery(pred, true, kf=cfg.FORCING_K),
            "forward_enstrophy_cascade_recovery_pct": metric_forward_enstrophy_cascade_recovery(pred, true, kf=cfg.FORCING_K)}


def run_cell(variant, seed, device, epochs=None):
    """Train+eval one cellular-flow model (variant, seed) and save its metrics."""
    cfg = CellConfig
    cfg.setup(); cfg.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    if epochs is not None:
        cfg.EPOCHS = epochs
    train_pool, test_set = cell_splits(device)
    use_phys = variant in ("wrsg_phys", "wrsg", "wrsg_fno")
    use_flux = variant in ("wrsg", "wrsg_fno")
    model, sched, _, compute = train_one_model(variant, seed, cfg, train_pool, device,
                                               use_physics=use_phys, use_flux=use_flux)
    agg = evaluate_cell(model, sched, test_set, cfg, device, seed)
    agg.update({"variant": variant, "seed": seed, "n_params": compute["n_params"]})
    outdir = cfg.RESULTS_DIR / "_cell_cells"; outdir.mkdir(parents=True, exist_ok=True)
    torch.save(agg, outdir / f"{variant}_seed{seed}.pt")
    print(f"[CellCell] {variant} seed{seed}: LSD={agg['lsd_aggregate']:.3f} "
          f"Efl={agg['energy_flux_rmse']:.3f} FwdZ={agg['forward_enstrophy_cascade_recovery_pct']:.0f}%")
    return agg


def aggregate_cell():
    cfg = CellConfig
    files = sorted((cfg.RESULTS_DIR / "_cell_cells").glob("*.pt"))
    rows = [torch.load(f, map_location="cpu") for f in files]
    df = pd.DataFrame(rows)
    df.to_csv(cfg.CSV_DIR / "cellular_poc_raw.csv", index=False)
    metrics = ["lsd_aggregate", "vorticity_S2_log_rmse", "vorticity_S3_log_rmse",
               "integral_length_rel_err", "energy_flux_rmse", "enstrophy_flux_rmse",
               "inverse_energy_cascade_recovery_pct", "forward_enstrophy_cascade_recovery_pct"]
    summ = df.groupby("variant")[metrics].agg(["mean", "std"]).reset_index()
    summ.columns = ["_".join(c).strip("_") for c in summ.columns]
    summ.to_csv(cfg.CSV_DIR / "cellular_poc_metrics.csv", index=False)
    print(summ.to_string(index=False))
    print("[CellAgg] wrote cellular_poc_metrics.csv")
    return summ


def _thirdflow_main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["validate", "gendata", "traincell", "aggregate"])
    p.add_argument("--variant", default="wrsg")
    p.add_argument("--seed", type=int, default=29)
    p.add_argument("--epochs", type=int, default=None)
    args = p.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[ThirdFlow] cmd={args.cmd} device={device}")
    if args.cmd == "validate":
        validate_cell(device)
    elif args.cmd == "gendata":
        generate_cell_dataset(device)
    elif args.cmd == "traincell":
        run_cell(args.variant, args.seed, device, epochs=args.epochs)
    elif args.cmd == "aggregate":
        aggregate_cell()


# ============================================================================
# Resolution Transfer (M2): the same Kolmogorov flow at a different grid
# ============================================================================

class Res64Config(Config):
    """The main Kolmogorov flow at a 64x64 grid (16x16 bottleneck) with the gate's bin
    count scaled to the smaller bottleneck by the fixed-fraction rule (8 bins for the
    ~11 radial shells of a 16^2 grid, matching 16 bins for the ~22 shells at 32^2)."""
    GRID = 64
    GATE_N_BINS = 8
    NU_LIST = [0.010, 0.020, 0.040]
    SNAPSHOTS_PER_REGIME = 800
    CKPT_DIR = Config.CKPT_DIR / "res64"


def run_res_cell(variant, seed, device, epochs=None):
    """Train+eval one 64^2 model (variant, seed); reuses the main DNS, training loop, and
    forced-flow eight-metric evaluation, with the resolution-scaled gate."""
    cfg = Res64Config
    cfg.setup(); cfg.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    if epochs is not None:
        cfg.EPOCHS = epochs
    payload = generate_dataset(cfg, device)
    full = TurbulenceDataset(payload["fields"], payload["re_labels"])
    tr_idx, te_idx = prepare_splits(full, cfg)
    train_pool, test_set = Subset(full, tr_idx), Subset(full, te_idx)
    use_phys = variant in ("wrsg_phys", "wrsg", "wrsg_fno")
    use_flux = variant in ("wrsg", "wrsg_fno")
    model, sched, _, compute = train_one_model(variant, seed, cfg, train_pool, device,
                                               use_physics=use_phys, use_flux=use_flux)
    agg = evaluate_cell(model, sched, test_set, cfg, device, seed)
    agg.update({"variant": variant, "seed": seed, "n_params": compute["n_params"],
                "gate_bins": cfg.GATE_N_BINS})
    outdir = cfg.RESULTS_DIR / "_res_cells"; outdir.mkdir(parents=True, exist_ok=True)
    torch.save(agg, outdir / f"{variant}_seed{seed}.pt")
    print(f"[ResCell] {variant} seed{seed} (64^2, {cfg.GATE_N_BINS} bins): "
          f"LSD={agg['lsd_aggregate']:.3f} Efl={agg['energy_flux_rmse']:.3f}")
    return agg


def aggregate_res():
    """Collect the 64^2 resolution-transfer cells into res64_poc_metrics.csv."""
    cfg = Res64Config
    files = sorted((cfg.RESULTS_DIR / "_res_cells").glob("*.pt"))
    rows = [torch.load(f, map_location="cpu") for f in files]
    df = pd.DataFrame(rows); df.to_csv(cfg.CSV_DIR / "res64_poc_raw.csv", index=False)
    metrics = ["lsd_aggregate", "vorticity_S2_log_rmse", "energy_flux_rmse",
               "inverse_energy_cascade_recovery_pct", "forward_enstrophy_cascade_recovery_pct"]
    summ = df.groupby("variant")[metrics].agg(["mean", "std"]).reset_index()
    summ.columns = ["_".join(c).strip("_") for c in summ.columns]
    summ.to_csv(cfg.CSV_DIR / "res64_poc_metrics.csv", index=False)
    print(summ.to_string(index=False)); print("[ResAgg] wrote res64_poc_metrics.csv")
    return summ


def _resolution_main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["gendata", "traincell", "aggregate"])
    p.add_argument("--variant", default="wrsg")
    p.add_argument("--seed", type=int, default=29)
    p.add_argument("--epochs", type=int, default=None)
    args = p.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Resolution] cmd={args.cmd} device={device}")
    if args.cmd == "gendata":
        generate_dataset(Res64Config, device)
    elif args.cmd == "traincell":
        run_res_cell(args.variant, args.seed, device, epochs=args.epochs)
    elif args.cmd == "aggregate":
        aggregate_res()


# ============================================================================
# Downstream Task (M7/M10): sparse-observation reconstruction with a generative prior
# ============================================================================

@torch.no_grad()
def assimilate(model, sched, x_true, mask, obs_noise_std, nu_cond, sched_cfg, n_steps, seed):
    """Reconstruct a full field from sparse noisy observations using the viscosity-conditioned
    diffusion model as a prior (RePaint-style replacement: at each reverse step the observed
    pixels are pinned to the noised observation, the rest is filled by the prior)."""
    sigmas = sched.build_inference_schedule(n_steps)
    g = torch.Generator(device=x_true.device).manual_seed(seed)
    y = x_true + obs_noise_std * torch.randn(x_true.shape, device=x_true.device, generator=g)
    x = torch.randn(x_true.shape, device=x_true.device, generator=g) * sigmas[0]
    cond = torch.full((x_true.shape[0],), float(nu_cond), device=x_true.device)
    for i in range(n_steps):
        sig = torch.full((x_true.shape[0],), sigmas[i].item(), device=x_true.device)
        x0 = denoise_preconditioned(model, x, sig, cond, sched)
        snext = sigmas[i + 1]
        if snext == 0:
            x = x0 * (1 - mask) + y * mask
        else:
            x = x0 + (snext / sigmas[i]) * (x - x0)
            x_obs = y + snext * torch.randn(x.shape, device=x.device, generator=g)
            x = x * (1 - mask) + x_obs * mask
    return x


def run_downstream(cfg, device, variants=("wrsg", "vanilla"), seed=None,
                   n_fields=48, fracs=(0.10, 0.25, 0.50), obs_noise=0.02, regime=3, n_ens=8):
    """Sparse-reconstruction (data-assimilation) downstream test: reconstruct held-out DNS
    fields from a fraction of noisy pixel observations, with each model as the prior, and
    score reconstruction RMSE and spectral fidelity. A mean-fill baseline bounds triviality.
    With n_ens>1 the posterior-mean reconstruction (the RMSE-optimal point estimate, averaged
    over n_ens posterior draws) is reported alongside the single-draw one. Writes
    downstream.csv and downstream.png."""
    seed = seed or cfg.SEEDS[0]
    payload = torch.load(cfg.DATA_DIR / f"kolmogorov_{cfg.dataset_hash()}.pt", map_location="cpu")
    fields = payload["fields"]; re = payload["re_labels"]
    full = TurbulenceDataset(fields, re)
    _, test_idx = prepare_splits(full, cfg)
    sel = [i for i in test_idx if int(re[i]) == regime][:n_fields]
    x_true = fields[sel].to(device)
    ncond = nu_cond_value(cfg.NU_LIST[regime], cfg.NU_LIST)
    sched = build_sched(cfg, device)
    rows = []
    g = torch.Generator(device=device).manual_seed(7)
    for frac in fracs:
        mask = (torch.rand(x_true.shape, device=device, generator=g) < frac).float()
        meanfill = x_true * mask
        rmse_mean = float(((meanfill - x_true)[mask == 0] ** 2).mean().sqrt().item())
        rows.append({"variant": "mean-fill", "frac": frac, "recon_rmse": rmse_mean,
                     "recon_lsd": metric_lsd_aggregate(meanfill, x_true)})
        for variant in variants:
            model = load_eval_model(variant, seed, cfg, device)
            draws = [assimilate(model, sched, x_true, mask, obs_noise, ncond, cfg,
                                cfg.N_SAMPLE_STEPS, seed=seed * 13 + int(frac * 100) + j * 101)
                     for j in range(n_ens)]
            recons = [(variant, draws[0])]
            if n_ens > 1:
                recons.append((f"{variant}-mean", torch.stack(draws, 0).mean(0)))
            for label, recon in recons:
                rmse = float(((recon - x_true)[mask == 0] ** 2).mean().sqrt().item())
                rows.append({"variant": label, "frac": frac, "recon_rmse": rmse,
                             "recon_lsd": metric_lsd_aggregate(recon, x_true)})
                print(f"[Downstream] {label:12s} frac={frac:.2f}: RMSE={rmse:.3f} "
                      f"LSD={rows[-1]['recon_lsd']:.3f}", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    df = pd.DataFrame(rows); df.to_csv(cfg.CSV_DIR / "downstream.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for variant in df.variant.unique():
        d = df[df.variant == variant]
        axes[0].plot(d.frac, d.recon_rmse, "o-", lw=2, label=variant)
        axes[1].plot(d.frac, d.recon_lsd, "o-", lw=2, label=variant)
    axes[0].set_xlabel("observed fraction", fontsize=12); axes[0].set_ylabel("reconstruction RMSE", fontsize=12)
    axes[1].set_xlabel("observed fraction", fontsize=12); axes[1].set_ylabel("reconstruction LSD", fontsize=12)
    for a in axes:
        a.legend(fontsize=11); a.tick_params(labelsize=11)
    axes[0].set_title("Sparse reconstruction error", fontsize=12)
    axes[1].set_title("Reconstruction spectral fidelity", fontsize=12)
    fig.tight_layout(); fig.savefig(cfg.FIG_DIR / "downstream.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[Downstream] wrote downstream.csv and downstream.png")
    return df


def _downstream_main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=29)
    p.add_argument("--n_fields", type=int, default=48)
    p.add_argument("--n_ens", type=int, default=8)
    args = p.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Downstream] device={device}")
    Config.setup()
    run_downstream(Config, device, seed=args.seed, n_fields=args.n_fields, n_ens=args.n_ens)


# ============================================================================
# Unified command-line entry point
# ============================================================================

def _dispatch():
    """Route to the reviewer-experiment / second-flow / third-flow sub-mains by the first
    positional token, or to the main training sweep otherwise."""
    import sys
    ns = sys.argv[1] if len(sys.argv) > 1 else ""
    if ns == "reviews":
        del sys.argv[1]; _reviews_main()
    elif ns == "secondflow":
        del sys.argv[1]; _secondflow_main()
    elif ns == "thirdflow":
        del sys.argv[1]; _thirdflow_main()
    elif ns == "resolution":
        del sys.argv[1]; _resolution_main()
    elif ns == "downstream":
        del sys.argv[1]; _downstream_main()
    else:
        main()


if __name__ == "__main__":
    _dispatch()
