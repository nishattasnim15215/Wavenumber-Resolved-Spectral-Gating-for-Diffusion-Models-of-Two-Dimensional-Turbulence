"""Scale-resolved uncertainty recalibration for WRSG (reviewer-driven improvement of the UQ
result). The paper's single global variance scale reaches 0.85-0.89 pixel coverage because the
per-pixel ensemble is under-dispersed in a SCALE-DEPENDENT way. Here we fit a per-wavenumber
inflation of the ensemble's deviation spectrum on a CALIBRATION split and a residual global
scale, then evaluate pixel coverage on a DISJOINT TEST split. It is ethical/sound only if the
gain appears on the held-out conditions -- which this script reports. Fully isolated: imports
the verified WRSG machinery, writes only uq_improve.csv / uq_improve.png, retrains nothing.

Usage: python code/uq_improve.py [--gpu ID] [--n_cond N]
"""

import argparse

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from WRSG import (
    Config, load_payload, prepare_splits, TurbulenceDataset, make_nu_cond_lut, build_sched,
    sample_dpmpp_2m, load_eval_model, metric_coverage_band, calibrate_alpha, coverage_calibrated,
)

VARIANT = "wrsd"
TARGET = 0.90


def shell_map(H, W, device):
    """Integer radial-wavenumber shell index for every 2D Fourier coefficient."""
    ky = torch.fft.fftfreq(H, d=1.0 / H).to(device)
    kx = torch.fft.fftfreq(W, d=1.0 / W).to(device)
    KY, KX = torch.meshgrid(ky, kx, indexing="ij")
    kbin = torch.sqrt(KX ** 2 + KY ** 2).round().long()
    return kbin, int(kbin.max().item()) + 1


def fit_beta(ens_cal, truth_cal, kbin, K):
    """Per-shell inflation factor beta_k = sqrt( <|F(truth-mean)|^2>_k / <|F(ens-mean)|^2>_k ),
    fit on the calibration conditions. beta_k>1 widens an under-dispersed shell. Robust: each
    beta_k averages over all members, calibration conditions, and Fourier coeffs in the shell."""
    mean = ens_cal.mean(0)
    dev_pow = torch.fft.fft2(ens_cal - mean, norm="ortho").abs().pow(2)
    err_pow = torch.fft.fft2(truth_cal - mean, norm="ortho").abs().pow(2)
    kf = kbin.reshape(-1)
    dev_flat = dev_pow.reshape(-1, kf.numel()).mean(0)
    err_flat = err_pow.reshape(-1, kf.numel()).mean(0)
    beta = torch.ones(K, device=ens_cal.device)
    for k in range(K):
        m_k = (kf == k)
        if m_k.any():
            dp = dev_flat[m_k].mean().clamp(min=1e-12)
            ep = err_flat[m_k].mean().clamp(min=1e-12)
            beta[k] = (ep / dp).sqrt()
    return beta.clamp(min=0.5, max=5.0)


def inflate(ens, beta, kbin):
    """Inflate each ensemble member's deviation from the ensemble mean by beta_k in Fourier
    space (beta real and radially symmetric -> the inverse transform stays real)."""
    mean = ens.mean(0)
    dev_hat = torch.fft.fft2(ens - mean, norm="ortho") * beta[kbin]
    return mean + torch.fft.ifft2(dev_hat, norm="ortho").real


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n_cond", type=int, default=24)
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[UQimprove] device={device}", flush=True)
    payload = load_payload(Config)
    fields = payload["fields"]; re = payload["re_labels"]
    _, test_idx = prepare_splits(TurbulenceDataset(fields, re), Config)
    nu_lut = make_nu_cond_lut(Config.NU_LIST, device)
    sched = build_sched(Config, device)
    model = load_eval_model(VARIANT, Config.SEEDS[0], Config, device)
    kbin, K = shell_map(Config.GRID, Config.GRID, device)
    n_ens = Config.N_ENSEMBLE
    rows = []
    for r in range(len(Config.NU_LIST)):
        sel = [i for i in test_idx if int(re[i]) == r][:args.n_cond]
        if len(sel) < 6:
            continue
        truth = fields[sel].to(device); m = len(sel)
        ens = torch.stack([sample_dpmpp_2m(model, sched, m, float(nu_lut[r]), Config.GRID, device,
                                           n_steps=Config.N_SAMPLE_STEPS, noise_seed=Config.SEEDS[0]*97000+j*131+r)
                           for j in range(n_ens)], 0)
        n_cal = m // 2
        ec, et = ens[:, :n_cal], ens[:, n_cal:]
        yc, yt = truth[:n_cal], truth[n_cal:]
        cov_raw = metric_coverage_band(et, yt)
        a1 = calibrate_alpha(ec, yc, TARGET)
        cov_single = coverage_calibrated(et, yt, a1)
        beta = fit_beta(ec, yc, kbin, K)
        ec_i, et_i = inflate(ec, beta, kbin), inflate(et, beta, kbin)
        a2 = calibrate_alpha(ec_i, yc, TARGET)
        cov_scale = coverage_calibrated(et_i, yt, a2)
        rows.append({"regime": r, "nu": Config.NU_LIST[r], "n_test": m - n_cal,
                     "coverage_raw": cov_raw, "coverage_single_alpha": cov_single,
                     "coverage_scale_resolved": cov_scale, "alpha_single": a1, "alpha_resid": a2})
        print(f"[UQimprove] nu={Config.NU_LIST[r]}: raw={cov_raw:.3f}  single-a={cov_single:.3f}  "
              f"scale-resolved={cov_scale:.3f}", flush=True)
    df = pd.DataFrame(rows)
    Config.CSV_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(Config.CSV_DIR / "uq_improve.csv", index=False)
    ov = df[["coverage_raw", "coverage_single_alpha", "coverage_scale_resolved"]].mean()
    print(f"[UQimprove] OVERALL (held-out test): raw={ov['coverage_raw']:.3f}  "
          f"single-alpha={ov['coverage_single_alpha']:.3f}  "
          f"scale-resolved={ov['coverage_scale_resolved']:.3f}", flush=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(df.nu, df.coverage_raw, "o-", label="raw band", lw=2)
    ax.plot(df.nu, df.coverage_single_alpha, "s-", label="single global scale (paper)", lw=2)
    ax.plot(df.nu, df.coverage_scale_resolved, "D-", label="scale-resolved calibration", lw=2)
    ax.axhline(TARGET, color="k", ls="--", lw=1, label="0.90 target")
    ax.set_xlabel(r"viscosity $\nu$"); ax.set_ylabel(r"$90\%$ pixel coverage (held-out)")
    ax.set_title("Scale-resolved UQ recalibration (WRSG)"); ax.legend(fontsize=9)
    fig.tight_layout(); Config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(Config.FIG_DIR / "uq_improve.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print("[UQimprove] wrote uq_improve.csv, uq_improve.png", flush=True)


if __name__ == "__main__":
    main()
