"""Robust evaluation of the deep-ensemble / conformal UQ improvements, with bootstrap CIs.

Loads the cached large ensembles (Results/_uqens/<variant>_seed*.pt) and compares, on the
disjoint TEST split with all calibration fit on CAL only:
  - single model (mean +/- seed spread over the 5 seeds)
  - deep ensemble (5 seeds pooled, 160 members) with a bootstrap CI over test samples
for three quantities that were the paper's UQ negatives: small-scale (k>20) spectral coverage
under the multiplicative alpha_k calibration and under split-conformal, per-pixel calibrated
coverage, and CRPS. No good metric is touched.

Usage: python code/uq_final_eval.py --variant wrsg --gpu 0
"""
import argparse
import numpy as np
import torch

from WRSG import (Config, compute_radial_spectrum, calibrate_alpha, coverage_calibrated,
                  metric_coverage_band, metric_crps_ensemble)


def spectra_chunked(fields, device, chunk=2048):
    """compute_radial_spectrum over a big (N,H,W) stack in chunks; returns (N, nshells), kk."""
    outs = []
    kk = None
    for i in range(0, fields.shape[0], chunk):
        E, kk = compute_radial_spectrum(fields[i:i+chunk].to(device).float())
        outs.append(E.cpu())
    return torch.cat(outs, 0), kk


def shell_metrics(E_ens, E_true, cal_i, tst_i, sel, q_level):
    """Per-(test-sample, shell) coverage indicators for the multiplicative and conformal bands
    (calibration fit on cal). Returns I_mult, I_conf each (n_test, len(sel))."""
    n_t = len(tst_i)
    I_mult = np.zeros((n_t, len(sel)), dtype=np.float32)
    I_conf = np.zeros((n_t, len(sel)), dtype=np.float32)
    for c, j in enumerate(sel):
        e = E_ens[:, :, j]
        t = E_true[:, j]
        lo = torch.quantile(e, 0.05, dim=0); hi = torch.quantile(e, 0.95, dim=0)
        med = torch.quantile(e, 0.50, dim=0)
        half_lo = (med - lo).clamp(min=0); half_hi = (hi - med).clamp(min=0)
        a = calibrate_alpha(e[:, cal_i], t[cal_i], target_cov=0.90)
        cov = ((t >= med - a * half_lo) & (t <= med + a * half_hi)).float()
        I_mult[:, c] = cov[tst_i].cpu().numpy()
        resid = (t - med).abs()
        q = torch.quantile(resid[cal_i], q_level)
        I_conf[:, c] = (resid[tst_i] <= q).float().cpu().numpy()
    return I_mult, I_conf


def pixel_cov_per_sample(ens, true, cal_i, tst_i):
    """Per-test-sample per-pixel calibrated coverage vector (scalar alpha fit on cal)."""
    lo = torch.quantile(ens, 0.05, dim=0); hi = torch.quantile(ens, 0.95, dim=0)
    med = torch.quantile(ens, 0.50, dim=0)
    half_lo = (med - lo).clamp(min=0); half_hi = (hi - med).clamp(min=0)
    a = calibrate_alpha(ens[:, cal_i], true[cal_i], target_cov=0.90)
    inb = ((true >= med - a * half_lo) & (true <= med + a * half_hi)).float()
    return inb[tst_i].mean(dim=(1, 2, 3)).cpu().numpy(), a


def boot_ci(per_sample_2d, B=2000, seed=0):
    """Bootstrap 95% CI of the grand mean by resampling rows (test samples)."""
    rng = np.random.default_rng(seed)
    n = per_sample_2d.shape[0]
    means = np.empty(B)
    flat = per_sample_2d.reshape(n, -1)
    for b in range(B):
        idx = rng.integers(0, n, n)
        means[b] = flat[idx].mean()
    return float(flat.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="wrsg")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--k_lo", type=int, default=20)
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    cfg = Config
    seeds = cfg.SEEDS
    packs = {s: torch.load(cfg.RESULTS_DIR / "_uqens" / f"{args.variant}_seed{s}.pt",
                           map_location="cpu") for s in seeds}
    true = packs[seeds[0]]["true"].float()
    n_uq = true.shape[0]
    n_cal = n_uq // 2
    perm = torch.randperm(n_uq, generator=torch.Generator().manual_seed(0))
    cal_i, tst_i = perm[:n_cal], perm[n_cal:]
    q_level = min(float(np.ceil((n_cal + 1) * 0.90) / n_cal), 1.0)

    E_true, kk = spectra_chunked(true.squeeze(1), device)
    E_true = E_true.to(device)
    sel = [j for j, kv in enumerate(kk.tolist()) if kv > args.k_lo]
    cal_g, tst_g = cal_i.to(device), tst_i.to(device)
    true_g = true.to(device)

    E_ens = {}
    for s in seeds:
        ens = packs[s]["ens"].float()
        M = ens.shape[0]
        Ef, _ = spectra_chunked(ens.reshape(M * n_uq, cfg.GRID, cfg.GRID), device)
        E_ens[s] = Ef.reshape(M, n_uq, -1).to(device)
    print(f"[{args.variant}] n_uq={n_uq} n_cal={n_cal} n_test={len(tst_i)} shells(k>{args.k_lo})={len(sel)}",
          flush=True)

    sm = {"mult": [], "conf": [], "pix": [], "crps": []}
    for s in seeds:
        Im, Ic = shell_metrics(E_ens[s], E_true, cal_g, tst_g, sel, q_level)
        sm["mult"].append(Im.mean()); sm["conf"].append(Ic.mean())
        ens_f = packs[s]["ens"].float().to(device)
        pc, _ = pixel_cov_per_sample(ens_f, true_g, cal_g, tst_g)
        sm["pix"].append(pc.mean())
        sm["crps"].append(metric_crps_ensemble(ens_f[:, tst_g], true_g[tst_g]))
        del ens_f
        torch.cuda.empty_cache() if device.type == "cuda" else None
    def ms(x): return f"{np.mean(x):.3f} +/- {np.std(x):.3f}"
    print(f"[single-model]  k>20 mult={ms(sm['mult'])}  conformal={ms(sm['conf'])}  "
          f"pixel={ms(sm['pix'])}  CRPS={ms(sm['crps'])}", flush=True)

    E_deep = torch.cat([E_ens[s] for s in seeds], dim=0)
    Im, Ic = shell_metrics(E_deep, E_true, cal_g, tst_g, sel, q_level)
    mult_m, mult_lo, mult_hi = boot_ci(Im)
    conf_m, conf_lo, conf_hi = boot_ci(Ic)
    ens_deep = torch.cat([packs[s]["ens"].float() for s in seeds], dim=0).to(device)
    pc, _ = pixel_cov_per_sample(ens_deep, true_g, cal_g, tst_g)
    pix_m, pix_lo, pix_hi = boot_ci(pc[:, None])
    crps_deep = metric_crps_ensemble(ens_deep[:, tst_g], true_g[tst_g])
    print(f"[deep-160]      k>20 mult={mult_m:.3f} [{mult_lo:.3f},{mult_hi:.3f}]  "
          f"conformal={conf_m:.3f} [{conf_lo:.3f},{conf_hi:.3f}]  "
          f"pixel={pix_m:.3f} [{pix_lo:.3f},{pix_hi:.3f}]  CRPS={crps_deep:.3f}", flush=True)

    rows = [
        {"config": "single_model", "k20_mult": round(float(np.mean(sm['mult'])), 4),
         "k20_mult_sd": round(float(np.std(sm['mult'])), 4),
         "k20_conf": round(float(np.mean(sm['conf'])), 4),
         "pixel_cal": round(float(np.mean(sm['pix'])), 4),
         "crps": round(float(np.mean(sm['crps'])), 4)},
        {"config": "deep_160", "k20_mult": round(mult_m, 4),
         "k20_mult_ci_lo": round(mult_lo, 4), "k20_mult_ci_hi": round(mult_hi, 4),
         "k20_conf": round(conf_m, 4), "k20_conf_ci_lo": round(conf_lo, 4),
         "k20_conf_ci_hi": round(conf_hi, 4),
         "pixel_cal": round(pix_m, 4), "crps": round(crps_deep, 4)},
    ]
    import pandas as pd
    df = pd.DataFrame(rows); df.insert(0, "variant", args.variant)
    out = cfg.CSV_DIR / f"uq_final_{args.variant}.csv"
    df.to_csv(out, index=False)
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
