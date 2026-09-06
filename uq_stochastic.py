"""Stochastic-ensemble uncertainty quantification for WRSG.

The headline UQ ensemble is drawn with the deterministic DPM-Solver++(2M) ODE, whose
members differ only in their initial latent noise. That map contracts diversity, worst at
the small scales, so the raw ensemble is under-dispersed (per-pixel coverage 0.72-0.85; the
band power at k>20 is covered only 0.23 of the time) and a single scalar variance rescaling
cannot recover the missing small-scale spread.

This script re-draws the UQ ensemble with the EDM stochastic sampler (Karras et al. 2022,
Algorithm 2): each step injects fresh noise controlled by S_churn, which restores
scale-appropriate spread. Everything is eval-only on the existing trained checkpoints; the
headline point metrics (LSD, flux, cascade) are produced by a separate deterministic
generation and are not touched. S_churn is chosen on the calibration split (like the scalar
alpha) and every reported number is on the disjoint test split, so there is no test leakage.

Writes uq_stochastic_<variant>_seed<seed>.csv (one row: ODE baseline vs stochastic, on test).
Usage: python code/uq_stochastic.py --variant wrsd --seed 29 --gpu 4
"""
import argparse
import torch

from WRSG import (
    Config, load_eval_model, build_sched, load_payload, prepare_splits,
    make_nu_cond_lut, TurbulenceDataset, denoise_preconditioned, compute_radial_spectrum,
    calibrate_alpha, coverage_calibrated, metric_coverage_band, metric_crps_ensemble,
    sample_dpmpp_2m,
)
from torch.utils.data import DataLoader, Subset


@torch.no_grad()
def sample_edm_stochastic(model, sched, n, nu_cond, image_size, device, n_steps,
                          s_churn=40.0, s_tmin=0.05, s_tmax=20.0, s_noise=1.003,
                          noise_seed=None):
    """EDM stochastic sampler (Karras 2022, Alg. 2): Heun step with a per-step churn that
    lifts sigma by gamma and injects noise, restoring ensemble spread the ODE contracts.
    s_churn=0 recovers deterministic Heun; larger s_churn gives a more dispersed ensemble."""
    sigmas = sched.build_inference_schedule(n_steps)
    g = (torch.Generator(device=device).manual_seed(int(noise_seed))
         if noise_seed is not None else None)
    x = torch.randn(n, 1, image_size, image_size, device=device, generator=g) * sigmas[0]
    cond = torch.full((n,), float(nu_cond), device=device)
    root2m1 = 2.0 ** 0.5 - 1.0
    for i in range(n_steps):
        sigma = float(sigmas[i]); sigma_next = float(sigmas[i + 1])
        gamma = min(s_churn / n_steps, root2m1) if (s_tmin <= sigma <= s_tmax) else 0.0
        sigma_hat = sigma * (1.0 + gamma)
        if gamma > 0.0:
            eps = torch.randn(n, 1, image_size, image_size, device=device,
                              generator=g) * s_noise
            x = x + eps * (sigma_hat ** 2 - sigma ** 2) ** 0.5
        sh_b = torch.full((n,), sigma_hat, device=device)
        denoised = denoise_preconditioned(model, x, sh_b, cond, sched)
        d = (x - denoised) / sigma_hat
        if sigma_next == 0.0:
            x = x + (sigma_next - sigma_hat) * d
        else:
            x_euler = x + (sigma_next - sigma_hat) * d
            sn_b = torch.full((n,), sigma_next, device=device)
            denoised2 = denoise_preconditioned(model, x_euler, sn_b, cond, sched)
            d2 = (x_euler - denoised2) / sigma_next
            x = x + (sigma_next - sigma_hat) * 0.5 * (d + d2)
    return x


def build_true_uq(cfg, device):
    """Reconstruct the exact true_uq / re_uq the headline evaluate() uses for UQ: iterate
    the fixed test loader, group by regime, take the first N_EVAL_SAMPLES then the first 64."""
    payload = load_payload(cfg)
    full = TurbulenceDataset(payload["fields"], payload["re_labels"])
    _, test_idx = prepare_splits(full, cfg)
    test_set = Subset(full, test_idx)
    loader = DataLoader(test_set, batch_size=64, shuffle=False,
                        pin_memory=(device.type == "cuda"))
    all_true, all_re = [], []
    for x, re in loader:
        x = x.to(device); re = re.to(device)
        for r in range(len(cfg.NU_LIST)):
            idx = (re == r)
            if idx.sum() == 0:
                continue
            all_true.append(x[idx])
            all_re.append(torch.full((int(idx.sum().item()),), r, device=device))
        if sum(t.shape[0] for t in all_true) >= cfg.N_EVAL_SAMPLES:
            break
    true = torch.cat(all_true, 0)[:cfg.N_EVAL_SAMPLES]
    re_arr = torch.cat(all_re, 0)[:cfg.N_EVAL_SAMPLES]
    n_uq = min(64, true.shape[0])
    return true[:n_uq], re_arr[:n_uq]


def gen_ensemble(model, sched, cfg, device, true_uq, re_uq, nu_lut, seed, kind, s_churn):
    """Draw an M=N_ENSEMBLE ensemble aligned with true_uq, using the ODE or stochastic sampler."""
    n_uq = true_uq.shape[0]
    ens = torch.zeros(cfg.N_ENSEMBLE, n_uq, 1, cfg.GRID, cfg.GRID, device=device)
    for m in range(cfg.N_ENSEMBLE):
        for r in range(len(cfg.NU_LIST)):
            idx = (re_uq == r)
            if idx.sum() == 0:
                continue
            ns = seed * 100000 + m * 1000 + r
            if kind == "ode":
                gens = sample_dpmpp_2m(model, sched, int(idx.sum().item()), float(nu_lut[r]),
                                       cfg.GRID, device, n_steps=cfg.N_SAMPLE_STEPS,
                                       noise_seed=ns)
            else:
                gens = sample_edm_stochastic(model, sched, int(idx.sum().item()),
                                             float(nu_lut[r]), cfg.GRID, device,
                                             n_steps=cfg.N_SAMPLE_STEPS, s_churn=s_churn,
                                             noise_seed=ns)
            ens[m][idx.nonzero(as_tuple=False).flatten()] = gens
    return ens


def spectral_coverage(ens, truth, k_lo, k_hi, alphas=None):
    """Ensemble band coverage of DNS band power over shells k in (k_lo, k_hi]. If alphas is
    given (per-shell scale from calibration) the band is alpha-scaled; else raw (alpha=1)."""
    M, m = ens.shape[0], ens.shape[1]
    E_ens_flat, kk = compute_radial_spectrum(ens.reshape(M * m, ens.shape[-1], ens.shape[-1]))
    E_ens = E_ens_flat.reshape(M, m, -1)
    E_true, _ = compute_radial_spectrum(truth.squeeze(1))
    sel = [j for j, kv in enumerate(kk.tolist()) if k_lo < kv <= k_hi]
    covs = []
    for j in sel:
        a = 1.0 if alphas is None else alphas.get(j, 1.0)
        covs.append(coverage_calibrated(E_ens[:, :, j], E_true[:, j], a))
    return sum(covs) / max(len(covs), 1), sel, kk, E_ens, E_true


def fit_shell_alphas(E_ens_cal, E_true_cal, sel):
    """Per-shell calibration scale alpha_k fit on the calibration split, for shells in sel."""
    alphas = {}
    for j in sel:
        alphas[j] = calibrate_alpha(E_ens_cal[:, :, j], E_true_cal[:, j], target_cov=0.90)
    return alphas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="wrsd")
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--churn_grid", default="0,10,20,40,80")
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    cfg = Config
    torch.cuda.set_device(device) if device.type == "cuda" else None

    sched = build_sched(cfg, device)
    nu_lut = make_nu_cond_lut(cfg.NU_LIST, device)
    true_uq, re_uq = build_true_uq(cfg, device)
    n_uq = true_uq.shape[0]
    model = load_eval_model(args.variant, args.seed, cfg, device)

    n_cal = max(2, int(cfg.UQ_CAL_FRACTION * n_uq))
    perm = torch.randperm(n_uq, generator=torch.Generator().manual_seed(args.seed)).to(device)
    cal_i, tst_i = perm[:n_cal], perm[n_cal:]
    K_LO, K_HI = 20, 10 ** 9

    def metrics_on(ens, s_churn_val):
        y = true_uq
        ens_c, ens_t = ens[:, cal_i], ens[:, tst_i]
        y_c, y_t = y[cal_i], y[tst_i]
        alpha = calibrate_alpha(ens_c, y_c, target_cov=0.90)
        _, sel, _, E_ens, E_true = spectral_coverage(ens, y, K_LO, K_HI)
        sh_alphas = fit_shell_alphas(E_ens[:, cal_i], E_true[cal_i], sel)
        def spec_cov(members_idx, alphas):
            covs = [coverage_calibrated(E_ens[:, members_idx, j], E_true[members_idx, j],
                                        1.0 if alphas is None else alphas.get(j, 1.0))
                    for j in sel]
            return sum(covs) / max(len(covs), 1)
        return {
            "s_churn": s_churn_val,
            "pixel_raw_test":  metric_coverage_band(ens_t, y_t),
            "pixel_cal_test":  coverage_calibrated(ens_t, y_t, alpha),
            "alpha": alpha,
            "crps_test":       metric_crps_ensemble(ens_t, y_t),
            "smallk_raw_test": spec_cov(tst_i, None),
            "smallk_cal_test": spec_cov(tst_i, sh_alphas),
            "smallk_cal_CAL":  spec_cov(cal_i, sh_alphas),
        }

    rows = []
    ens_ode = gen_ensemble(model, sched, cfg, device, true_uq, re_uq, nu_lut, args.seed, "ode", 0.0)
    r_ode = metrics_on(ens_ode, -1.0); r_ode["kind"] = "ode"; rows.append(r_ode)
    print(f"[ODE ] pix_raw={r_ode['pixel_raw_test']:.3f} pix_cal={r_ode['pixel_cal_test']:.3f} "
          f"crps={r_ode['crps_test']:.3f} k>20_raw={r_ode['smallk_raw_test']:.3f} "
          f"k>20_cal={r_ode['smallk_cal_test']:.3f}", flush=True)

    best = None
    for sc in [float(x) for x in args.churn_grid.split(",")]:
        ens_s = gen_ensemble(model, sched, cfg, device, true_uq, re_uq, nu_lut, args.seed,
                             "stoch", sc)
        r = metrics_on(ens_s, sc); r["kind"] = "stoch"; rows.append(r)
        print(f"[Sc={sc:5.1f}] pix_raw={r['pixel_raw_test']:.3f} pix_cal={r['pixel_cal_test']:.3f} "
              f"crps={r['crps_test']:.3f} k>20_raw={r['smallk_raw_test']:.3f} "
              f"k>20_cal={r['smallk_cal_test']:.3f}  (CAL k>20_cal={r['smallk_cal_CAL']:.3f})",
              flush=True)
        gap_cal = abs(r["smallk_cal_CAL"] - 0.90)
        if best is None or gap_cal < best[0]:
            best = (gap_cal, r)

    sel_row = dict(best[1]); sel_row["kind"] = "stoch_selected"
    rows.append(sel_row)
    print(f"[SELECTED on CAL] S_churn={sel_row['s_churn']}  -> TEST k>20_cal="
          f"{sel_row['smallk_cal_test']:.3f} pixel_cal={sel_row['pixel_cal_test']:.3f} "
          f"crps={sel_row['crps_test']:.3f}", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.insert(0, "variant", args.variant); df.insert(1, "seed", args.seed)
    out = cfg.CSV_DIR / f"uq_stochastic_{args.variant}_seed{args.seed}.csv"
    df.to_csv(out, index=False)
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
