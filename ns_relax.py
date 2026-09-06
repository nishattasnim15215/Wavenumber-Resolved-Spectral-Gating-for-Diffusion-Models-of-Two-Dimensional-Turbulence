"""Short-time Navier-Stokes relaxation as a physics-consistency post-processor (WRSG+NS).

WRSG generates a vorticity snapshot with a good spectrum but imperfect spectral-flux (cascade)
transport, because a soft loss cannot enforce the conservation law. Integrating the TRUE forced-
dissipative 2D Navier-Stokes dynamics for a short time tau from a generated field injects the
physical information the score lacks: the nonlinear advection redistributes energy/enstrophy
toward the dynamically consistent flux profile. We report the relaxation trajectory over tau on
the disjoint TEST split as a characterization (not a selected operating point), together with a
true-field relaxation as a solver-faithfulness check. This produces a NEW hybrid variant; the
canonical deterministic pred and its headline metrics are never overwritten. The transform needs
only the known viscosity and forcing (no test labels), so it is also usable out of distribution.

Writes ns_relax_<variant>_seed<seed>.csv. Usage: python code/ns_relax.py --variant wrsd --seed 29 --gpu 0
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from WRSG import (Config, load_eval_model, build_sched, load_payload, prepare_splits,
                  make_nu_cond_lut, TurbulenceDataset, KolmogorovDNS, sample_dpmpp_2m,
                  metric_lsd_aggregate, metric_vorticity_structure_log_rmse,
                  metric_inverse_energy_cascade_recovery, metric_forward_enstrophy_cascade_recovery)


def gen_pred(model, sched, cfg, device, test_set, seed):
    """Reproduce the headline deterministic pred fields (noise_seed=seed*1000+r) + regime labels."""
    nu_lut = make_nu_cond_lut(cfg.NU_LIST, device)
    loader = DataLoader(test_set, batch_size=64, shuffle=False)
    P, T, R = [], [], []
    with torch.no_grad():
        for x, re in loader:
            x, re = x.to(device), re.to(device)
            for r in range(len(cfg.NU_LIST)):
                idx = (re == r)
                if idx.sum() == 0:
                    continue
                n = int(idx.sum().item())
                g = sample_dpmpp_2m(model, sched, n, float(nu_lut[r]), cfg.GRID, device,
                                    n_steps=cfg.N_SAMPLE_STEPS, noise_seed=seed * 1000 + r)
                P.append(g); T.append(x[idx]); R.append(torch.full((n,), r, device=device))
            if sum(p.shape[0] for p in P) >= cfg.N_EVAL_SAMPLES:
                break
    return (torch.cat(P)[:cfg.N_EVAL_SAMPLES], torch.cat(T)[:cfg.N_EVAL_SAMPLES],
            torch.cat(R)[:cfg.N_EVAL_SAMPLES])


def relax(fields_norm, re_arr, cfg, device, raw_mean, raw_std, n_relax):
    """Integrate each field's physical vorticity forward n_relax RK4 steps under its regime's NS."""
    if n_relax == 0:
        return fields_norm
    out = fields_norm.clone()
    for r in range(len(cfg.NU_LIST)):
        idx = (re_arr == r).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        phys = fields_norm[idx].squeeze(1) * raw_std + raw_mean
        dns = KolmogorovDNS(cfg.GRID, cfg.DOMAIN, cfg.NU_LIST[r], cfg.FORCING_K,
                            cfg.ALPHA_DRAG, cfg.DT, device)
        w_hat = torch.fft.fft2(phys)
        for _ in range(n_relax):
            w_hat = dns.step_rk4(w_hat)
        relaxed = torch.fft.ifft2(w_hat).real
        out[idx] = ((relaxed - raw_mean) / raw_std).unsqueeze(1)
    return out


def metrics(pred, true):
    return {
        "lsd": metric_lsd_aggregate(pred, true),
        "s3":  metric_vorticity_structure_log_rmse(pred, true, order=3),
        "inv": metric_inverse_energy_cascade_recovery(pred, true, kf=Config.FORCING_K),
        "fwd": metric_forward_enstrophy_cascade_recovery(pred, true, kf=Config.FORCING_K),
    }


def casc_err(m):
    return abs(m["inv"] - 100.0) + abs(m["fwd"] - 100.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="wrsd")
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--grid", default="0,2,5,10,20,40")
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    cfg = Config
    payload = load_payload(cfg)
    raw_mean, raw_std = payload["raw_mean"], payload["raw_std"]
    full = TurbulenceDataset(payload["fields"], payload["re_labels"])
    _, test_idx = prepare_splits(full, cfg)
    test_set = Subset(full, test_idx)
    sched = build_sched(cfg, device)
    model = load_eval_model(args.variant, args.seed, cfg, device)
    pred, true, re = gen_pred(model, sched, cfg, device, test_set, args.seed)

    n = pred.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed)).to(device)
    cal, tst = perm[:n // 2], perm[n // 2:]
    taus = [int(t) for t in args.grid.split(",")]

    import pandas as pd
    rows = []
    for t in taus:
        rl = relax(pred, re, cfg, device, raw_mean, raw_std, t)
        mc, mt = metrics(rl[cal], true[cal]), metrics(rl[tst], true[tst])
        trl = relax(true, re, cfg, device, raw_mean, raw_std, t)
        ts = metrics(trl[tst], true[tst])
        rows.append({"variant": args.variant, "seed": args.seed, "tau": t,
                     "inv": round(mt["inv"], 2), "fwd": round(mt["fwd"], 2),
                     "lsd": round(mt["lsd"], 4), "s3": round(mt["s3"], 4),
                     "casc_err": round(casc_err(mt), 2),
                     "cal_lsd": round(mc["lsd"], 4), "cal_s3": round(mc["s3"], 4),
                     "true_relax_inv": round(ts["inv"], 2), "true_relax_fwd": round(ts["fwd"], 2),
                     "true_relax_lsd": round(ts["lsd"], 4)})
        print(f"[tau={t:4d}] TEST inv={mt['inv']:.1f} fwd={mt['fwd']:.1f} "
              f"casc_err={casc_err(mt):.1f} lsd={mt['lsd']:.4f} s3={mt['s3']:.4f}  "
              f"(true-relax inv/fwd={ts['inv']:.1f}/{ts['fwd']:.1f} lsd={ts['lsd']:.4f})", flush=True)
    out = cfg.CSV_DIR / f"ns_relax_{args.variant}_seed{args.seed}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
