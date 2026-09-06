"""Generate and cache one (variant, seed) deterministic-ODE UQ ensemble at a large sample
size, so the deep-ensemble / calibration study can be evaluated with tight bootstrap CIs.

Saves Results/_uqens/<variant>_seed<seed>.pt = {ens(fp16, M x n_uq x 1 x H x W), true, re}.
The true fields / regime labels are the fixed test set (identical across seeds and variants).
Eval-only on existing checkpoints; nothing about the headline point metrics is touched.

Usage: python code/uq_gen_ens.py --variant wrsd --seed 29 --gpu 0 --n_uq 256
"""
import argparse
import torch
from torch.utils.data import DataLoader, Subset

from WRSG import (Config, load_eval_model, build_sched, make_nu_cond_lut, load_payload,
                  prepare_splits, TurbulenceDataset)
from uq_stochastic import gen_ensemble


def build_true_uq_n(cfg, device, n_uq):
    payload = load_payload(cfg)
    full = TurbulenceDataset(payload["fields"], payload["re_labels"])
    _, test_idx = prepare_splits(full, cfg)
    loader = DataLoader(Subset(full, test_idx), batch_size=64, shuffle=False)
    all_true, all_re = [], []
    target = max(cfg.N_EVAL_SAMPLES, n_uq)
    for x, re in loader:
        x, re = x.to(device), re.to(device)
        for r in range(len(cfg.NU_LIST)):
            idx = (re == r)
            if idx.sum() == 0:
                continue
            all_true.append(x[idx])
            all_re.append(torch.full((int(idx.sum().item()),), r, device=device))
        if sum(t.shape[0] for t in all_true) >= target:
            break
    true = torch.cat(all_true, 0)[:n_uq]
    re_arr = torch.cat(all_re, 0)[:n_uq]
    return true, re_arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="wrsd")
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n_uq", type=int, default=256)
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    cfg = Config
    sched = build_sched(cfg, device)
    nu_lut = make_nu_cond_lut(cfg.NU_LIST, device)
    true_uq, re_uq = build_true_uq_n(cfg, device, args.n_uq)
    model = load_eval_model(args.variant, args.seed, cfg, device)
    ens = gen_ensemble(model, sched, cfg, device, true_uq, re_uq, nu_lut, args.seed, "ode", 0.0)
    out = cfg.RESULTS_DIR / "_uqens" / f"{args.variant}_seed{args.seed}.pt"
    torch.save({"ens": ens.half().cpu(), "true": true_uq.half().cpu(),
                "re": re_uq.cpu().to(torch.int16), "n_uq": args.n_uq}, out)
    print(f"[gen-ens] {args.variant}/seed{args.seed}: ens{tuple(ens.shape)} -> {out}", flush=True)


if __name__ == "__main__":
    main()
