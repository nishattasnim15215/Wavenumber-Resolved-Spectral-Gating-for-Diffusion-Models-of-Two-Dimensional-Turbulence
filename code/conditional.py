"""Conditional next-step diffusion emulator for the reviewer-gJ33 distributional comparison.

Fully isolated from the main pipeline: separate data cache, checkpoint directory, CSV, and
figure; nothing in WRSG.py is modified. This trains a conditional (previous-state -> next-state)
EDM diffusion U-Net at a single viscosity regime, rolls it out autoregressively to a
statistically stationary state, and compares its steady-state spectrum and spectral fluxes
against DNS and the unconditional WRSG model. The conditional model reuses the paper's verified
vanilla U-Net blocks, EDM preconditioning, DPM-Solver++ sampler, DNS solver, and metrics.

Usage: python code/conditional.py [--smoke] [--epochs N] [--gpu ID]
"""

import argparse
import math
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from WRSG import (
    Config, KolmogorovDNS, ConditionEmbedding, ResidualBlock, DownsampleBlock,
    UpsampleBlock, EMA, adaptive_ema_decay, build_sched, load_payload, load_eval_model,
    nu_cond_value, sample_dpmpp_2m, compute_dealiased_spectrum, compute_energy_flux,
    compute_enstrophy_flux, metric_lsd_aggregate, metric_energy_flux_rmse,
    metric_enstrophy_flux_rmse, metric_inverse_energy_cascade_recovery,
    metric_forward_enstrophy_cascade_recovery,
)

REGIME_IDX = 3
DELTA_STEPS = 25
N_TRAJ = 8
SNAPS_PER_TRAJ = 320
ROLL_CHAINS = 64
ROLL_STEPS = 96
ROLL_BURNIN = 32
ROLL_COLLECT_EVERY = 8
TRAIN_SEED = 29
CKPT_DIR = Config.CKPT_DIR / "conditional"
DATA_CACHE = Config.DATA_DIR / f"conditional_traj_nu{REGIME_IDX}_d{DELTA_STEPS}.pt"
CSV_OUT = Config.CSV_DIR / "conditional_comparison.csv"
FIG_OUT = Config.FIG_DIR / "conditional_comparison.png"


class ConditionalDenoiser(nn.Module):
    """Vanilla U-Net EDM denoiser conditioned on the previous state by input concatenation:
    the input is [c_in * noised_next, prev_state] (2 channels), the output is the 1-channel
    denoised next state. Same blocks and capacity as the paper's vanilla baseline plus one
    extra input channel, so it is a fair standard conditional next-step emulator."""

    def __init__(self, base_ch=48, emb_dim=128):
        super().__init__()
        self.emb = ConditionEmbedding(emb_dim)
        self.in_conv = nn.Conv2d(2, base_ch, 3, padding=1)
        self.b1 = ResidualBlock(base_ch,     base_ch,     emb_dim)
        self.d1 = DownsampleBlock(base_ch)
        self.b2 = ResidualBlock(base_ch,     base_ch * 2, emb_dim)
        self.d2 = DownsampleBlock(base_ch * 2)
        self.b3 = ResidualBlock(base_ch * 2, base_ch * 4, emb_dim)
        self.u2 = UpsampleBlock(base_ch * 4)
        self.b4 = ResidualBlock(base_ch * 4 + base_ch * 2, base_ch * 2, emb_dim)
        self.u1 = UpsampleBlock(base_ch * 2)
        self.b5 = ResidualBlock(base_ch * 2 + base_ch,     base_ch,     emb_dim)
        self.out_conv = nn.Conv2d(base_ch, 1, 3, padding=1)

    def forward(self, x2, c_noise, nu_cond):
        emb = self.emb(c_noise, nu_cond)
        h1 = self.b1(self.in_conv(x2), emb)
        h2 = self.b2(self.d1(h1), emb)
        h3 = self.b3(self.d2(h2), emb)
        u2 = self.b4(torch.cat([self.u2(h3), h2], dim=1), emb)
        u1 = self.b5(torch.cat([self.u1(u2), h1], dim=1), emb)
        return self.out_conv(u1)


def denoise_cond(model, x_noised, prev, sigma, nu_cond, sched):
    """Conditional EDM denoise, D(next; sigma | prev) = c_skip*next_noised
    + c_out * f([c_in*next_noised, prev], c_noise, nu). The prev state is clean conditioning."""
    c_skip, c_out, c_in, c_noise = sched.preconditioning(sigma)
    cs = c_skip.view(-1, 1, 1, 1); co = c_out.view(-1, 1, 1, 1); ci = c_in.view(-1, 1, 1, 1)
    inp = torch.cat([ci * x_noised, prev], dim=1)
    return cs * x_noised + co * model(inp, c_noise, nu_cond)


@torch.no_grad()
def sample_cond_next(model, sched, prev, nu_cond, image_size, device, n_steps, gen):
    """DPM-Solver++(2M) sampling of the next state conditioned on prev (batched over chains)."""
    model.eval()
    B = prev.shape[0]
    sigmas = sched.build_inference_schedule(n_steps)
    x = torch.randn(B, 1, image_size, image_size, device=device, generator=gen) * sigmas[0]
    cond = torch.full((B,), float(nu_cond), device=device)
    old = None
    for i in range(n_steps):
        sigma = sigmas[i]; sigma_next = sigmas[i + 1]
        sigma_b = torch.full((B,), sigma.item(), device=device)
        denoised = denoise_cond(model, x, prev, sigma_b, cond, sched)
        if sigma_next == 0:
            x = denoised
        else:
            t = -sigma.log(); t_next = -sigma_next.log(); h = t_next - t
            if old is None:
                x = (sigma_next / sigma) * x - (-h).expm1() * denoised
            else:
                r = (t - (-sigmas[i - 1].log())) / h
                D = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old
                x = (sigma_next / sigma) * x - (-h).expm1() * D
            old = denoised
    return x


def generate_traj(device, smoke=False):
    """Generate short-interval DNS trajectory pairs at nu=NU_LIST[REGIME_IDX], normalized with
    the main dataset's global mean/std so the fields share the paper's data scale."""
    if DATA_CACHE.exists() and not smoke:
        print(f"[CondData] cached: {DATA_CACHE.name}", flush=True)
        return torch.load(DATA_CACHE, map_location="cpu")
    payload = load_payload(Config)
    mean, std = payload["raw_mean"], payload["raw_std"]
    nu = Config.NU_LIST[REGIME_IDX]
    n_traj = 2 if smoke else N_TRAJ
    n_snap = 40 if smoke else SNAPS_PER_TRAJ
    prevs, nexts = [], []
    for j in range(n_traj):
        dns = KolmogorovDNS(Config.GRID, Config.DOMAIN, nu, Config.FORCING_K, Config.ALPHA_DRAG,
                            Config.DT, device, dealias_jacobian=Config.USE_DEALIASED_JACOBIAN)
        snaps = dns.simulate(n_snap, DELTA_STEPS, Config.SPINUP_STEPS, seed=7000 + j)
        snaps = ((snaps.cpu() - mean) / (std + 1e-8)).float()
        prevs.append(snaps[:-1]); nexts.append(snaps[1:])
        print(f"[CondData] traj {j+1}/{n_traj}: {snaps.shape[0]-1} pairs", flush=True)
    data = {"prev": torch.cat(prevs, 0).unsqueeze(1), "next": torch.cat(nexts, 0).unsqueeze(1),
            "raw_mean": mean, "raw_std": std, "nu": nu}
    if not smoke:
        Config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(data, DATA_CACHE)
        print(f"[CondData] saved {data['prev'].shape[0]} pairs -> {DATA_CACHE.name}", flush=True)
    return data


def train_cond(data, device, epochs, seed=TRAIN_SEED):
    """Train the conditional denoiser with the paper's EDM recipe (AdamW, warmup+cosine,
    EMA, AMP, grad clip). Same optimizer/schedule/capacity as the vanilla baseline."""
    torch.manual_seed(seed); np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    prev, nxt = data["prev"], data["next"]
    N = prev.shape[0]
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(seed))
    n_val = max(1, int(0.1 * N))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    tl = DataLoader(TensorDataset(prev[tr_idx], nxt[tr_idx]), batch_size=Config.BATCH_SIZE,
                    shuffle=True, pin_memory=(device.type == "cuda"))
    vl = DataLoader(TensorDataset(prev[val_idx], nxt[val_idx]), batch_size=Config.BATCH_SIZE,
                    shuffle=False)
    model = ConditionalDenoiser(base_ch=Config.BASE_CHANNELS).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[CondTrain] parameters: {n_params/1e6:.3f} M  pairs: {N}", flush=True)
    sched = build_sched(Config, device)
    ncond = nu_cond_value(data["nu"], Config.NU_LIST)
    total_steps = epochs * max(1, len(tl))
    ema = EMA(model, decay=adaptive_ema_decay(total_steps, decay_max=Config.EMA_DECAY_MAX))
    optim = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    warmup = max(1, epochs // 20)

    def lr_lambda(e):
        if e < warmup:
            return (e + 1) / warmup
        p = (e - warmup) / max(1, epochs - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * p))

    lr_sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    use_amp = device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)
    hist = []
    for ep in range(epochs):
        model.train(); ep_loss = 0.0; nb = 0
        for p, x in tl:
            p = p.to(device, non_blocking=True); x = x.to(device, non_blocking=True)
            B = x.shape[0]
            cond = torch.full((B,), ncond, device=device)
            sigma = sched.sample_sigma_train(B)
            x_noisy = x + torch.randn_like(x) * sigma.view(-1, 1, 1, 1)
            optim.zero_grad(set_to_none=True)
            ctx = autocast("cuda") if use_amp else nullcontext()
            with ctx:
                x0 = denoise_cond(model, x_noisy, p, sigma, cond, sched)
                w = sched.loss_weight(sigma).view(-1, 1, 1, 1)
                loss = (w * (x0 - x) ** 2).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim); scaler.update(); ema.update(model)
            ep_loss += loss.item(); nb += 1
        model.eval(); v_mse = 0.0; v_n = 0
        with torch.no_grad():
            for p, x in vl:
                p = p.to(device); x = x.to(device); B = x.shape[0]
                cond = torch.full((B,), ncond, device=device)
                sigma = sched.sample_sigma_train(B)
                x_noisy = x + torch.randn_like(x) * sigma.view(-1, 1, 1, 1)
                x0 = denoise_cond(model, x_noisy, p, sigma, cond, sched)
                v_mse += ((x0 - x) ** 2).mean().item(); v_n += 1
        lr_sched.step()
        hist.append({"epoch": ep + 1, "train_loss": ep_loss / nb, "val_mse": v_mse / max(v_n, 1)})
        if ep == 0 or (ep + 1) % 10 == 0 or ep == epochs - 1:
            print(f"[CondTrain] ep {ep+1:03d}/{epochs}  loss={ep_loss/nb:.4f}  "
                  f"val_mse={v_mse/max(v_n,1):.4f}  lr={optim.param_groups[0]['lr']:.2e}", flush=True)
    ema.copy_to(model)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    cpath = CKPT_DIR / f"cond_nu{REGIME_IDX}_d{DELTA_STEPS}_seed{seed}.pt"
    torch.save({"model": model.state_dict(), "hist": hist, "ncond": ncond, "n_params": n_params}, cpath)
    print(f"[CondTrain] saved -> {cpath.name}", flush=True)
    return model, sched, ncond


@torch.no_grad()
def rollout(model, sched, data, device, ncond, n_steps, burn_in, collect_every, chains, seed=123):
    """Autoregressive rollout: start each chain from a real DNS state, sample the next state
    repeatedly, discard a burn-in so the chain forgets its initial condition, and collect the
    stationary states. Returns a (n_samples, 1, H, W) ensemble."""
    prev = data["prev"]
    idx = torch.randperm(prev.shape[0], generator=torch.Generator().manual_seed(seed))[:chains]
    x = prev[idx].to(device)
    gen = torch.Generator(device=device).manual_seed(seed)
    collected = []
    for t in range(n_steps):
        x = sample_cond_next(model, sched, x, ncond, Config.GRID, device, Config.N_SAMPLE_STEPS, gen)
        if t >= burn_in and (t - burn_in) % collect_every == 0:
            collected.append(x.detach().cpu())
        if (t + 1) % 16 == 0:
            print(f"[Rollout] step {t+1}/{n_steps}", flush=True)
    return torch.cat(collected, 0)


def _metrics(gen, dns):
    return {"lsd": metric_lsd_aggregate(gen, dns),
            "energy_flux_rmse": metric_energy_flux_rmse(gen, dns),
            "enstrophy_flux_rmse": metric_enstrophy_flux_rmse(gen, dns),
            "inv_cascade_pct": metric_inverse_energy_cascade_recovery(gen, dns),
            "fwd_cascade_pct": metric_forward_enstrophy_cascade_recovery(gen, dns)}


def compare_and_plot(cond_samples, device):
    """Score the conditional rollout and the unconditional WRSG samples against the DNS
    reference at the same regime, write the CSV, and draw the spectrum/flux comparison."""
    payload = load_payload(Config)
    fields = payload["fields"]; re = payload["re_labels"]
    dns = fields[re == REGIME_IDX]
    n = min(dns.shape[0], cond_samples.shape[0], 512)
    dns = dns[:n].to(device)
    ncond = nu_cond_value(Config.NU_LIST[REGIME_IDX], Config.NU_LIST)
    wrsg = load_eval_model("wrsg", Config.SEEDS[0], Config, device)
    sched = build_sched(Config, device)
    wrsg_s = sample_dpmpp_2m(wrsg, sched, n, ncond, Config.GRID, device,
                             n_steps=Config.N_SAMPLE_STEPS, noise_seed=4242)
    cond_s = cond_samples[:n].to(device)
    rows = [{"model": "WRSG (unconditional)", **_metrics(wrsg_s, dns)},
            {"model": "Conditional rollout",  **_metrics(cond_s, dns)}]
    Config.CSV_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(CSV_OUT, index=False)
    print(f"[CondCompare] wrote {CSV_OUT.name}", flush=True)
    for r in rows:
        print(f"  {r['model']:22s} LSD={r['lsd']:.3f}  Eflux={r['energy_flux_rmse']:.3f}  "
              f"Zflux={r['enstrophy_flux_rmse']:.3f}  inv={r['inv_cascade_pct']:.1f}%  "
              f"fwd={r['fwd_cascade_pct']:.1f}%", flush=True)

    def spec(f):
        E, k = compute_dealiased_spectrum(f.squeeze(1))
        return k.cpu().numpy(), E.mean(0).clamp(min=1e-12).cpu().numpy()

    def eflux(f):
        P, k = compute_energy_flux(f)
        return k.cpu().numpy(), P.mean(0).cpu().numpy()

    def zflux(f):
        P, k = compute_enstrophy_flux(f)
        return k.cpu().numpy(), P.mean(0).cpu().numpy()

    series = [("DNS", dns, "k", "-"), ("WRSG (uncond.)", wrsg_s, "#1f77b4", "--"),
              ("Conditional rollout", cond_s, "#d62728", "--")]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for name, f, c, ls in series:
        k, E = spec(f); ax[0].loglog(k[k > 0], E[k > 0], c, ls=ls, lw=2, label=name)
        ke, Pe = eflux(f); ax[1].semilogx(ke[ke > 0], Pe[ke > 0], c, ls=ls, lw=2, label=name)
        kz, Pz = zflux(f); ax[2].semilogx(kz[kz > 0], Pz[kz > 0], c, ls=ls, lw=2, label=name)
    ax[0].set_title(r"Dealiased vorticity spectrum $|\hat{\omega}(k)|^2$"); ax[0].set_xlabel("k")
    ax[1].set_title(r"Energy flux $\Pi_E(k)$"); ax[1].set_xlabel("k")
    ax[2].set_title(r"Enstrophy flux $\Pi_Z(k)$"); ax[2].set_xlabel("k")
    for a in ax:
        a.axvline(Config.FORCING_K, color="orange", ls=":", lw=1); a.legend(fontsize=9)
    fig.suptitle(rf"Unconditional WRSG vs conditional autoregressive rollout at $\nu={Config.NU_LIST[REGIME_IDX]}$ "
                 f"(steady state, {n} samples each)", fontsize=12)
    fig.tight_layout()
    Config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[CondCompare] wrote {FIG_OUT.name}", flush=True)
    return rows


def _load_cond(seed, device):
    """Rebuild the conditional denoiser and load a trained checkpoint by seed."""
    model = ConditionalDenoiser(base_ch=Config.BASE_CHANNELS).to(device)
    ck = torch.load(CKPT_DIR / f"cond_nu{REGIME_IDX}_d{DELTA_STEPS}_seed{seed}.pt", map_location="cpu")
    model.load_state_dict(ck["model"]); model.eval()
    return model, ck["ncond"]


@torch.no_grad()
def rollout_windows(model, sched, data, device, ncond, n_steps, windows, chains, seed=123):
    """One long rollout; collect the stationary samples in several disjoint step-windows so the
    drift can be measured as a function of rollout depth (is the drifted state stationary?)."""
    prev = data["prev"]
    idx = torch.randperm(prev.shape[0], generator=torch.Generator().manual_seed(seed))[:chains]
    x = prev[idx].to(device)
    gen = torch.Generator(device=device).manual_seed(seed)
    bucket = {i: [] for i in range(len(windows))}
    for t in range(n_steps):
        x = sample_cond_next(model, sched, x, ncond, Config.GRID, device, Config.N_SAMPLE_STEPS, gen)
        for i, (lo, hi) in enumerate(windows):
            if lo <= t < hi:
                bucket[i].append(x.detach().cpu())
        if (t + 1) % 20 == 0:
            print(f"[Robust] rollout step {t+1}/{n_steps}", flush=True)
    return {i: torch.cat(v, 0) for i, v in bucket.items() if v}


def robustness_main(device):
    """Robustness of the conditional-drift finding: (1) roll the trained seed-29 model out to
    depth 200 and measure the steady-state LSD in successive windows (is the drift a stable
    stationary property or a burn-in transient?); (2) train an independent second seed and
    check the rollout LSD reproduces. Writes conditional_robustness.csv; does not touch the
    seed-29 conditional_comparison.csv the paper cites."""
    data = generate_traj(device, smoke=False)
    sched = build_sched(Config, device)
    payload = load_payload(Config)
    dns = payload["fields"][payload["re_labels"] == REGIME_IDX].to(device)
    rows = []
    m29, nc29 = _load_cond(TRAIN_SEED, device)
    windows = [(32, 64), (80, 128), (152, 200)]
    wins = rollout_windows(m29, sched, data, device, nc29, n_steps=200, windows=windows, chains=ROLL_CHAINS)
    for i, (lo, hi) in enumerate(windows):
        if i in wins:
            n = min(dns.shape[0], wins[i].shape[0])
            lsd = metric_lsd_aggregate(wins[i][:n].to(device), dns[:n])
            rows.append({"check": f"seed{TRAIN_SEED}_window_{lo}_{hi}", "n": wins[i].shape[0], "lsd": lsd})
            print(f"[Robust] seed{TRAIN_SEED} window {lo}-{hi}: LSD={lsd:.3f} (n={wins[i].shape[0]})", flush=True)
    m47, sched47, nc47 = train_cond(data, device, Config.EPOCHS, seed=47)
    s47 = rollout(m47, sched47, data, device, nc47, ROLL_STEPS, ROLL_BURNIN, ROLL_COLLECT_EVERY,
                  ROLL_CHAINS, seed=47)
    n = min(dns.shape[0], s47.shape[0])
    lsd47 = metric_lsd_aggregate(s47[:n].to(device), dns[:n])
    rows.append({"check": "seed47_rollout", "n": s47.shape[0], "lsd": lsd47})
    print(f"[Robust] seed47 rollout: LSD={lsd47:.3f} (n={s47.shape[0]})", flush=True)
    pd.DataFrame(rows).to_csv(Config.CSV_DIR / "conditional_robustness.csv", index=False)
    print("[Robust] wrote conditional_robustness.csv", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end run to check for bugs")
    ap.add_argument("--robust", action="store_true", help="drift-stability + second-seed robustness")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    if args.robust:
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        print(f"[Conditional] robustness on device={device}", flush=True)
        robustness_main(device)
        print("[Conditional] robustness done", flush=True)
        return
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[Conditional] device={device}  smoke={args.smoke}", flush=True)
    epochs = args.epochs or (2 if args.smoke else Config.EPOCHS)
    data = generate_traj(device, smoke=args.smoke)
    model, sched, ncond = train_cond(data, device, epochs)
    if args.smoke:
        samples = rollout(model, sched, data, device, ncond, n_steps=6, burn_in=2,
                          collect_every=2, chains=8)
    else:
        samples = rollout(model, sched, data, device, ncond, n_steps=ROLL_STEPS,
                          burn_in=ROLL_BURNIN, collect_every=ROLL_COLLECT_EVERY, chains=ROLL_CHAINS)
    print(f"[Conditional] collected {samples.shape[0]} steady-state samples", flush=True)
    compare_and_plot(samples, device)
    print("[Conditional] done", flush=True)


if __name__ == "__main__":
    main()
