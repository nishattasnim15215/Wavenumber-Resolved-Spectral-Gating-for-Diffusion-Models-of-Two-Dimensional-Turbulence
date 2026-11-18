# WRSG — TMLR Talk · End-to-End Speaker Notes

**Wavenumber-Resolved Spectral Gating for Diffusion Models of 2D Turbulence**  
Nishat Tasnim · Yong Shi — Kennesaw State University

_18 slides · target ~15–18 min. Times are approximate; the three headline numbers (−59% LSD, 21→89% cascade, +3.5% cost) are the spine — land them at least twice._

---

## Slide 1 — Title

Good [morning/afternoon] everyone. Thank you for coming. My name is Nishat Tasnim, and I'll be presenting our work on Wavenumber-Resolved Spectral Gating for diffusion models of two-dimensional turbulence — joint work with Yong Shi at Kennesaw State University.

Here's the one-sentence version, and I'll spend the next fifteen minutes unpacking it: diffusion models are becoming attractive generative surrogates for turbulence, but they get the physics subtly wrong — they reproduce the point statistics of the field while distorting its energy spectrum and the fluxes that move energy across scales. We introduce a small Fourier-domain component that targets exactly that failure, it costs about three and a half percent extra at inference, and it transfers beyond the flow it was trained on.

All the code and the trained checkpoints are public at the link on the slide. Let me start with why you'd want a generative model of turbulence at all.

---

## Slide 2 — The takeaway

Before any detail, here is what I want you to remember.

First number: our model cuts the aggregate log-spectral distance to the ground-truth simulation by fifty-nine percent relative to a plain diffusion U-Net. Log-spectral distance measures how wrong the energy spectrum is, averaged over scales — so this is a direct measure of spectral fidelity.

Second number: the forward-enstrophy cascade — the flux that carries enstrophy from large to small scales — goes from twenty-one percent recovered by the baseline to eighty-nine percent with our model. That's the transport physics, not just the static spectrum.

Third number: this costs three and a half percent more at inference for eight percent more parameters. It is not a heavy add-on.

And the part I'm most excited about scientifically: the gate transfers. When we move to a different forcing or a different physical field entirely, the architectural piece keeps working. That tells us we've captured something general about spectral bias, not a trick tuned to one dataset. Let me build up to each of these.

---

## Slide 3 — Why generative turbulence

Why generate turbulence at all? Direct numerical simulation is accurate but costly, and many downstream tasks — data assimilation, ensemble priors, uncertainty studies — need many independent samples of the flow. A generative model that produces physically valid fields quickly is a useful prior.

An important framing choice: we solve the unconditional problem. We want to sample the stationary distribution of the turbulence directly. This is different from a conditional emulator that predicts the next state from the current one. Conditional models look great on short horizons — near-zero one-step error — but when you run them free for a long time, their statistics drift and the small scales can collapse. Sampling the distribution directly sidesteps that. I'll show a direct comparison later.

The key requirement for science is physical faithfulness. It is easy to make a field that looks like turbulence to the eye. It is hard to get the energy spectrum and — harder still — the spectral fluxes right, because those encode how energy and enstrophy move across scales. That distinction is the whole story of this talk.

And the practical payoff is real: at this resolution our sampler produces an independent field about twenty times faster than advancing the solver over one decorrelation time.

---

## Slide 4 — Background

A quick orientation for those who don't work in turbulence. Our testbed is forced two-dimensional Kolmogorov flow. It's a standard, well-understood system, which is exactly what you want for a benchmark: we know what the correct statistics look like.

The defining feature of two-dimensional turbulence is a dual cascade. Energy is transported toward large scales — the inverse cascade — while enstrophy, which is vorticity squared, is transported toward small scales — the forward cascade. These transports are physical fingerprints. A model can get the spectrum roughly right and still get the fluxes wrong, and that matters.

So we don't score samples on pixel error. We score them against direct numerical simulation on three families of diagnostics: the energy spectrum, which is power by scale; the spectral fluxes, which measure the cascades; and structure functions, which capture small-scale increment statistics.

Finally, scope. We simulate seven viscosity regimes spanning roughly a factor of nine in Reynolds number. Crucially it is a single model conditioned on a continuous log-viscosity — not seven separate models — and the panel on the right confirms our simulations are well-resolved across all of them.

---

## Slide 5 — The problem

Here is the failure mode we set out to fix. If you train a plain diffusion U-Net on this data, the point-wise statistics of the vorticity come out well — the one-point distribution looks right. So by naive measures the model seems fine.

But look at the spectrum and the fluxes. The model distorts the inertial range of the energy spectrum and misplaces power across wavenumber bands. The top-right panel shows the per-band spectral error — you can see the model systematically over- and under-weighting specific bands. This is spectral bias, and it is well known that convolutional networks have it.

The consequence is most severe in the fluxes: the plain model recovers only about ten to twenty percent of the inverse-energy and forward-enstrophy cascades. The transport physics is largely wrong.

And you can't buy your way out with a generic bigger bottleneck. A squeeze-and-excite block of the same conditioning form gives a log-spectral distance of 0.745 — essentially identical to the plain U-Net's 0.744. The problem is not capacity. It's that nothing in the architecture is set up to correct power on a per-scale basis. That's the gap the gate fills.

---

## Slide 6 — Section: Method

So how do we fix it? The method has two ingredients — an architectural component, the gate, and a set of soft physics losses — and I'll take them in turn, starting with the gate itself.

---

## Slide 7 — The gate

This is the core idea, and it's deliberately simple. Inside the U-Net bottleneck we insert a small Fourier-domain component. We take the bottleneck feature map, transform it to the Fourier domain, multiply each radial-wavenumber band by a learned factor, and transform back. That's the gate.

Three design choices make it work and make it general. First, the correction is multiplicative, not additive, because spectral bias is a multiplicative distortion — the model under- or over-predicts the power in a band by some factor. Second, it is indexed by the radial wavenumber, the magnitude of k, because the statistics we care about are isotropic. A very useful side effect: the number of parameters depends on the number of bands, not on the grid size, so the same gate definition scales across resolutions. Third, and this is what separates it from a fixed spectral filter, the band factors are produced by a small network conditioned on the diffusion noise level and on the viscosity. The correction adapts as denoising proceeds and as the flow regime changes.

We initialize the scale to zero, so the gate starts as the identity and only learns a correction if it helps — this keeps the comparison to the no-gate baselines honest. We use sixteen bands on the thirty-two-by-thirty-two bottleneck. The panel on the right shows what the gate actually learns: which bands it chooses to amplify and which to attenuate.

---

## Slide 8 — Two ingredients + hybrid

The full model has two parts, and then one optional extension.

Part one is the gate I just described — the architectural component. Its job, as the ablation will show, is to route spectral power: it fixes the large scales and the cross-scale transport.

Part two is a set of soft physics losses. Alongside the standard denoising objective we add penalties that push the samples' enstrophy, spectral amplitude, structure functions, integral length, and spectral flux toward the DNS statistics. These mostly fix the small-scale point statistics. The two parts are complementary — that complementarity is the key finding of the ablation.

The optional extension: the gate controls per-band amplitude, but it does not mix Fourier modes, and cross-mode mixing is what the fluxes really need. So we add a Fourier-neural-operator branch beside the gate — the model computes gate of h plus a learned FNO of h. The FNO supplies the cross-mode mixing the gate lacks, and this hybrid gives the best cascade transport and the lowest flux error, at the cost of the FNO block's parameters. Throughout, I'll call the gate-plus-losses model WRSG, and the hybrid WRSG-plus-FNO.

---

## Slide 9 — Experimental setup

Let me set up the evaluation so the numbers land. The data is forced two-dimensional Kolmogorov flow at 128-squared resolution, seven viscosity regimes, a thousand decorrelated snapshots per regime.

We compare eight model variants, each over five random seeds, so every number I show is a mean over seeds with a real spread. The baselines are chosen to be fair and hard: a plain U-Net, a squeeze-and-excite bottleneck, an FNO-block bottleneck with 2.8 times our parameters, and a standalone Fourier neural operator. The two strongest baselines carry more parameters than we do.

For diffusion we use EDM training and the DPM-Solver-plus-plus deterministic sampler at fifty steps, and again, a single model conditioned on log-viscosity covers all seven regimes.

We report eight physics metrics against DNS — spectral distance, structure functions, integral length, the two flux errors, and the two cascade-recovery percentages. Because we have five seeds we use paired Wilcoxon tests — the smallest possible p-value there is 0.0625, and we hit it, meaning every seed agrees on the direction — with bootstrap confidence intervals and effect sizes. The whole sweep is about a hundred GPU-hours, and everything is released for reproduction.

---

## Slide 10 — Headline results

Here are the headline numbers. Start with spectral fidelity: the aggregate log-spectral distance to DNS drops from 0.744 for the plain U-Net to 0.304 for WRSG — the fifty-nine percent I promised — and the hybrid nudges it a little further to 0.293.

Now the transport. Inverse-energy cascade recovery goes from ten percent to forty-nine percent for WRSG; forward-enstrophy from twenty-one to eighty-nine. The hybrid pushes the inverse cascade to eighty-four percent and the forward cascade to a hundred and ten — and I'll be honest about that overshoot past a hundred later — while also giving the lowest flux error.

The bar chart shows all eight metrics across all eight variants. The two rightmost groups are ours. WRSG-plus-FNO is best on seven of the eight metrics and WRSG is second on those same seven. Both of them beat the FNO block, which has almost three times our parameters, on spectral distance.

And these are not noise. The effect sizes are large — Cohen's d between about eight and forty — and every seed agrees on the direction of the effect, which is why the paired tests sit at their significance floor. One honest nuance: WRSG-plus-FNO is a statistical tie with WRSG on aggregate spectral distance. The FNO branch buys you cascade transport and flux accuracy, not aggregate spectral fidelity. There is one metric where we lose — I'll come back to it in the negative results.

---

## Slide 11 — The cascades

I want to dwell on the fluxes for a moment, because this is the diagnostic that's hardest to fake and where the method really earns its keep.

A spectral flux measures how much energy or enstrophy is being transported through a given scale. In a correct simulation you see characteristic plateaus — a constant flux across the inertial range — with a definite sign that encodes the direction of the cascade. This is transport, a two-point cross-scale quantity, so a model can match the one-point spectrum reasonably and still get the flux completely wrong.

And that's exactly what the plain U-Net does — its flux curves are flat or wrong-signed where they should show a clean plateau. The colored curves are the generated fluxes, the black curves are DNS. As you move to the gate and then the hybrid, the plateaus reappear and line up with DNS: the gate restores the large-scale plateau by routing per-band power, and the FNO branch restores the cross-mode transport that a purely per-band correction can't produce. This panel is the clearest picture of why both components are needed, which is the ablation I'll show next.

---

## Slide 12 — Factorial ablation

Why do we claim you need both the gate and the losses? Because of a clean factorial ablation, and this is my favorite result in the paper.

Look at the four cases. The plain model sits at spectral distance 0.744. Now add only the gate: you go to 0.688 — barely a dent. Add only the physics losses, with no gate: it actually gets worse, 0.925, the worst of any variant, because you're pushing hard on the statistics with no mechanism to route the spectral correction, and that distorts the aggregate spectrum. So on their own, one component does almost nothing and the other backfires.

But put them together and you land at 0.304 — cut by more than half. That's a strong positive interaction: the whole is far more than the sum of the parts. The reason is a division of labor. The gate routes spectral power — it fixes the large scales and the cascade transport. The losses correct the small-scale point statistics — notice losses-only already gets the structure function down to 0.054 and forward-enstrophy up to seventy-one percent. Each handles a different part of the spectrum, and only together do they cover it. This is also why a single mechanism, however tuned, wouldn't be enough.

---

## Slide 13 — Pareto / cost

A natural worry with a physics-motivated add-on is that you're just spending parameters. You're not. This is the quality-versus-cost trade-off.

WRSG reaches spectral distance 0.304 at two and a half million parameters. The FNO-block baseline, sitting at 0.553, carries two point eight times as many parameters and is still well behind. So on the Pareto frontier we are both cheaper and better than the strongest architectural baseline.

Relative to the plain U-Net, the gate itself adds only about eight percent more parameters and about three and a half percent more inference time — from roughly sixty-six to sixty-eight milliseconds per field. And in absolute terms the sampler produces a statistically independent field about twenty times faster than advancing the DNS solver over a decorrelation time. So this is a cheap, fast component. The takeaway is that what helps is the specific frequency-domain correction, not raw capacity — the squeeze-and-excite and FNO baselines had capacity and didn't get here.

---

## Slide 14 — Calibrated uncertainty

A generative surrogate is only useful if its uncertainty is trustworthy, so we calibrate it. We draw a thirty-two-member ensemble per condition and form per-pixel percentile bands.

Out of the box those bands are under-dispersed — too narrow. Raw empirical coverage is between 0.72 and 0.85 against a ninety-percent target. The fix is deliberately minimal: a single scalar that rescales the band width, fit on a held-out calibration split that is stratified across the seven regimes. Stratifying matters because the under-dispersion varies with viscosity. With that one scalar, every variant lands on target — 0.899 to 0.902. I want to be careful with language: this is a variance recalibration, not a conformal method with a coverage guarantee.

Now, a single global scalar can't fix everything, and we say where it fails. When you resolve coverage by scale, the band is well calibrated at low wavenumber but badly under-dispersed at the small scales — only about twenty-three percent coverage for k above twenty. If you let the calibration vary with scale — a per-shell rescaling, or better a split-conformal band that adds an absolute width — the small-scale coverage climbs to the mid-eighties and up to 0.92. So the residual gap is a matter of calibration resolution and calibration-sample size, not spread that is fundamentally missing. We think that honesty about where calibration is easy and where it's hard is part of the contribution.

---

## Slide 15 — Generalization and transfer

Now the generalization story, which is what elevates this from an engineering fix to a claim about mechanism. Three tests.

First, out-of-distribution viscosity. Because we condition on continuous log-viscosity, we can query the model at viscosities it never trained on. It beats the plain U-Net at every unseen viscosity. At interpolation points its error tracks the in-distribution trend, so the conditioning genuinely interpolates rather than memorizing seven regimes. Extrapolation is asymmetric — below the training range it's as good as its best in-distribution regime; above the range everything degrades, but we still reach 1.975 where the U-Net blows up to 3.55.

Second, a different flow. We swap the shear forcing for a two-dimensional cellular forcing at a scale the gate was never tuned for, and retrain the factorial. The pattern reproduces: the gate lowers spectral distance, and gate-plus-losses recovers ninety-eight percent of the inverse cascade — with no retuning.

Third, and most telling, a different field entirely: a passive scalar advected by the flow, obeying a different PDE. Here the two components split apart cleanly. The gate alone transfers and lowers spectral distance. But the physics-loss weights, which were calibrated on vorticity, do not transfer — they help the scalar's structure functions while raising its aggregate distance. And that split is the punchline: the architectural component is general; the loss weights are field-specific. That's the strongest evidence that the gate is capturing spectral bias itself, not a property of one dataset.

---

## Slide 16 — Negative results

I want to spend real time on what doesn't work, because a benchmark paper is only as trustworthy as its negative results, and we report all of ours.

First, the metric we lose on. The physics losses worsen the large-scale integral length compared to the gate alone — 0.112 versus 0.013. This is the one metric in the headline table where our full model falls behind the plain baselines, and it's a genuine trade-off: the same losses that sharpen the small scales pull the large-scale length off. If your application cares most about the integral length, the gate-only model is actually your best choice, and we say so.

Second, the cascade peaks. Even our best model does not fully recover the sharp DNS cascade peaks, and the hybrid overshoots the forward cascade to a hundred and ten percent. The transport is faithful in shape and far better than any baseline, but it is not exact.

Third, we ran the conditional-emulator comparison I promised earlier. A standard next-step emulator gets near-zero one-step error, but over a long free-running rollout its statistics drift to a spectral distance of 1.39, versus 0.49 for our unconditional samples in the same regime. That drift is precisely why we chose to sample the stationary distribution directly.

So the honest summary: the gate is a general fix for spectral bias. It is not a universal turbulence solver, and we're precise about the boundary.

---

## Slide 17 — Conclusion

Let me pull it together. The message is a single idea with strong evidence behind it.

The core failure of diffusion models on turbulence is spectral bias: the samples look right point-wise but the spectrum and the cross-scale fluxes are wrong. A small, conditioned, per-band Fourier gate targets that directly and fixes it — fifty-nine percent lower spectral distance, cascades from ten to forty-nine and twenty-one to eighty-nine percent, for about three and a half percent more compute.

The gate and the physics losses are complementary: neither works well alone, together they cut the error by more than half, because each covers a different band of the spectrum.

And the result generalizes in a specific, interpretable way — the gate is general across forcing and even across a different physical field, while the loss weights are field-specific. On top of that the uncertainty is calibrated, sampling is about twenty times faster than the solver, and we're explicit about the two places it falls short.

Everything — code, configs, checkpoints — is public, and this work was supported by NSF. Thank you. I'd be glad to take your questions.

---

## Slide 18 — Thank you / Q&A

Thank you — happy to take questions. Here are a few I anticipate, with short answers:

Q: Why a hand-designed gate instead of letting an FNO learn everything? A: We tried — the FNO block has 2.8x our parameters and is well behind on spectral distance. The gate's multiplicative, per-band, conditioned form is the right inductive bias for spectral bias, and it's grid-size independent. The FNO branch is complementary and we include it as the hybrid.

Q: Is the recalibration a conformal guarantee? A: No, and we're careful about that. The single-scalar rescaling has no finite-sample guarantee; we call it variance recalibration. The split-conformal variant that reaches 0.92 at small scales does add an absolute width, but our headline coverage claim is marginal calibration, not a guarantee.

Q: Does 110% forward-cascade recovery mean it's better than DNS? A: No — it's an overshoot past the DNS reference, which we count as a defect, not a win. The hybrid trades a little of that for the lowest flux error overall.

Q: Two dimensions — what about 3D? A: Untested, and we say so. The gate is defined on radial |k| with a grid-independent parameter count, so nothing in its construction is 2D-specific, but 3D flows, anisotropic forcing, and experimental data are the honest next targets.

Q: Why unconditional rather than a solver-in-the-loop? A: We see the samples as a fast statistical prior to be corrected by data or a solver, not a replacement for one — the residual flux error is why. Thanks again.

---

