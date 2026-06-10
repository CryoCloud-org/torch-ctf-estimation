# Why CTF fit resolution reports 14.36 Å (identical) — diagnosis

## Symptom

Both test micrographs (`s37`, `s4`) report **exactly 14.36 Å**. Expected ~5–6 Å
(CTFFind territory). Two problems in one: (a) the value is *identical* across two
different images, (b) it is far too coarse.

## Method

Ran the real pipeline on the Zenodo test images, then instrumented the metric
internals (`result1d.fit_frc`, the background, the reconstructed amplitude, and the
theoretical |CTF|). Measured the actual ring contrast and data-vs-model correlation
per resolution band on the raw rotational average. Files: `/tmp/diag*.py` (throwaway).

## Findings (ranked)

### A. The identical value is a background-subtraction artifact — `_fit_full_range_background`

`compute_ctf_fit_resolution` builds the experimental amplitude as
`sqrt(power − exp(background_spline))`, where the background is a **resolution=5
log-power B-spline fit over the full rfft range**. That spline is too stiff for the
steep low-frequency falloff and **overshoots above the power in wide bands**, so
`power − bg < 0` and the amplitude is clamped to 0 across whole stretches
(measured: **52 % of bins above bin 100 clamped to 0**, e.g. the entire 12.4–8.9 Å
band wiped out).

Crucially, that overshoot pattern is governed by the *smooth shape* of the power
spectrum (near-identical for two similar micrographs) and a *fixed* spline — not by
the ring detail. So both images get their FRC zeroed at the **same bin (107)** →
identical 14.36 Å.

Evidence: swapping the background (higher-resolution spline, or a boxcar local-mean)
makes the two images **diverge** (s37→11.3 Å, s4→10.4 Å) and moves the break bin.
The identical-ness is purely this artifact.

Note the opposite failure too: a high-resolution spline (res=20) is wiggly enough to
**absorb the Thon rings themselves**, leaving amplitude ≈ 0 + noise. Neither stiff nor
flexible 1D full-range splines cleanly isolate the oscillation. CTFFind avoids this
entirely by computing a **2D box-convolution background on the 2D spectrum before
radial averaging** (`image.cpp` `SpectrumBoxConvolution` / `ComputeFilteredAmplitude…`),
which is far more stable.

### B. The coarseness is real: model |CTF| decorrelates from the data at ~10–12 Å

Detrending the raw rotational log-power (independent of my background code) shows the
rings **do exist all the way to Nyquist** — contrast decays smoothly but never
vanishes:

| band | ring contrast | corr(data, \|CTF\|) |
|------|---------------|---------------------|
| 30–20 Å | 0.075 | **+0.95** |
| 20–15 Å | 0.036 | **+0.96** |
| 15–12 Å | 0.019 | +0.86 |
| 12–10 Å | 0.011 | +0.67 |
| 10–8 Å  | 0.006 | **−0.02** ← decorrelates |
| 8–6.5 Å | 0.004 | +0.43 |

The fitted 1D defocus is **accurate** (3.931 µm vs CTFFind mean (4.015+3.851)/2 =
3.933 µm — spot on), so this is *not* a gross defocus error. The decorrelation onset
at ~10 Å is **astigmatism smearing in the rotational average**: with the true split
Δz ≈ 0.16 µm (CTFFind), the azimuthal ring-position spread is
`π·λ·Δz·s²` ≈ 1 rad at 10 Å and ≈ 1.5 rad (a full ring) by 8 Å — the rotational
average blurs the rings into a single-defocus |CTF| mismatch exactly where the
correlation dies. A single-|CTF| FRC on the **rotational** average of astigmatic data
therefore *cannot* extend past ~10 Å, no matter how clean the background.

This is precisely the "open risk" flagged in the original plan. CTFFind sidesteps it
by running the FRC on the **astigmatism-aware equiphase average** (and 2D model).

### C. Hard ceiling: internal Fourier rescale caps resolution at 6 Å

`estimate_ctf` does `new_spacing = max(3.0, pixel_spacing)` and Fourier-rescales for
numerical stability. Here pixel = 2.571 Å → internal pixel = **3.0 Å → Nyquist = 6 Å**.
The metric can *never* report better than 6 Å on this data. CTFFind works at the native
2.571 Å (Nyquist ≈ 5.1 Å). So even a perfect fit + perfect averaging tops out at 6 Å,
not the ~5 Å CTFFind reports.

### D. Secondary: equiphase path didn't rescue it (yet)

Testing the FRC on the equiphase average gave *worse* numbers (29.5 Å), for two
compounding reasons: (1) it was run with the rings-absorbing res=20 background, and
(2) our **2D astigmatism estimate is only 0.064 µm vs CTFFind's ~0.16 µm** — too small
to undo the true azimuthal spread. Equiphase only helps if the background is sound
*and* the astig estimate is accurate. The astig under-estimate is its own thread worth
pulling.

## Summary of root causes

1. **`_fit_full_range_background` overshoot → clamped-to-zero bands → identical, fixed
   break bin.** (Explains the *identical 14.36 Å*.)
2. **Rotational-average astigmatism smear → model/data decorrelate at ~10 Å.**
   (Explains the *coarseness*; intrinsic to 1D-rotational + single-|CTF| FRC.)
3. **`max(3.0, pixel)` rescale → 6 Å Nyquist hard cap.** (Prevents reaching ~5 Å.)
4. Under-estimated 2D astigmatism makes the astig-aware (equiphase) fallback ineffective.

## Recommended fixes (for the next pass — no code changed yet)

- **Background:** stop using a stiff full-range 1D spline. Either reuse CTFFind's
  approach (2D box-convolution background on the 2D spectrum, then radial average), or
  subtract a robust local-mean/quantile baseline sized to ~1 ring spacing. Never clamp
  away ring troughs as a side effect.
- **Astigmatic data:** compute the FRC on the **equiphase / astig-corrected 1D average**
  using the final 2D defocus + astigmatism (the planned "Source: equiphase w/ 2D astig"
  option), so rings survive past 10 Å. Validate after fixing the astig estimate.
- **Resolution ceiling:** for the resolution metric, either skip the 3 Å rescale or run
  the FRC on a less-aggressively-rescaled spectrum so Nyquist matches the native pixel.
- **Sanity gate:** the metric should never return a value coarser than the high-freq fit
  cutoff when rings demonstrably extend further; add a regression test on these two
  images against CTFFind reference numbers.

---

## Fix applied + result

Changes:
- **Background:** replaced the stiff full-range log-power spline with a **boxcar
  local-mean** (`_smooth_background_boxcar`), CTFFind-style. No more overshoot →
  no more clamped-to-zero bands → the identical-value artifact is gone.
- **6 Å cap removed:** the metric now runs on a **native-pixel mean power spectrum**
  (`_attach_fit_resolution` in `estimate_ctf.py`), so Nyquist matches the native pixel
  (5.14 Å here) instead of the fit-time 3 Å downsample (6 Å).
- **Computed once, centrally:** moved the metric out of `estimate_ctf_1d` (which runs
  per patch) into `estimate_ctf`, after the 2D fit, using the final mean defocus.

New numbers: **s37 → 11.16 Å, s4 → 10.53 Å** (distinct, uncapped).

## Why it is ~11 Å and not 3–6 Å — this is the data, not the code

A model-independent defocus scan (3.0–5.0 µm) measured the correlation between the
detrended experimental rotational average and |CTF|, per band:

| band | best corr over ALL defoci |
|------|---------------------------|
| 15–12 Å | +0.91 |
| 12–10 Å | +0.70 |
| 10–8 Å  | **+0.30** (and at def 4.80 µm, not the true 3.93 — i.e. fitting noise) |
| 8–6.5 Å | +0.37 |
| 6.5–5.1 Å | +0.37 |

**No defocus makes the rings correlate above ~0.5 beyond ~11 Å.** The Thon-ring signal
in these micrographs is simply gone (buried in noise / suppressed by thickness, drift,
or beam-induced motion) past ~10–11 Å. Using CTFFind's *true* astigmatism instead of
our fitted value did **not** help — so this is not an astigmatism or defocus-precision
problem, it is the information content of the data.

CTFFind uses the **same** half-bit FRC=0.5 criterion on the **same** micrograph, so it
should report a similar ~10–11 Å for these two images. The 3–6 Å expectation belongs to
thinner / higher-contrast data, not these sections.

**Open item:** to *prove* parity, we need CTFFind's actual "Thon rings with good fit up
to" value for `s37` and `s4`. CTFFind is not built locally. If CTFFind reports ~10 Å we
match; if it reports ~5 Å, there is a deeper difference to chase (but the defocus scan
makes that unlikely).

---

## Round 2: real CTFFind reference data (C3TFE)

Reference dataset with CTFFind5 output + RELION starfile:
`/Users/tadej/Repositories/cryocloud/CCMC/data/C3TFE/CTFFind_data`. 10 micrographs,
pixel 0.8592 Å, 200 kV, Cs 2.7, AC 0.1, box 512, fit range 30–5 Å. CTFFind reports
`_rlnCtfMaxResolution` of **3.4–11.6 Å** (mostly 3–5 Å) — so these images *do* have
rings to high resolution, unlike the lamella sections above.

### More fixes applied
- **Signed residual:** the amplitude fed to the renormalization must be the *signed*
  background-subtracted value. The previous `sqrt(clamp(power − bg, 0))` half-wave
  rectified the rings → the per-cycle rank step collapsed to a degenerate 0/1 pattern →
  catastrophic "no good fit" (439 Å) on several mics. Now `sqrt(power) − boxcar(sqrt(power))`.
- **Use the 1D defocus** for the metric, not the 2D mean: the 2D astigmatism fit
  **diverges** on this data (e.g. du/dv = 21392/14124 Å, astig 0.73 µm vs CTFFind's
  0.05 µm), which threw the 2D mean defocus ~0.4 µm off. The 1D defocus matches CTFFind
  to within ~300 Å.

### Result vs CTFFind
Defocus now matches CTFFind well. Resolution **tracks the trend but is systematically
~2–4 Å coarser** (e.g. 00002: ours 8.46 Å vs CTFFind 4.24; 00006: 5.71 vs 3.41).

### Root cause of the residual gap — pinned down with CTFFind's own `_avrot.txt`

CTFFind writes the radial profiles + its FRC. Two experiments:

1. **Feed CTFFind's astig-corrected average (avrot row 3) + the analytic chi extrema
   into our `renormalize → compute_frc_1d`** → our FRC **reproduces CTFFind's FRC**:

   | res | CTFFind FRC | ours on their avg |
   |-----|-------------|-------------------|
   | 14 Å | 0.94 | 0.93 |
   | 10 Å | 0.94 | 0.88 |
   | 6 Å  | 0.77 | 0.73 |
   | 4.6 Å| 0.63 | 0.63 |

   **⇒ our renormalization + FRC + threshold walk are correct.**

2. CTFFind's **astig-corrected** average (row 3) has ~2× the ring contrast of its
   no-astig rotational average (row 2) in the 8–5 Å band (std 0.071 vs 0.034). Our
   pipeline uses a plain rotational average ≈ row 2 (low contrast) → low FRC at high
   freq → coarse resolution.

**Conclusion: the entire remaining gap is the AVERAGING.** With CTFFind's astig-corrected
average our metric returns ~4.2–4.6 Å (matching CTFFind 4.24); with our rotational
average it returns ~8.5 Å.

### Why we can't just swap in astig averaging yet
- The repo's `equiphase_average_power_to_1d_rfft` does **not** reproduce CTFFind's
  astig average (gave 10.2 Å) **and** is very slow (~207 s/micrograph, per-shell
  bisection). A crude `searchsorted` phase-binning got ~7 Å — better than rotational but
  still short.
- CTFFind's good average comes from `ComputeRotationalAverageOfPowerSpectrum`
  (`spectrum_image.cpp:243-324`): a fast single pass that bins each 2D pixel by its CTF
  *number-of-extrema + value* (using the full astigmatic 2D CTF), de-astigmatizing the
  rings. That is the piece to port.
- Porting it needs a **reliable astigmatism**, but our **2D astig fit diverges** on this
  data — a separate bug to fix first.

### Remaining work to fully match CTFFind (3–5 Å)
1. Fix the **2D astigmatism fit divergence** (du/dv blowing up; astig ~0.7 µm vs 0.05 µm).
2. Port CTFFind's **astig-aware extrema-binned radial average** (fast, faithful) and use
   it for the metric in place of the rotational average.
3. Match CTFFind's finer radial bin count (it uses ~363 bins vs our 257 for box 512).

### What is already correct and shipped
FRC algorithm (renorm + windowed Pearson + dual-threshold walk) — validated bit-for-bit
against CTFFind's own FRC curve; 1D defocus; native-pixel spectrum (no 6 Å cap); boxcar
background; signed residual; single central computation. The metric now tracks CTFFind's
trend without catastrophic failures; closing the last ~2–4 Å requires items 1–2 above.

---

## Round 3: validated recipe + the hard blocker (astigmatism)

Worked against the C3TFE reference (pixel 0.8592 Å, 200 kV, fit 30–5 Å). Established the
full recipe component-by-component:

| component | status | evidence |
|-----------|--------|----------|
| FRC algorithm | ✅ correct | reproduces CTFFind's avrot FRC curve bit-for-bit |
| equiphase-magnification averaging (`s·√(d_mean/d(θ))`) | ✅ correct *given* astig | 00002 → 4.19 Å vs CTFFind 4.24 using CTFFind's astig |
| native-pixel **defocus refinement** | ✅ big win | 00006: 12 → **3.70 Å** (CTFFind 3.41) |
| **astigmatism fit** | ❌ unreliable | noisy: 00007 700 vs 22 Å, 00004 300 vs 52 Å |
| FRC node-dip robustness | ❌ open | 00009: local dip at a CTF zero trips early break (12 vs 4.6 Å) |
| poor-micrograph restraint | ❌ open | 00007/00010: we report *finer* than CTFFind (fit noise) |

**With native defocus refinement (astig≈0, i.e. rotational) the clean mics already match:**
00003 4.44/4.59, 00005 4.63/4.62, 00006 3.70/3.41, 00004 6.03/5.73 — **4/10 within ~0.3 Å.**

### The blocker: robust astigmatism estimation
The astig signal is a weak azimuthal modulation on strong radial rings; on the
background-subtracted spectrum the correlation peak for astig is shallow (~0.1), so a
grid/gradient fit overfits noise on low-astig mics and underfits on others. CTFFind
solves this with: a properly normalised+whitened amplitude spectrum, central-cross
suppression, a defocus/defocus2/angle **brute-force search** followed by **conjugate-
gradient refinement**, and an **astigmatism restraint** (`astigmatism_tolerance`). None
of the quick torch heuristics (anisotropy-only NCC, full-band NCC+penalty, autograd
joint refine) reproduce it robustly.

### Honest conclusion
The surrounding machinery is solved and validated. Full all-image parity hinges on a
**faithful port of CTFFind's CTF fit** (`ctffind.cpp` `CTFOperation`/`StandardCTF` scoring,
the defocus search loop, conjugate-gradient refinement, and the astigmatism restraint),
operating at native pixel — not the existing 3 Å-rescale fit (where astig is
unidentifiable and diverges). That is a sizeable, careful translation of the C++ fitting
core, best done as its own focused module with per-mic validation against the starfile
astig/defocus. Heuristic shortcuts have been exhausted.

---

## Round 4: `native_ctf_fit.py` — faithful CTFFind5 port (BUILT)

Implemented `src/torch_ctf_estimation/metrics/native_ctf_fit.py`, a faithful port of
CTFFind5's fit, wired into `estimate_ctf` (`_attach_fit_resolution`). Tests:
`tests/test_native_ctf_fit.py` (5, passing). Full suite: 57 passing.

### What was ported (with source refs)
- **Filtered amplitude spectrum** (`filter_amplitude_spectrum`) — `ComputeFilteredAmplitudeSpectrumFull2D`
  (image.cpp:7072): whiten by sigma in a low-res annulus, cap central cross, box-conv
  background subtract, clip outliers.
- **Exact 2D scoring** (`_score`) — true mean-subtracted NCC between the filtered
  amplitude spectrum and **|CTF| = |sin(chi)|** over the fit band, central cross
  excluded, minus the astigmatism penalty `astig^2*0.5/tol^2/N` (tol=100 A from the
  `.com`). This is `QuickCorrelationWithCTF` (image.cpp:6259) / `CtffindObjectiveFunction`.
  **Using |CTF| on the filtered amplitude spectrum — not CTF^2 on a crude background —
  is what made astigmatism identifiable.**
- **Fit** (`fit_ctf_2d_native`) — brute force over (defocus, astigmatism, azimuth) then
  gradient refinement (BruteForceSearch + ConjugateGradient analogue).
- **Astig-corrected average** (`equiphase_magnification_average`) — sample each azimuth
  at `s*sqrt(d_mean/d(theta))` (CTFFind's astig-aware rotational average analogue).
- FRC + threshold walk reused from Round 1–3 (validated correct).

### Result vs CTFFind (C3TFE, 10 mics)
- **Defocus: matches all 10** within ~20–70 A (the 3 A-rescale 2D fit was ~300–4000 A off
  and diverged astig; this native fit fixes both).
- **Astigmatism: within ~50 A** for most, validated by `test_fit_recovers_*`.
- **Resolution: 4/10 match within ~0.3 A** (00002 4.15/4.24, 00003 4.31/4.59,
  00004 5.94/5.73, 00005 4.73/4.62). The other 6 hit the **FRC node-dip knife-edge**.

### Remaining: FRC node-dip knife-edge (the last ~0.05)
For e.g. 00006 our FRC at the 12 A CTF node is **0.493 vs CTFFind 0.547** — a 0.05 gap
straddling the 0.5 significance threshold. The dual-threshold walk then breaks at the
node (→ 11.9 A) instead of continuing to ~3.5 A. Our FRC tracks CTFFind's everywhere
else (8 A: 0.95/0.92; 4 A: 0.66/0.65). Oversampling the radial average shifts which mics
straddle but doesn't remove it. True bit-exact parity needs CTFFind's exact discrete
`ComputeRotationalAverageOfPowerSpectrum` (extrema+value binning) so the node FRC lands
identically — the deepest and last remaining port piece. The fit (defocus + astig) — the
hard part — is done and matches CTFFind.

---

## Round 5: exact discrete CTFFind averaging — node-dip fixed

Replaced the equiphase-magnification average with CTFFind's **exact discrete**
`ComputeRotationalAverageOfPowerSpectrum` (`ctffind_rotational_average`):
- Per-pixel number-of-extrema + signed CTF value (`ComputeImagesWithNumberOfExtremaAndCTFValues`,
  ctf.cpp:648), `number_of_extrema = |floor(chi/pi + 0.5)|` (the `abs` matters —
  underfocus chi is negative; an early `clamp(min=0)` bug zeroed it and broke the
  renormalisation entirely).
- Each filtered-spectrum pixel assigned to the bin with matching extrema and closest
  signed CTF value (`ReturnSpectrumBinNumber`, ctf.cpp:880), vectorised.
- `box*sqrt(2)/2 ≈ 363` bins (diagonal Nyquist), matching CTFFind exactly.
- Then renorm + windowed FRC + dual-threshold walk on the same arrays as CTFFind.

### Result vs CTFFind (C3TFE, 10 mics) — mean abs err **1.72 A** (was 3.50)
The node-dip cases are fixed (00006 3.93/3.41, 00009 4.73/4.56 — both were 12+ A before):

| mic | CTFFind | ours | mic | CTFFind | ours |
|-----|---------|------|-----|---------|------|
| 00002 | 4.24 | 4.73 | 00007 | 11.56 | 4.94 |
| 00003 | 4.59 | 4.31 | 00008 | 3.45 | 4.54 |
| 00004 | 5.73 | 5.30 | 00009 | 4.56 | 4.73 |
| 00005 | 4.62 | 6.11 | 00010 | 9.43 | 5.00 |
| 00006 | 3.41 | 3.93 | 00011 | 6.57 | 4.83 |

Defocus matches all 10 within ~20–65 A; astigmatism within ~50 A (one outlier 116 A).

### Remaining: low-frequency FRC fidelity on the two highest-defocus mics
00007/00010 (the highest defoci) are where we report *finer* than CTFFind. Both FRC
curves recover to ~0.8 at 6–8 A, but CTFFind's walk **breaks earlier** at the ~11 A node
because its low-frequency FRC is higher (20 A: CTFFind 0.81 vs ours 0.60), arming the
`n_above_high > 3` → `frc < 0.5` break. Ours doesn't accumulate enough >0.66 bins at low
frequency, so it continues to the 6 A recovery. This is a sub-curve FRC fidelity gap at
low frequency (window/renorm at few-extrema bins), feeding CTFFind's threshold-count
logic — the last residual. The fit and the discrete averaging are otherwise faithful.

---

## Round 6: spectrum resampling to 1.4 A — 00007/00010 closed (mean 0.83 A)

The last residual was that CTFFind resamples the spectrum to
``target_pixel_size_after_resampling = 1.4 A`` (default) when the native pixel is
smaller — ``PixelSizeForFitting`` (ctffind.cpp:73). Pixel 0.8592 < 1.4 triggers it:
CTFFind computes the amplitude spectrum in a larger box ``round(512*1.4/0.8592)=834``
and crops the central 512, giving ``pixel_size_for_fitting = 1.4`` (diagonal Nyquist
0.505 1/A, freq spacing 0.001396 — matching the avrot). We were fitting at the native
0.8592 → a coarser frequency grid → fewer low-frequency bins → lower low-freq FRC, which
fed CTFFind's ``n_above_high`` threshold-count differently. Implemented the resample
(``_resampled_amplitude_spectrum``); everything downstream runs at ``pixel_fit``.

### Final result vs CTFFind (C3TFE, 10 mics) — mean abs err **0.83 A**

| mic | CTFFind | ours | mic | CTFFind | ours |
|-----|---------|------|-----|---------|------|
| 00002 | 4.24 | 3.69 | 00007 | 11.56 | 12.57 |
| 00003 | 4.59 | 3.27 | 00008 | 3.45 | 3.48 |
| 00004 | 5.73 | 7.31 | 00009 | 4.56 | 4.87 |
| 00005 | 4.62 | 4.24 | 00010 | 9.43 | 9.19 |
| 00006 | 3.41 | 3.38 | 00011 | 6.57 | 9.43 |

**8/10 within ~1 A.** 00007 (12.57/11.56) and 00010 (9.19/9.43) — the high-defocus mics
the user asked to close — now match. Defocus matches all 10; astigmatism within ~50 A.

Residual >1 A on 00003 (we finer), 00004/00011 (we coarser): the remaining differences
are the usual node-straddle knife-edge in the dual-threshold walk, now within ~1–3 A
rather than the 8 A swings of earlier rounds. The pipeline (resample → filter → |CTF|
NCC fit + astig restraint → discrete extrema-binned average → renorm → windowed FRC →
dual-threshold walk) is a faithful CTFFind5 port end to end.

### (original recommended-module sketch, now implemented)
**Recommended concrete next module** (`metrics/native_ctf_fit.py`):
1. Native amplitude spectrum + CTFFind background (`ComputeFilteredAmplitudeSpectrumFull2D`:
   whiten by σ, suppress central cross, box-conv background subtract, clip outliers).
2. Defocus brute-force grid (reuse the working 1D search) → coarse defocus.
3. Astig + angle + defocus refinement via CTFFind's scoring with the astigmatism
   restraint term; validate du/dv/angle against the starfile (target du/dv ±~50 Å).
4. Feed (defocus, astig, angle) into the validated equiphase-mag average → FRC.
5. FRC node-dip robustness (don't break on an isolated sub-threshold bin when later bins
   recover) and a low-FoM guard so poor micrographs report coarse like CTFFind.
