"""CTF fit max-resolution metric, ported from CTFFind.

Reports the highest spatial frequency (best resolution in Angstroms) to which the
fitted CTF model still agrees with the experimental Thon rings -- CTFFind's
"Thon rings with good fit up to: X Angstroms".

This is a faithful port of three CTFFind routines plus a threshold walk:
  - ``Renormalize1DSpectrumForFRC``        (ctf.cpp:697-783)
  - ``ComputeFRCBetween1DSpectrumAndFit``  (ctf.cpp:786-878)
  - dual-threshold resolution walk         (ctffind.cpp:1756-1799)

Domain bridge: CTFFind works on the amplitude spectrum (|F|) and correlates against
|CTF|. torch-ctf-estimation stores power (|F|^2). We build the experimental curve as
``amplitude = sqrt(power - background)`` over the full rfft range and correlate it
against ``|CTF| = |sin(chi)|``. The per-cycle rank-renormalization and the windowed
Pearson FRC both make absolute scale irrelevant, so this matches CTFFind's intent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch_ctf.ctf_utils import calculate_total_phase_shift

# CTFFind thresholds (ctffind.cpp:1759-1761).
_LOW_THRESHOLD = 0.1
_SIGNIFICANCE_THRESHOLD = 0.5  # half-bit criterion
_HIGH_THRESHOLD = 0.66
_FIRST_BIN_FRACTION = 0.1  # first_bin_to_check = 0.1 * number_of_bins


@dataclass
class CTFFitResolution:
    """Result of the CTF fit max-resolution computation."""

    resolution_angstroms: float
    last_bin_with_good_fit: int
    fit_frc: torch.Tensor  # per-bin FRC over the full rfft range
    frc_frequencies: torch.Tensor  # spatial frequencies (cycles/Angstrom), full range


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def chi_and_extrema_profile(
    spatial_freqs: torch.Tensor,
    *,
    defocus_um: float,
    voltage_kev: float,
    spherical_aberration_mm: float,
    amplitude_contrast: float,
    phase_shift_deg: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Phase aberration chi(s) and cumulative CTF-extrema count per frequency bin.

    Mirrors ``CTF::ReturnNumberOfExtremaBeforeSquaredSpatialFrequency``
    (ctf.cpp:151-171): ``n_extrema = floor(chi/pi + 0.5)``, reflected past the chi
    maximum so the count keeps rising once the aberration folds over.

    Parameters
    ----------
    spatial_freqs
        Spatial frequencies in cycles/Angstrom (full rfft range, length n_bins).
    defocus_um
        Fitted defocus in micrometers.
    voltage_kev
        Acceleration voltage in keV.
    spherical_aberration_mm
        Spherical aberration in mm.
    amplitude_contrast
        Amplitude contrast fraction.
    phase_shift_deg
        Additional phase shift in degrees.

    Returns
    -------
    chi
        Phase aberration at each bin (radians).
    n_extrema
        Cumulative extrema count per bin (float, monotonic non-decreasing).
    """
    sf = spatial_freqs.detach().cpu().float()
    chi = calculate_total_phase_shift(
        defocus_um=torch.as_tensor(float(defocus_um)),
        voltage_kv=torch.as_tensor(float(voltage_kev)),
        spherical_aberration_mm=torch.as_tensor(float(spherical_aberration_mm)),
        phase_shift_degrees=torch.as_tensor(float(phase_shift_deg)),
        amplitude_contrast_fraction=torch.as_tensor(float(amplitude_contrast)),
        fftfreq_grid_angstrom_squared=sf**2,
    ).reshape(-1)

    n_raw = torch.floor(chi / math.pi + 0.5)
    peak = int(torch.argmax(chi).item())
    n_extrema = n_raw.clone()
    # Reflect past the chi extremum (ctf.cpp:160-163): extrema keep accumulating.
    n_extrema[peak:] = n_raw[peak] + torch.abs(n_raw[peak:] - n_raw[peak])
    n_extrema = torch.abs(n_extrema)
    return chi, n_extrema


def _rank_sort(values: list[float]) -> list[int]:
    """
    Port of cisTEM ``rankSort`` (functions.cpp:966-985).

    Returns, for each input position, the index at which its value first appears in
    the ascending sort (ties share the lowest index). 0-based ranks.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0] * len(values)
    rank_value = None
    rank_index = 0
    for sorted_pos, original_index in enumerate(order):
        v = values[original_index]
        if v != rank_value:
            rank_value = v
            rank_index = sorted_pos
        result[original_index] = rank_index
    return result


def renormalize_1d_spectrum_for_frc(
    average: torch.Tensor,
    fit: torch.Tensor,
    n_extrema: torch.Tensor,
) -> torch.Tensor:
    """
    Per-cycle rank-renormalization (port of ``Renormalize1DSpectrumForFRC``).

    For each extremum-to-extremum interval (where the extrema count steps by ~1),
    locate the zero (lowest |fit| point), rank-sort the experimental values on the
    downslope and upslope separately, and map ranks through ``sin(rank * pi/2)`` so
    the experimental curve becomes |CTF|-shaped in [0, 1]. Falls back to per-interval
    min-max when the count is high. Operates on a copy; returns the renormalized
    spectrum (same length).

    Parameters
    ----------
    average
        Experimental (background-subtracted) amplitude per bin.
    fit
        |CTF| per bin (used only to locate the zero within an interval).
    n_extrema
        Cumulative extrema count per bin.
    """
    a = [float(x) for x in average.detach().cpu().tolist()]
    f = [float(x) for x in fit.detach().cpu().tolist()]
    ne = [float(x) for x in n_extrema.detach().cpu().tolist()]
    n = len(a)

    bin_of_previous_extremum = 0
    for bin_counter in range(1, n):
        delta = ne[bin_counter] - ne[bin_counter - 1]
        if not (0.9 <= delta <= 1.9):
            # CTF oscillating too quickly (or flat): leave this interval untouched.
            continue
        bin_of_current_extremum = bin_counter - 1
        if bin_of_previous_extremum > 0:
            if ne[bin_counter] < 7:
                # Rank-based renormalization (ctf.cpp:717-757).
                # Find the zero between the two extrema (lowest |fit|).
                bin_of_zero = (
                    bin_of_current_extremum - bin_of_previous_extremum
                ) // 2 + bin_of_previous_extremum
                for i in range(bin_of_previous_extremum, bin_of_current_extremum):
                    if f[i] < f[i - 1] and f[i] < f[i + 1]:
                        bin_of_zero = i

                # Downslope, including the zero.
                seg = [a[i] for i in range(bin_of_previous_extremum, bin_of_zero + 1)]
                if len(seg) > 1:
                    ranks = _rank_sort(seg)
                    denom = float(len(seg) - 1)
                    for k, i in enumerate(
                        range(bin_of_previous_extremum, bin_of_zero + 1)
                    ):
                        a[i] = math.sin((ranks[k] / denom) * math.pi * 0.5)

                # Upslope, after the zero.
                seg = [a[i] for i in range(bin_of_zero + 1, bin_of_current_extremum)]
                if len(seg) >= 1:
                    ranks = _rank_sort(seg)
                    denom = float(len(seg) + 1)
                    for k, i in enumerate(
                        range(bin_of_zero + 1, bin_of_current_extremum)
                    ):
                        a[i] = math.sin(((ranks[k] + 1) / denom) * math.pi * 0.5)
            else:
                # Simpler min-max normalization per interval (ctf.cpp:759-777).
                lo = min(
                    a[bin_of_previous_extremum:bin_of_current_extremum], default=0.0
                )
                hi = max(
                    a[bin_of_previous_extremum:bin_of_current_extremum], default=0.0
                )
                span = hi - lo
                for i in range(bin_of_previous_extremum, bin_of_current_extremum):
                    a[i] -= lo
                    if span > 0.0001:
                        a[i] /= span
        bin_of_previous_extremum = bin_of_current_extremum

    return torch.tensor(a, dtype=average.dtype)


def compute_frc_1d(
    average: torch.Tensor,
    fit: torch.Tensor,
    n_extrema: torch.Tensor,
    first_fit_bin: int,
) -> torch.Tensor:
    """
    Windowed Pearson FRC between renormalized spectrum and |CTF|.

    Port of ``ComputeFRCBetween1DSpectrumAndFit`` (ctf.cpp:786-878). The sliding
    half-window adapts to CTF oscillation spacing (~1.1x extremum spacing, floored at
    ``n_bins/40``, capped at ``n_bins/2 - 1``). Bins below ``first_fit_bin`` are set to
    1.0. Returns the per-bin FRC (length n_bins).
    """
    avg = [float(x) for x in average.detach().cpu().tolist()]
    ft = [float(x) for x in fit.detach().cpu().tolist()]
    ne = [float(x) for x in n_extrema.detach().cpu().tolist()]
    n = len(avg)

    minimum_half_width = max(n // 40, 1)
    half_window = [minimum_half_width] * n

    # Size the window from the local extremum spacing (ctf.cpp:807-823).
    bin_of_previous_extremum = 0
    for bin_counter in range(1, n):
        if ne[bin_counter] != ne[bin_counter - 1]:
            for i in range(bin_of_previous_extremum, bin_counter):
                w = max(
                    minimum_half_width,
                    int(
                        (1.0 + 0.1 * ne[bin_counter])
                        * float(bin_counter - bin_of_previous_extremum + 1)
                    ),
                )
                half_window[i] = min(w, n // 2 - 1)
            bin_of_previous_extremum = bin_counter
    half_window[0] = half_window[1] if n > 1 else minimum_half_width
    if bin_of_previous_extremum > 0:
        for bin_counter in range(bin_of_previous_extremum, n):
            half_window[bin_counter] = half_window[bin_of_previous_extremum - 1]

    frc = [0.0] * n
    for bin_counter in range(n):
        if bin_counter < first_fit_bin:
            frc[bin_counter] = 1.0
            continue
        hw = half_window[bin_counter]
        first_bin = bin_counter - hw
        last_bin = bin_counter + hw
        if first_bin < first_fit_bin:
            first_bin = first_fit_bin
            last_bin = first_bin + 2 * hw + 1
        if last_bin >= n:
            last_bin = n - 1
            first_bin = last_bin - 2 * hw - 1
        first_bin = max(first_bin, 0)

        count = last_bin - first_bin + 1
        if count <= 1:
            frc[bin_counter] = 0.0
            continue
        spectrum_mean = sum(avg[first_bin : last_bin + 1]) / count
        fit_mean = sum(ft[first_bin : last_bin + 1]) / count
        cross = 0.0
        s_sigma = 0.0
        f_sigma = 0.0
        for i in range(first_bin, last_bin + 1):
            ds = avg[i] - spectrum_mean
            df = ft[i] - fit_mean
            cross += ds * df
            s_sigma += ds * ds
            f_sigma += df * df
        if s_sigma > 0.0 and f_sigma > 0.0:
            # Pearson correlation coefficient (ctf.cpp:868).
            frc[bin_counter] = (
                cross
                / (math.sqrt(s_sigma / count) * math.sqrt(f_sigma / count))
                / count
            )
        else:
            frc[bin_counter] = 0.0

    return torch.tensor(frc, dtype=average.dtype)


def max_resolution_from_frc(
    fit_frc: torch.Tensor,
    spatial_freqs: torch.Tensor,
) -> tuple[float, int]:
    """
    Dual-threshold walk to find the last bin with a good fit, then convert to Angstroms.

    Port of the resolution-cutoff loop (ctffind.cpp:1756-1799). Walks from
    ``0.1 * n_bins`` and stops at the first bin where the FRC drops below a threshold,
    conditioned on having seen enough good bins (thresholds 0.1 / 0.5 / 0.66).

    Returns ``(resolution_angstroms, last_bin)``.
    """
    frc = [float(x) for x in fit_frc.detach().cpu().tolist()]
    n = len(frc)
    first_bin_to_check = int(_FIRST_BIN_FRACTION * n)

    n_above_low = 0
    n_above_high = 0
    n_above_significance = 0
    last_bin = -1
    if first_bin_to_check >= n:
        last_bin = 1
    else:
        for counter in range(first_bin_to_check, n):
            at_last = ((n_above_low > 3) and (frc[counter] < _LOW_THRESHOLD)) or (
                (n_above_high > 3) and (frc[counter] < _SIGNIFICANCE_THRESHOLD)
            )
            if at_last:
                last_bin = counter
                break
            if frc[counter] > _LOW_THRESHOLD:
                n_above_low += 1
            if frc[counter] > _SIGNIFICANCE_THRESHOLD:
                n_above_significance += 1
            if frc[counter] > _HIGH_THRESHOLD:
                n_above_high += 1
        if n_above_significance == n - first_bin_to_check:
            last_bin = n - 1  # good all the way out
        if n_above_significance == 0:
            last_bin = 1  # never any good fit
        # Loop completed without a break and not an all-good / all-bad edge case:
        # treat the fit as good across the checked range.
        if last_bin == -1:
            last_bin = n - 1
    last_bin = min(last_bin, n - 1)
    last_bin = max(last_bin, 1)

    sf = float(spatial_freqs.detach().cpu()[last_bin].item())
    resolution = float("inf") if sf <= 0.0 else 1.0 / sf
    return resolution, last_bin


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _smooth_background_boxcar(power_full: torch.Tensor) -> torch.Tensor:
    """
    Local-mean (boxcar) background over the full rfft range.

    Mirrors CTFFind's ``SpectrumBoxConvolution`` idea: a moving average sized to a few
    Thon-ring widths is subtracted so the oscillation becomes ~zero-mean. Unlike a
    stiff full-range spline this neither overshoots above the power (which would clamp
    whole bands to zero) nor hugs the curve tightly enough to absorb the rings.

    Window is ``n // 15`` bins (odd), edge-padded. Returned per bin (same length).
    """
    n = power_full.shape[0]
    k = max(3, n // 15)
    if k % 2 == 0:
        k += 1
    pad = k // 2
    padded = torch.nn.functional.pad(
        power_full.reshape(1, 1, -1), (pad, pad), mode="replicate"
    )
    background = torch.nn.functional.avg_pool1d(padded, kernel_size=k, stride=1)
    return background.reshape(-1)


def compute_ctf_fit_resolution(
    rotationally_averaged_power_spectrum: torch.Tensor,
    spatial_freqs: torch.Tensor,
    fit_mask: torch.Tensor,
    *,
    defocus_um: float,
    voltage_kev: float,
    spherical_aberration_mm: float,
    amplitude_contrast: float,
    phase_shift_deg: float,
) -> CTFFitResolution:
    """
    Compute CTFFind's "Thon rings with good fit up to X Angstroms" metric.

    Builds a full-range background-subtracted amplitude spectrum, renormalizes it per
    CTF cycle, computes the windowed Pearson FRC against |CTF|, and walks the
    dual-threshold criterion to find the resolution cutoff.

    Parameters
    ----------
    rotationally_averaged_power_spectrum
        Full rfft-length 1D power spectrum (e.g. ``_Background1DResult``'s field).
    spatial_freqs
        Spatial frequencies (cycles/Angstrom), same length, full range.
    fit_mask
        Boolean mask of the fitting band; its first True marks ``first_fit_bin``.
    defocus_um
        Fitted defocus in micrometers.
    voltage_kev
        Acceleration voltage in keV.
    spherical_aberration_mm
        Spherical aberration in mm.
    amplitude_contrast
        Amplitude contrast fraction.
    phase_shift_deg
        Additional phase shift in degrees.

    Returns
    -------
    CTFFitResolution
        resolution_angstroms, last_bin_with_good_fit, fit_frc, frc_frequencies.
    """
    power = rotationally_averaged_power_spectrum.detach().cpu().float().reshape(-1)
    sf = spatial_freqs.detach().cpu().float().reshape(-1)

    # Experimental amplitude with smooth background removed (CTFFind amplitude domain).
    # Subtract the background in the amplitude domain and keep the SIGNED residual:
    # the per-cycle rank-renormalization needs the trough (sub-background) values to
    # rank against the peaks. Clamping to zero here half-wave rectifies the rings and
    # collapses the renormalization to a degenerate binary pattern (CTFFind ranks the
    # signed background-subtracted amplitude, not a rectified one).
    amplitude_spectrum = torch.sqrt(torch.clamp(power, min=0.0))
    background = _smooth_background_boxcar(amplitude_spectrum)
    amplitude = amplitude_spectrum - background

    # Theoretical |CTF| and extrema profile from the fitted parameters.
    chi, n_extrema = chi_and_extrema_profile(
        sf,
        defocus_um=defocus_um,
        voltage_kev=voltage_kev,
        spherical_aberration_mm=spherical_aberration_mm,
        amplitude_contrast=amplitude_contrast,
        phase_shift_deg=phase_shift_deg,
    )
    fit_abs_ctf = torch.sin(chi).abs()

    # first_fit_bin = lowest bin inside the fitting band (ctffind.cpp:1726-1730).
    mask = fit_mask.detach().cpu().reshape(-1)
    nonzero = torch.nonzero(mask, as_tuple=False)
    first_fit_bin = int(nonzero[0].item()) if nonzero.numel() > 0 else 0

    average_renorm = renormalize_1d_spectrum_for_frc(amplitude, fit_abs_ctf, n_extrema)
    fit_frc = compute_frc_1d(average_renorm, fit_abs_ctf, n_extrema, first_fit_bin)
    resolution, last_bin = max_resolution_from_frc(fit_frc, sf)

    return CTFFitResolution(
        resolution_angstroms=resolution,
        last_bin_with_good_fit=last_bin,
        fit_frc=fit_frc,
        frc_frequencies=sf,
    )
