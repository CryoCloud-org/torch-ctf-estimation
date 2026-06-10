"""Tests for the CTFFind-style CTF fit max-resolution metric."""

import math

import torch
from torch_ctf import calculate_ctf_1d

from torch_ctf_estimation.metrics.fit_resolution import (
    _rank_sort,
    chi_and_extrema_profile,
    compute_ctf_fit_resolution,
    compute_frc_1d,
    max_resolution_from_frc,
)

VOLTAGE = 300.0
CS = 2.7
AC = 0.1
PIXEL = 1.0


def _synthetic_1d_power(
    image_sidelength: int,
    defocus_um: float,
    envelope_b: float,
    noise_std: float,
    seed: int = 0,
):
    """
    Build a synthetic rotationally-averaged 1D power spectrum + freqs + fit_mask.

    power = CTF^2 * exp(-B s^2) + smooth background + noise. Larger B (or noise)
    makes the Thon rings vanish at lower frequency -> worse resolution.
    """
    torch.manual_seed(seed)
    n = image_sidelength // 2 + 1
    freqs = torch.fft.rfftfreq(image_sidelength)  # cycles/px == cycles/Å (pixel=1)
    spatial_freqs = freqs / PIXEL

    ctf = calculate_ctf_1d(
        defocus=torch.tensor([defocus_um]),
        voltage=VOLTAGE,
        spherical_aberration=CS,
        amplitude_contrast=AC,
        phase_shift=0.0,
        pixel_size=PIXEL,
        n_samples=n,
        oversampling_factor=1,
    ).squeeze(0)
    envelope = torch.exp(-envelope_b * spatial_freqs**2)
    background = 5.0 * torch.exp(-spatial_freqs * 3.0) + 0.5
    noise = noise_std * torch.randn(n)
    power = (ctf**2) * envelope + background + noise
    power = torch.clamp(power, min=1e-6)

    low_ang, high_ang = 30.0, 5.0
    fit_mask = torch.logical_and(
        spatial_freqs >= 1.0 / low_ang, spatial_freqs <= 1.0 / high_ang
    )
    return power, spatial_freqs, fit_mask


def _resolution(power, spatial_freqs, fit_mask, defocus_um):
    return compute_ctf_fit_resolution(
        power,
        spatial_freqs,
        fit_mask,
        defocus_um=defocus_um,
        voltage_kev=VOLTAGE,
        spherical_aberration_mm=CS,
        amplitude_contrast=AC,
        phase_shift_deg=0.0,
    )


# ---------------------------------------------------------------------------
# Unit tests for building blocks
# ---------------------------------------------------------------------------


def test_rank_sort_matches_ctffind_semantics():
    # Ascending sort positions; ties share the lowest index (0-based).
    assert _rank_sort([3.0, 1.0, 2.0]) == [2, 0, 1]
    assert _rank_sort([5.0, 5.0, 1.0]) == [1, 1, 0]


def test_max_resolution_from_frc_monotone_drop():
    # FRC stays high then drops below 0.5 after enough good bins.
    n = 100
    spatial_freqs = torch.linspace(0, 0.5, n)
    frc = torch.ones(n)
    frc[60:] = 0.0  # collapses at bin 60
    resolution, last_bin = max_resolution_from_frc(frc, spatial_freqs)
    # Needs >3 bins above high threshold before the <0.5 break; break lands at 60.
    assert last_bin == 60
    assert math.isclose(resolution, 1.0 / float(spatial_freqs[60]), rel_tol=1e-6)


def test_max_resolution_all_good_returns_nyquist_bin():
    n = 80
    spatial_freqs = torch.linspace(0, 0.5, n)
    frc = torch.ones(n)
    _resolution_value, last_bin = max_resolution_from_frc(frc, spatial_freqs)
    assert last_bin == n - 1


def test_max_resolution_all_bad_returns_low_bin():
    n = 80
    spatial_freqs = torch.linspace(0, 0.5, n)
    frc = torch.zeros(n)
    _resolution_value, last_bin = max_resolution_from_frc(frc, spatial_freqs)
    assert last_bin == 1


def test_extrema_profile_is_monotonic():
    freqs = torch.fft.rfftfreq(256)
    _chi, n_extrema = chi_and_extrema_profile(
        freqs,
        defocus_um=1.5,
        voltage_kev=VOLTAGE,
        spherical_aberration_mm=CS,
        amplitude_contrast=AC,
        phase_shift_deg=0.0,
    )
    diffs = n_extrema[1:] - n_extrema[:-1]
    assert torch.all(diffs >= -1e-6)  # non-decreasing
    assert n_extrema[-1] > 0


def test_frc_high_for_perfect_match():
    # When the renormalized spectrum equals |CTF|, the windowed FRC is ~1 in band.
    n = 200
    freqs = torch.linspace(0, 0.5, n)
    _chi, n_extrema = chi_and_extrema_profile(
        freqs,
        defocus_um=1.0,
        voltage_kev=VOLTAGE,
        spherical_aberration_mm=CS,
        amplitude_contrast=AC,
        phase_shift_deg=0.0,
    )
    chi, _ = chi_and_extrema_profile(
        freqs,
        defocus_um=1.0,
        voltage_kev=VOLTAGE,
        spherical_aberration_mm=CS,
        amplitude_contrast=AC,
        phase_shift_deg=0.0,
    )
    fit = torch.sin(chi).abs()
    frc = compute_frc_1d(fit, fit, n_extrema, first_fit_bin=20)
    assert torch.all(frc[:20] == 1.0)
    assert frc[40:120].mean() > 0.8


# ---------------------------------------------------------------------------
# End-to-end behaviour on synthetic spectra
# ---------------------------------------------------------------------------


def test_resolution_finite_and_in_range():
    power, sf, mask = _synthetic_1d_power(
        512, defocus_um=1.5, envelope_b=80.0, noise_std=0.02
    )
    res = _resolution(power, sf, mask, defocus_um=1.5)
    assert math.isfinite(res.resolution_angstroms)
    # Resolution between Nyquist (2 Å) and the low-res fit edge (30 Å).
    assert 2.0 <= res.resolution_angstroms <= 30.0
    assert res.fit_frc.shape == sf.shape


def test_resolution_degrades_with_stronger_envelope():
    # Larger B kills high-frequency rings sooner -> worse (larger Å) resolution.
    p_sharp, sf, mask = _synthetic_1d_power(
        512, 1.5, envelope_b=40.0, noise_std=0.02, seed=1
    )
    p_soft, _, _ = _synthetic_1d_power(
        512, 1.5, envelope_b=400.0, noise_std=0.02, seed=1
    )
    res_sharp = _resolution(p_sharp, sf, mask, 1.5).resolution_angstroms
    res_soft = _resolution(p_soft, sf, mask, 1.5).resolution_angstroms
    assert res_soft > res_sharp


def test_resolution_degrades_with_more_noise():
    p_clean, sf, mask = _synthetic_1d_power(
        512, 1.5, envelope_b=80.0, noise_std=0.01, seed=2
    )
    p_noisy, _, _ = _synthetic_1d_power(
        512, 1.5, envelope_b=80.0, noise_std=0.5, seed=2
    )
    res_clean = _resolution(p_clean, sf, mask, 1.5).resolution_angstroms
    res_noisy = _resolution(p_noisy, sf, mask, 1.5).resolution_angstroms
    assert res_noisy >= res_clean
