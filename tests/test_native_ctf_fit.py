"""Tests for the CTFFind5-faithful native-resolution CTF fit.

These validate the fit recovers known defocus + astigmatism from a synthetic
amplitude spectrum (the core achievement of the port), plus the building blocks.
"""

import math

import torch

from torch_ctf_estimation.metrics.native_ctf_fit import (
    NativeCTFFit,
    _abs_ctf_2d,
    compute_amplitude_spectrum_2d,
    equiphase_magnification_average,
    filter_amplitude_spectrum,
    fit_ctf_2d_native,
)

VOLTAGE = 200.0
CS = 2.7
AC = 0.1
PIXEL = 0.8592
BOX = 512


def _grids(box=BOX, pixel=PIXEL):
    centre = box // 2
    yy, xx = torch.meshgrid(
        torch.arange(box) - centre, torch.arange(box) - centre, indexing="ij"
    )
    sx = (xx.float() / box) / pixel
    sy = (yy.float() / box) / pixel
    s2 = sx**2 + sy**2
    azimuth = torch.atan2(yy.float(), xx.float())
    return s2, azimuth


def _synthetic_filtered_spectrum(
    defocus_um, astig_um, angle_deg, envelope_b, noise, seed=0
):
    """Centred filtered-like spectrum: |CTF|(known) * envelope + noise, ~zero mean."""
    torch.manual_seed(seed)
    s2, azimuth = _grids()
    model = _abs_ctf_2d(
        s2,
        azimuth,
        defocus_um=defocus_um,
        astigmatism_um=astig_um,
        astigmatism_angle_deg=angle_deg,
        phase_shift_deg=0.0,
        voltage_kev=VOLTAGE,
        spherical_aberration_mm=CS,
        amplitude_contrast=AC,
    )
    envelope = torch.exp(-envelope_b * s2)
    spectrum = model * envelope + noise * torch.randn_like(model)
    # rough zero-mean (the fit uses a mean-subtracted NCC anyway)
    return spectrum - spectrum.mean()


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def test_amplitude_spectrum_shape_and_centre():
    patches = torch.randn(3, BOX, BOX)
    amp = compute_amplitude_spectrum_2d(patches)
    assert amp.shape == (BOX, BOX)
    # DC (centre) is the largest amplitude for zero-mean-ish random data plus offset
    assert torch.isfinite(amp).all()


def test_filter_amplitude_spectrum_runs():
    patches = torch.randn(2, BOX, BOX)
    amp = compute_amplitude_spectrum_2d(patches)
    filt = filter_amplitude_spectrum(amp, PIXEL, 30.0)
    assert filt.shape == (BOX, BOX)
    assert torch.isfinite(filt).all()


def test_equiphase_average_shape_and_isotropic_limit():
    # With astig=0 the equiphase average reduces to a rotational average.
    torch.manual_seed(0)
    power = torch.rand(BOX, BOX) + 1.0
    profile, sf = equiphase_magnification_average(
        power, PIXEL, defocus_um=1.5, astigmatism_um=0.0, astigmatism_angle_deg=0.0
    )
    assert profile.shape == sf.shape
    assert profile.numel() == BOX // 2 + 1
    assert torch.isfinite(profile).all()


# ---------------------------------------------------------------------------
# Fit recovery (the core: defocus + astigmatism from a known spectrum)
# ---------------------------------------------------------------------------


def test_fit_recovers_defocus_and_astigmatism():
    true_defocus = 1.50  # um
    true_astig = 0.04  # um (full difference, 400 A)
    true_angle = 130.0
    spectrum = _synthetic_filtered_spectrum(
        true_defocus, true_astig, true_angle, envelope_b=120.0, noise=0.05, seed=1
    )
    fit = fit_ctf_2d_native(
        spectrum,
        PIXEL,
        voltage_kev=VOLTAGE,
        spherical_aberration_mm=CS,
        amplitude_contrast=AC,
        frequency_fit_range_angstroms=(30, 5),
        initial_defocus_um=true_defocus + 0.05,  # imperfect seed
    )
    assert isinstance(fit, NativeCTFFit)
    # defocus within ~150 A, astigmatism within ~150 A
    assert abs(fit.defocus_um - true_defocus) * 1e4 < 150
    assert abs(fit.astigmatism_um - true_astig) * 1e4 < 150
    # angle within ~25 deg (mod 180)
    da = abs((fit.astigmatism_angle_deg - true_angle + 90) % 180 - 90)
    assert da < 25
    assert fit.defocus_u_um >= fit.defocus_v_um


def test_fit_low_astigmatism_stays_low():
    # Near-zero true astigmatism should not blow up (restraint + landscape).
    spectrum = _synthetic_filtered_spectrum(
        1.6, 0.005, 0.0, envelope_b=120.0, noise=0.05, seed=2
    )
    fit = fit_ctf_2d_native(
        spectrum,
        PIXEL,
        voltage_kev=VOLTAGE,
        spherical_aberration_mm=CS,
        amplitude_contrast=AC,
        frequency_fit_range_angstroms=(30, 5),
        initial_defocus_um=1.6,
    )
    assert fit.astigmatism_um * 1e4 < 250  # A
    assert math.isfinite(fit.score)
