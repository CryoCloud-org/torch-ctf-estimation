"""Native-resolution CTF fit, ported faithfully from CTFFind5.

The fit-time pipeline downsamples to >=3 A (Nyquist 6 A), where astigmatism — a
high-frequency effect — is unidentifiable and the 2D astigmatism optimiser diverges.
CTFFind fits at the native pixel size. This module reproduces CTFFind5's fit so the
``ctf_fit_resolution`` metric can run on a faithful defocus + astigmatism estimate.

Ported pieces (CTFFind5, ``cisTEM/src``):
- Filtered amplitude spectrum — ``Image::ComputeFilteredAmplitudeSpectrumFull2D``
  (``core/image.cpp:7072``): whiten by sigma in a low-resolution annulus, cap the
  central cross, subtract a box-convolution background, clip outliers.
- 2D scoring — ``Image::GetCorrelationWithCTF`` / ``QuickCorrelationWithCTF``
  (``core/image.cpp:6129``) and ``CtffindObjectiveFunction``
  (``programs/ctffind/ctffind_objective_functions.cpp:5``): the true (mean-subtracted)
  normalised cross-correlation between the filtered amplitude spectrum and
  ``|CTF| = |sin(chi)|`` over the fit band, excluding the central cross, minus an
  astigmatism penalty ``astig^2 * 0.5 / tol^2 / N``.
- Search — brute-force over defocus/astigmatism/azimuth then a local refinement, in the
  spirit of CTFFind's ``BruteForceSearch`` + ``ConjugateGradient``
  (``programs/ctffind/ctffind.cpp:1466``).

Conventions here: defocus in micrometers, spatial frequency in cycles/Angstrom (so
``calculate_total_phase_shift`` takes ``s^2`` in Angstrom^-2). Astigmatism magnitude is
the full difference ``defocus_u - defocus_v`` (um); ``defocus_u >= defocus_v``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import einops
import torch
import torch.nn.functional as F
from torch_ctf.ctf_utils import calculate_total_phase_shift

# CTFFind defaults / constants
_CENTRAL_CROSS_HALF_WIDTH = 10  # pixels, GetCorrelationWithCTF
_DEFAULT_ASTIGMATISM_TOLERANCE_ANGSTROMS = 100.0  # "expected (tolerated) astigmatism"


@dataclass
class NativeCTFFit:
    """Result of the native-resolution CTF fit."""

    defocus_u_um: float  # higher principal defocus
    defocus_v_um: float  # lower principal defocus
    defocus_um: float  # mean = 0.5 * (u + v)
    astigmatism_um: float  # full difference u - v (>= 0)
    astigmatism_angle_deg: float  # azimuth of defocus_u, in [0, 180)
    phase_shift_deg: float
    score: float  # final NCC (penalty-free) at the optimum


# ---------------------------------------------------------------------------
# Amplitude spectrum + CTFFind filtering
# ---------------------------------------------------------------------------


def compute_amplitude_spectrum_2d(
    patches: torch.Tensor,
) -> torch.Tensor:
    """
    Mean amplitude spectrum |F| of square patches, fftshifted (DC at the centre).

    Parameters
    ----------
    patches
        Real-space patches, shape (..., box, box).

    Returns
    -------
    torch.Tensor
        (box, box) mean amplitude spectrum, centred.
    """
    patches = patches.reshape(-1, patches.shape[-2], patches.shape[-1])
    f = torch.fft.fftshift(torch.fft.fft2(patches), dim=(-2, -1))
    return f.abs().mean(0)


def filter_amplitude_spectrum(
    amplitude: torch.Tensor,
    pixel_spacing_angstroms: float,
    minimum_resolution_angstroms: float,
) -> torch.Tensor:
    """
    CTFFind ``ComputeFilteredAmplitudeSpectrumFull2D`` (image.cpp:7072).

    Whiten by sigma measured in a low-resolution annulus, cap the central cross,
    subtract a box-convolution background (box size ``box*px/minres*sqrt(2)``), and clip
    outliers to the local-max. Operates on a centred amplitude spectrum.
    """
    box = amplitude.shape[-1]
    centre = box // 2
    yy, xx = torch.meshgrid(
        torch.arange(box, device=amplitude.device) - centre,
        torch.arange(box, device=amplitude.device) - centre,
        indexing="ij",
    )
    radius = torch.sqrt(yy.float() ** 2 + xx.float() ** 2)

    # sigma/average over the fitting annulus (>= box*px/minres), then whiten
    r_inner = box * pixel_spacing_angstroms / minimum_resolution_angstroms
    annulus = radius >= r_inner
    sigma = amplitude[annulus].std()
    average = amplitude[annulus].mean()
    spectrum = amplitude / sigma

    # cap the central cross (SetMaximumValueOnCentralCross)
    cross = (yy.abs() <= 1) | (xx.abs() <= 1)
    cap = average / sigma + 10.0
    spectrum = torch.where(cross, torch.clamp(spectrum, max=cap), spectrum)

    # box-convolution low-pass background, then subtract
    k = int(box * pixel_spacing_angstroms / minimum_resolution_angstroms * (2.0**0.5))
    if k % 2 == 0:
        k += 1
    k = max(k, 3)
    background = F.avg_pool2d(
        spectrum.view(1, 1, box, box),
        kernel_size=k,
        stride=1,
        padding=k // 2,
        count_include_pad=False,
    ).view(box, box)
    spectrum = spectrum - background

    # clip very high values (SetMaximumValue(ReturnMaximumValue(3,3))) — a robust local
    # max; a global max clamp is a faithful, simpler stand-in that never raises values.
    spectrum = torch.clamp(spectrum, max=float(spectrum.max()))
    return spectrum


# ---------------------------------------------------------------------------
# Scoring (CTFFind GetCorrelationWithCTF / QuickCorrelationWithCTF)
# ---------------------------------------------------------------------------


def _abs_ctf_2d(
    s2: torch.Tensor,
    azimuth: torch.Tensor,
    *,
    defocus_um: torch.Tensor | float,
    astigmatism_um: torch.Tensor | float,
    astigmatism_angle_deg: torch.Tensor | float,
    phase_shift_deg: torch.Tensor | float,
    voltage_kev: float,
    spherical_aberration_mm: float,
    amplitude_contrast: float,
) -> torch.Tensor:
    """|CTF| = |sin(chi)| with azimuth-dependent defocus (CTF::Evaluate)."""
    angle_rad = (
        astigmatism_angle_deg
        if isinstance(astigmatism_angle_deg, torch.Tensor)
        else torch.tensor(float(astigmatism_angle_deg))
    ) * (torch.pi / 180.0)
    # defocus(azimuth) = mean + 0.5*astig*cos(2*(azimuth - angle)); astig = u - v
    d_eff = defocus_um + 0.5 * astigmatism_um * torch.cos(2.0 * (azimuth - angle_rad))
    chi = calculate_total_phase_shift(
        defocus_um=d_eff,
        voltage_kv=torch.as_tensor(float(voltage_kev)),
        spherical_aberration_mm=torch.as_tensor(float(spherical_aberration_mm)),
        phase_shift_degrees=(
            phase_shift_deg
            if isinstance(phase_shift_deg, torch.Tensor)
            else torch.as_tensor(float(phase_shift_deg))
        ),
        amplitude_contrast_fraction=torch.as_tensor(float(amplitude_contrast)),
        fftfreq_grid_angstrom_squared=s2,
    )
    return torch.sin(chi).abs()


def _score(
    spectrum_band: torch.Tensor,
    s2_band: torch.Tensor,
    azimuth_band: torch.Tensor,
    n_band: int,
    *,
    defocus_um: torch.Tensor | float,
    astigmatism_um: torch.Tensor | float,
    astigmatism_angle_deg: torch.Tensor | float,
    phase_shift_deg: torch.Tensor | float,
    voltage_kev: float,
    spherical_aberration_mm: float,
    amplitude_contrast: float,
    astigmatism_tolerance_angstroms: float,
) -> torch.Tensor:
    """
    True (mean-subtracted) NCC between filtered spectrum and |CTF|, minus astig penalty.

    Matches ``QuickCorrelationWithCTF`` (image.cpp:6259).
    """
    model = _abs_ctf_2d(
        s2_band,
        azimuth_band,
        defocus_um=defocus_um,
        astigmatism_um=astigmatism_um,
        astigmatism_angle_deg=astigmatism_angle_deg,
        phase_shift_deg=phase_shift_deg,
        voltage_kev=voltage_kev,
        spherical_aberration_mm=spherical_aberration_mm,
        amplitude_contrast=amplitude_contrast,
    )
    y = spectrum_band - spectrum_band.mean()
    m = model - model.mean()
    ncc = (y * m).sum() / (y.norm() * m.norm() + 1e-12)

    astig_full_angstroms = torch.abs(
        astigmatism_um * 1e4
        if isinstance(astigmatism_um, torch.Tensor)
        else torch.tensor(astigmatism_um * 1e4)
    )
    penalty = (
        astig_full_angstroms**2
        * 0.5
        / (astigmatism_tolerance_angstroms**2)
        / float(n_band)
    )
    return ncc - penalty


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


def fit_ctf_2d_native(
    filtered_spectrum: torch.Tensor,
    pixel_spacing_angstroms: float,
    *,
    voltage_kev: float,
    spherical_aberration_mm: float,
    amplitude_contrast: float,
    frequency_fit_range_angstroms: tuple[float, float],
    initial_defocus_um: float,
    defocus_half_range_um: float = 0.3,
    max_astigmatism_um: float = 0.2,
    phase_shift_deg: float = 0.0,
    astigmatism_tolerance_angstroms: float = _DEFAULT_ASTIGMATISM_TOLERANCE_ANGSTROMS,
    refine_steps: int = 120,
) -> NativeCTFFit:
    """
    Fit defocus + astigmatism on a filtered amplitude spectrum (CTFFind-faithful score).

    Coarse brute-force over (mean defocus, astigmatism, azimuth) seeded from
    ``initial_defocus_um``, then a gradient refinement of all three on the exact score.

    Parameters
    ----------
    filtered_spectrum
        Centred filtered amplitude spectrum from ``filter_amplitude_spectrum``.
    pixel_spacing_angstroms
        Native pixel spacing.
    voltage_kev, spherical_aberration_mm, amplitude_contrast
        Optics.
    frequency_fit_range_angstroms
        (low, high) resolution bounds in Angstroms (e.g. (30, 5)).
    initial_defocus_um
        Mean-defocus seed (e.g. the 1D fit).
    defocus_half_range_um
        Half-range of the mean-defocus brute search around the seed.
    max_astigmatism_um
        Upper bound for the astigmatism brute search (full difference).
    phase_shift_deg
        Fixed additional phase shift (degrees).
    astigmatism_tolerance_angstroms
        CTFFind ``astigmatism_tolerance`` for the restraint penalty.
    refine_steps
        Gradient-refinement iterations.

    Returns
    -------
    NativeCTFFit
    """
    box = filtered_spectrum.shape[-1]
    device = filtered_spectrum.device
    centre = box // 2
    yy, xx = torch.meshgrid(
        torch.arange(box, device=device) - centre,
        torch.arange(box, device=device) - centre,
        indexing="ij",
    )
    sx = (xx.float() / box) / pixel_spacing_angstroms
    sy = (yy.float() / box) / pixel_spacing_angstroms
    s2 = sx**2 + sy**2
    s = torch.sqrt(s2)
    azimuth = torch.atan2(yy.float(), xx.float())

    low_ang, high_ang = frequency_fit_range_angstroms
    cross = (yy.abs() > _CENTRAL_CROSS_HALF_WIDTH) & (
        xx.abs() > _CENTRAL_CROSS_HALF_WIDTH
    )
    band = (s >= 1.0 / low_ang) & (s <= 1.0 / high_ang) & cross
    spectrum_band = filtered_spectrum[band]
    s2_band = s2[band]
    azimuth_band = azimuth[band]
    n_band = int(band.sum().item())

    common = {
        "voltage_kev": voltage_kev,
        "spherical_aberration_mm": spherical_aberration_mm,
        "amplitude_contrast": amplitude_contrast,
        "astigmatism_tolerance_angstroms": astigmatism_tolerance_angstroms,
    }

    # --- Coarse brute-force over (mean defocus, astigmatism, azimuth) ---
    d_grid = torch.arange(
        initial_defocus_um - defocus_half_range_um,
        initial_defocus_um + defocus_half_range_um + 1e-6,
        0.03,
    )
    a_grid = torch.arange(0.0, max_astigmatism_um + 1e-6, 0.01)
    ang_grid = torch.arange(0.0, 180.0, 15.0)
    best = (-1e9, float(initial_defocus_um), 0.0, 0.0)
    for d in d_grid.tolist():
        for a in a_grid.tolist():
            for ang in ang_grid.tolist():
                sc = float(
                    _score(
                        spectrum_band,
                        s2_band,
                        azimuth_band,
                        n_band,
                        defocus_um=d,
                        astigmatism_um=a,
                        astigmatism_angle_deg=ang,
                        phase_shift_deg=phase_shift_deg,
                        **common,
                    )
                )
                if sc > best[0]:
                    best = (sc, d, a, ang)

    # --- Gradient refinement of (defocus, astigmatism, azimuth) ---
    d_p = torch.tensor(best[1], requires_grad=True)
    a_raw = torch.tensor(_inverse_softplus(max(best[2], 1e-4)), requires_grad=True)
    ang_rad = torch.tensor(best[3] * torch.pi / 180.0)
    u = torch.tensor(float(torch.cos(ang_rad)), requires_grad=True)
    v = torch.tensor(float(torch.sin(ang_rad)), requires_grad=True)
    opt = torch.optim.Adam(
        [
            {"params": [d_p], "lr": 1e-3},
            {"params": [a_raw], "lr": 5e-3},
            {"params": [u, v], "lr": 2e-2},
        ]
    )
    for _ in range(refine_steps):
        astig = F.softplus(a_raw)
        astig = torch.clamp(astig, max=max_astigmatism_um)
        angle_deg = torch.atan2(v, u) * (180.0 / torch.pi)
        loss = -_score(
            spectrum_band,
            s2_band,
            azimuth_band,
            n_band,
            defocus_um=d_p,
            astigmatism_um=astig,
            astigmatism_angle_deg=angle_deg,
            phase_shift_deg=phase_shift_deg,
            **common,
        )
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        defocus_mean = float(d_p.detach())
        astig = float(torch.clamp(F.softplus(a_raw), max=max_astigmatism_um))
        angle_deg = float(torch.atan2(v, u) * (180.0 / torch.pi)) % 180.0
        final_score = float(
            _score(
                spectrum_band,
                s2_band,
                azimuth_band,
                n_band,
                defocus_um=defocus_mean,
                astigmatism_um=astig,
                astigmatism_angle_deg=angle_deg,
                phase_shift_deg=phase_shift_deg,
                # report penalty-free NCC at the optimum
                **{**common, "astigmatism_tolerance_angstroms": 1e12},
            )
        )

    return NativeCTFFit(
        defocus_u_um=defocus_mean + 0.5 * astig,
        defocus_v_um=defocus_mean - 0.5 * astig,
        defocus_um=defocus_mean,
        astigmatism_um=astig,
        astigmatism_angle_deg=angle_deg,
        phase_shift_deg=float(phase_shift_deg),
        score=final_score,
    )


def _inverse_softplus(y: float) -> float:
    """Inverse of softplus, for initialising a softplus-parameterised value."""
    return math.log(math.expm1(max(y, 1e-6)))


# ---------------------------------------------------------------------------
# Astigmatism-corrected 1D average (equiphase magnification resampling)
# ---------------------------------------------------------------------------


def equiphase_magnification_average(
    power_centered: torch.Tensor,
    pixel_spacing_angstroms: float,
    *,
    defocus_um: float,
    astigmatism_um: float,
    astigmatism_angle_deg: float,
    n_theta: int = 240,
    oversample: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Astigmatism-corrected 1D power profile by equiphase-magnification resampling.

    For each target isotropic frequency ``s`` and azimuth ``theta``, the matching
    astigmatic Thon ring sits at ``s * sqrt(d_mean / d(theta))`` with
    ``d(theta) = d_mean + 0.5*astig*cos(2*(theta - angle))`` (the defocus magnification
    that maps an astigmatic ring back to its isotropic radius). Sampling the 2D power
    there and averaging over azimuth de-astigmatises the rings, recovering the sharp
    high-frequency contrast that a plain rotational average smears out — the torch
    analogue of CTFFind's ``ComputeRotationalAverageOfPowerSpectrum``.

    Parameters
    ----------
    power_centered
        Centred 2D power spectrum |F|^2 (DC at the middle), shape (box, box).
    pixel_spacing_angstroms
        Native pixel spacing.
    defocus_um, astigmatism_um, astigmatism_angle_deg
        Fitted CTF (mean defocus, full-difference astigmatism, azimuth of defocus_u).
    n_theta
        Number of azimuthal samples averaged per shell.
    oversample
        Radial oversampling factor for the output profile (1.0 = ``box//2+1`` bins).

    Returns
    -------
    (profile, spatial_freqs)
        ``profile`` is the 1D power profile (``box//2+1`` bins times oversample);
        ``spatial_freqs`` are the matching spatial frequencies (cycles/Angstrom).
    """
    box = power_centered.shape[-1]
    device = power_centered.device
    n = int((box // 2 + 1) * oversample)
    nyquist = 0.5 / pixel_spacing_angstroms
    sf = torch.linspace(0.0, nyquist, n, device=device)
    thetas = torch.arange(n_theta, device=device) * (2.0 * torch.pi / n_theta)
    angle_rad = astigmatism_angle_deg * torch.pi / 180.0
    d_theta = defocus_um + 0.5 * astigmatism_um * torch.cos(2.0 * (thetas - angle_rad))
    magnification = torch.sqrt(defocus_um / d_theta)  # (n_theta,)

    power4 = power_centered.view(1, 1, box, box)
    cos_t = torch.cos(thetas)
    sin_t = torch.sin(thetas)
    profile = torch.zeros(n, device=device)
    for i in range(n):
        s = sf[i] * magnification  # (n_theta,) cycles/Angstrom
        # cycles/Angstrom -> centred pixel index (DC at box//2): box/2 + s*px*box
        cx = box // 2 + s * cos_t * pixel_spacing_angstroms * box
        cy = box // 2 + s * sin_t * pixel_spacing_angstroms * box
        grid = torch.stack(
            [2.0 * cx / (box - 1) - 1.0, 2.0 * cy / (box - 1) - 1.0], dim=-1
        ).reshape(1, 1, n_theta, 2)
        sampled = F.grid_sample(
            power4, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        profile[i] = sampled.reshape(-1).mean()
    return profile, sf


def _azimuth_for_1d_plots(astigmatism_angle_deg: float) -> float:
    """Mid-defocus azimuth (rad) — ``ReturnAzimuthToUseFor1DPlots`` (ctf.cpp:618)."""
    min_dist = 10.0 / 180.0 * math.pi
    az = astigmatism_angle_deg * math.pi / 180.0 + math.pi * 0.25
    dist = math.fmod(az, math.pi * 0.5)
    if abs(dist) < min_dist:
        az = min_dist if dist > 0.0 else -min_dist
    if abs(dist) > 0.5 * math.pi - min_dist:
        az = (0.5 * math.pi - min_dist) if dist > 0.0 else (-0.5 * math.pi + min_dist)
    return az


def _chi_grid(
    s2: torch.Tensor,
    azimuth: torch.Tensor,
    *,
    defocus_um: float,
    astigmatism_um: float,
    astigmatism_angle_deg: float,
    phase_shift_deg: float,
    voltage_kev: float,
    spherical_aberration_mm: float,
    amplitude_contrast: float,
) -> torch.Tensor:
    """Phase aberration chi over a frequency/azimuth grid (astigmatic defocus)."""
    angle_rad = astigmatism_angle_deg * math.pi / 180.0
    d_eff = defocus_um + 0.5 * astigmatism_um * torch.cos(2.0 * (azimuth - angle_rad))
    return calculate_total_phase_shift(
        defocus_um=d_eff,
        voltage_kv=torch.as_tensor(float(voltage_kev)),
        spherical_aberration_mm=torch.as_tensor(float(spherical_aberration_mm)),
        phase_shift_degrees=torch.as_tensor(float(phase_shift_deg)),
        amplitude_contrast_fraction=torch.as_tensor(float(amplitude_contrast)),
        fftfreq_grid_angstrom_squared=s2,
    )


def ctffind_rotational_average(
    filtered_spectrum: torch.Tensor,
    pixel_spacing_angstroms: float,
    fit: NativeCTFFit,
    *,
    voltage_kev: float,
    spherical_aberration_mm: float,
    amplitude_contrast: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Exact CTFFind ``ComputeRotationalAverageOfPowerSpectrum`` (spectrum_image.cpp:243).

    Each pixel of the (centred) filtered amplitude spectrum is assigned to the 1D bin
    with the same number-of-preceding-CTF-extrema and the closest signed CTF value
    (``ReturnSpectrumBinNumber``, ctf.cpp:880), de-astigmatising the Thon rings. Uses
    ``box*sqrt(2)/2`` bins (out to the diagonal Nyquist), matching CTFFind.

    Returns ``(average, average_fit, number_of_extrema_profile, spatial_freqs)`` where
    ``average`` is the per-bin mean of the filtered spectrum, ``average_fit`` is
    ``|CTF|`` along the mid-defocus azimuth; ``spatial_freqs`` in cycles/Angstrom.
    """
    box = filtered_spectrum.shape[-1]
    device = filtered_spectrum.device
    centre = box // 2
    voxel = 1.0 / box  # fourier_voxel_size (cycles/pixel)
    n_bins = int(box / 2 * math.sqrt(2.0)) + 1

    azimuth_mid = _azimuth_for_1d_plots(fit.astigmatism_angle_deg)
    common = {
        "defocus_um": fit.defocus_um,
        "astigmatism_um": fit.astigmatism_um,
        "astigmatism_angle_deg": fit.astigmatism_angle_deg,
        "phase_shift_deg": fit.phase_shift_deg,
        "voltage_kev": voltage_kev,
        "spherical_aberration_mm": spherical_aberration_mm,
        "amplitude_contrast": amplitude_contrast,
    }

    # 1D profile along the mid-defocus azimuth
    k = torch.arange(n_bins, device=device).float()
    sf = k * voxel / pixel_spacing_angstroms  # cycles/Angstrom
    s2_1d = sf**2
    az_1d = torch.full_like(s2_1d, azimuth_mid)
    chi_1d = _chi_grid(s2_1d, az_1d, **common)
    cv_profile = -torch.sin(chi_1d)  # signed, == CTF::Evaluate
    ne_profile = torch.floor(chi_1d / math.pi + 0.5).abs()
    average_fit = cv_profile.abs()

    # 2D extrema + signed CTF value per pixel
    yy, xx = torch.meshgrid(
        torch.arange(box, device=device) - centre,
        torch.arange(box, device=device) - centre,
        indexing="ij",
    )
    i_logi = xx.float() * voxel
    j_logi = yy.float() * voxel
    s2_2d = (i_logi**2 + j_logi**2) / (pixel_spacing_angstroms**2)
    az_2d = torch.atan2(j_logi, i_logi)
    chi_2d = _chi_grid(s2_2d, az_2d, **common)
    cv_2d = -torch.sin(chi_2d)
    ne_2d = torch.floor(chi_2d / math.pi + 0.5).abs()

    # --- ReturnSpectrumBinNumber, vectorised: match extrema, then closest CTF value ---
    ne_prof_int = ne_profile.round().long()
    ne_pix = ne_2d.round().long().reshape(-1)
    cv_pix = cv_2d.reshape(-1)
    spec_pix = filtered_spectrum.reshape(-1)
    chosen = torch.full_like(ne_pix, -1)

    last_extrema = int(ne_prof_int[-1].item())
    over = ne_pix > last_extrema
    chosen[over] = n_bins - 1

    remaining = ~over
    for e in torch.unique(ne_pix[remaining]).tolist():
        bins_e = torch.nonzero(ne_prof_int == e, as_tuple=False).reshape(-1)
        if bins_e.numel() == 0:
            continue
        mask = remaining & (ne_pix == e)
        if not bool(mask.any()):
            continue
        cvp = cv_profile[bins_e]  # (nb_e,)
        diff = (cv_pix[mask].unsqueeze(1) - cvp.unsqueeze(0)).abs()  # (npix, nb_e)
        chosen[mask] = bins_e[diff.argmin(dim=1)]

    valid = chosen >= 0
    average = torch.zeros(n_bins, device=device)
    counts = torch.zeros(n_bins, device=device)
    average.index_add_(0, chosen[valid], spec_pix[valid])
    counts.index_add_(0, chosen[valid], torch.ones_like(spec_pix[valid]))
    average = torch.where(
        counts > 0, average / counts.clamp(min=1), torch.zeros_like(average)
    )

    return average, average_fit, ne_profile, sf


# ---------------------------------------------------------------------------
# Orchestrator: native CTF fit + astig-corrected average + FRC resolution
# ---------------------------------------------------------------------------


def compute_native_ctf_fit_resolution(
    image: torch.Tensor,
    *,
    pixel_spacing_angstroms: float,
    voltage_kev: float,
    spherical_aberration_mm: float,
    amplitude_contrast: float,
    frequency_fit_range_angstroms: tuple[float, float],
    initial_defocus_um: float,
    box: int = 512,
    phase_shift_deg: float = 0.0,
    target_pixel_size_after_resampling: float = 1.4,
):
    """
    Full CTFFind-faithful path: native fit -> discrete astig-corrected average -> FRC.

    1. Extract native-pixel patches and build the filtered amplitude spectrum.
    2. Fit defocus + astigmatism with CTFFind's scoring (``fit_ctf_2d_native``).
    3. Bin the filtered spectrum into a 1D profile with CTFFind's exact extrema+value
       binning (``ctffind_rotational_average``).
    4. Renormalise per cycle, compute the windowed FRC, and walk the dual threshold
       (``fit_resolution``), exactly as CTFFind.

    Parameters
    ----------
    image
        (t, h, w) or (h, w) real-space image at native pixel spacing.
    pixel_spacing_angstroms, voltage_kev, spherical_aberration_mm, amplitude_contrast
        Optics at native resolution.
    frequency_fit_range_angstroms
        (low, high) resolution bounds in Angstroms.
    initial_defocus_um
        Mean-defocus seed for the fit (e.g. the existing 1D estimate).
    box
        Amplitude-spectrum box size (default 512, CTFFind default).
    phase_shift_deg
        Fixed additional phase shift.
    target_pixel_size_after_resampling
        CTFFind's resampling target (default 1.4 A). When the native pixel is smaller,
        the spectrum is resampled to this pixel size for fitting (matching CTFFind), so
        the frequency grid and FRC are computed on the same bins CTFFind uses.

    Returns
    -------
    (fit, resolution)
        ``fit`` is a :class:`NativeCTFFit`; ``resolution`` is the
        :class:`~torch_ctf_estimation.metrics.fit_resolution.CTFFitResolution`.
    """
    from torch_ctf_estimation.metrics.fit_resolution import (
        CTFFitResolution,
        compute_frc_1d,
        max_resolution_from_frc,
        renormalize_1d_spectrum_for_frc,
    )

    low_ang, high_ang = frequency_fit_range_angstroms

    # CTFFind ``PixelSizeForFitting`` (ctffind.cpp:73): if the native pixel is smaller
    # than the target, compute the amplitude spectrum in a larger box and crop the
    # central box*box, giving an effective ``pixel_size_for_fitting`` == target. This
    # standardises the frequency grid (and hence the FRC bins) the way CTFFind does.
    amplitude, pixel_fit = _resampled_amplitude_spectrum(
        image, box, pixel_spacing_angstroms, target_pixel_size_after_resampling
    )
    filtered = filter_amplitude_spectrum(amplitude, pixel_fit, low_ang)
    fit = fit_ctf_2d_native(
        filtered,
        pixel_fit,
        voltage_kev=voltage_kev,
        spherical_aberration_mm=spherical_aberration_mm,
        amplitude_contrast=amplitude_contrast,
        frequency_fit_range_angstroms=frequency_fit_range_angstroms,
        initial_defocus_um=initial_defocus_um,
        phase_shift_deg=phase_shift_deg,
    )

    # CTFFind's exact astig-aware discrete average of the filtered spectrum, then the
    # renorm + windowed FRC + dual-threshold walk (all on the same arrays as CTFFind).
    average, average_fit, ne_profile, sf = ctffind_rotational_average(
        filtered,
        pixel_fit,
        fit,
        voltage_kev=voltage_kev,
        spherical_aberration_mm=spherical_aberration_mm,
        amplitude_contrast=amplitude_contrast,
    )
    # first_fit_bin = lowest bin inside the fitting band (ctffind.cpp:1726)
    nonzero = torch.nonzero(sf >= 1.0 / low_ang, as_tuple=False)
    first_fit_bin = int(nonzero[0].item()) if nonzero.numel() > 0 else 0

    average_renorm = renormalize_1d_spectrum_for_frc(average, average_fit, ne_profile)
    fit_frc = compute_frc_1d(average_renorm, average_fit, ne_profile, first_fit_bin)
    resolution_angstroms, last_bin = max_resolution_from_frc(fit_frc, sf)

    resolution = CTFFitResolution(
        resolution_angstroms=resolution_angstroms,
        last_bin_with_good_fit=last_bin,
        fit_frc=fit_frc,
        frc_frequencies=sf,
    )
    return fit, resolution


def _resampled_amplitude_spectrum(
    image: torch.Tensor,
    box: int,
    pixel_spacing_angstroms: float,
    target_pixel_size_after_resampling: float,
) -> tuple[torch.Tensor, float]:
    """
    Amplitude spectrum + effective pixel; ``PixelSizeForFitting`` (ctffind.cpp:73).

    If the native pixel is smaller than the target, compute the mean amplitude spectrum
    in a larger box ``round(box * target / pixel)`` and crop the central ``box*box`` —
    this lowers the effective Nyquist to the target pixel and gives CTFFind's frequency
    grid. Otherwise compute it directly at ``box``. Returns ``(amplitude, pixel_fit)``.
    """
    max_box = min(image.shape[-2], image.shape[-1])
    if pixel_spacing_angstroms < target_pixel_size_after_resampling:
        temp_box = round(
            box / pixel_spacing_angstroms * target_pixel_size_after_resampling
        )
        if temp_box % 2 == 1:
            temp_box += 1
        if temp_box > max_box:
            # Image too small to resample to target; fit at native pixel instead.
            patches = patches_from_image(image, box)
            return compute_amplitude_spectrum_2d(patches), pixel_spacing_angstroms
        pixel_fit = pixel_spacing_angstroms * temp_box / box
        patches = patches_from_image(image, temp_box)
        amp_big = compute_amplitude_spectrum_2d(patches)  # (temp_box, temp_box)
        c = temp_box // 2
        lo = c - box // 2
        amplitude = amp_big[lo : lo + box, lo : lo + box].contiguous()
        return amplitude, pixel_fit
    patches = patches_from_image(image, box)
    return compute_amplitude_spectrum_2d(patches), pixel_spacing_angstroms


def patches_from_image(
    image: torch.Tensor,
    box: int,
) -> torch.Tensor:
    """
    Extract overlapping square patches (50% step) from a (t, h, w) or (h, w) image.

    Returns (n_patches, box, box). Used to build the native amplitude spectrum.
    """
    from torch_grid_utils.patch_grid import patch_grid

    img, _ = einops.pack([image], "* h w")
    patches, _ = patch_grid(
        images=img,
        patch_shape=(1, box, box),
        patch_step=(1, box // 2, box // 2),
    )
    return einops.rearrange(patches, "t gh gw 1 ph pw -> (t gh gw) ph pw")
