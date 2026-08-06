"""Build whitened, labelled 2-channel examples: [strain, witness].

For each class the strain channel uses real interferometer background; the witness
channel is synthesised. Signals are injected into strain only; glitches are coupled
into both strain and witness (see ``witness.py``); background is noise only. A fourth
``signal_glitch`` class injects a coincident signal and blip together (signal in strain
only, blip in both channels), modelling a real signal contaminated by a glitch.

Adapted from chreissel/GWDatasetGeneration, extended with a witness channel, a glitch
class, and a combined signal+glitch class.
"""

import importlib
from pathlib import Path

import torch
import torch.nn.functional as F

from ml4gw.transforms import SpectralDensity, Whiten
from ml4gw.dataloading import Hdf5TimeSeriesDataset
from ml4gw.gw import compute_network_snr, reweight_snrs

from utils import load_config
from waveforms import generate_signals, generate_glitch_sources
from witness import make_witness_noise, derive_witness, bandlimit

LABELS = {"background": 0, "signal": 1, "glitch": 2, "signal_glitch": 3}


def _sample_target_snr(snr_cfg, batch_size, device):
    func_path = snr_cfg.func
    module_name, func_name = func_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    args = getattr(snr_cfg, "args", [])
    return func(*args).sample((batch_size,)).to(device)


def _interp_psd(psd, num_freqs):
    if psd.shape[-1] != num_freqs:
        while psd.ndim < 3:
            psd = psd[None]
        psd = F.interpolate(psd, size=(num_freqs,), mode="linear")
    return psd


def _inject_center(kernel, response, kernel_size, pad):
    """Add ``response`` (batch, 1, time) into the central region of ``kernel``."""
    injected = kernel.detach().clone()
    injected[:, :, pad:-pad] += response[..., -kernel_size:]
    return injected


def _time_jitter(response, config, sample_rate, device):
    """Randomly shift each batch element in time (zero-padded).

    Returns ``(shifted, shifts_seconds)``. The shift is drawn per element from a
    uniform distribution of half-width ``config.general.time_jitter`` seconds. The
    transient nominally peaks ``right_pad`` s from the right edge of the injected
    region, so there is only ``right_pad`` s of room on the late side; the range is
    clipped to the room available on each side (with a small guard) so a transient is
    never pushed out of the analysis window. All channels of ``response`` are shifted
    by the same amount, so a glitch's strain and witness copies stay aligned.
    """
    b, c, t = response.shape
    time_jitter = float(getattr(config.general, "time_jitter", 0.0))
    jit = int(round(time_jitter * sample_rate))
    if jit <= 0:
        return response, torch.zeros(b, device=device)

    peak = int(round(config.general.right_pad * sample_rate))  # samples from right edge
    guard = int(round(0.05 * sample_rate))                     # keep the body in-window
    hi = min(jit, max(0, peak - guard))                        # room on the late side
    lo = -min(jit, max(0, t - peak - guard))                   # room on the early side
    if hi <= lo:
        return response, torch.zeros(b, device=device)

    shifts = torch.randint(lo, hi + 1, (b,), device=device)
    src = torch.arange(t, device=device).view(1, t) - shifts.view(b, 1)
    valid = (src >= 0) & (src < t)
    idx = src.clamp(0, t - 1).unsqueeze(1).expand(b, c, t)
    shifted = torch.gather(response, -1, idx) * valid.unsqueeze(1).to(response.dtype)
    return shifted, shifts.to(torch.float32) / sample_rate


def _inject_signal(config, device, strain_kernel, strain_psd_i,
                   sample_rate, f_min, kernel_size, pad):
    """Inject a CBC signal into the strain kernel; the witness is left untouched."""
    batch_size = config.general.batch_size
    waveforms, params = generate_signals(config, device)
    waveforms, params["time_shift"] = _time_jitter(waveforms, config, sample_rate, device)
    target_snrs = _sample_target_snr(config.snr_reweighting, batch_size, device)
    waveforms = reweight_snrs(
        responses=waveforms, target_snrs=target_snrs, psd=strain_psd_i,
        sample_rate=sample_rate, highpass=f_min,
    )
    # Store the achieved SNR; reweight_snrs can miss the target by a few percent.
    params["snr"] = compute_network_snr(waveforms, strain_psd_i, sample_rate, highpass=f_min)
    strain_kernel = _inject_center(strain_kernel, waveforms, kernel_size, pad)
    return strain_kernel, params


def _inject_glitch(config, device, strain_kernel, witness_kernel, strain_psd_i,
                   witness_psd_i, sample_rate, f_min, f_max, kernel_size, pad):
    """Inject a blip glitch into both the strain and the (derived) witness kernels.

    The SineGaussian blip is injected into the strain channel; the witness glitch is
    derived from it (see witness.derive_witness). The blip is confined to the detector
    band [f_min, f_max] so it stays physical and carries no power in PSD bins where the
    SNR reweighting would misbehave.
    """
    batch_size = config.general.batch_size
    blip, params = generate_glitch_sources(config, device)
    blip_indep, _ = generate_glitch_sources(config, device)
    blip = bandlimit(blip, sample_rate, f_min, f_max)
    strain_g, witness_g = derive_witness(
        blip, blip_indep, sample_rate, config.witness.coupling
    )
    strain_g = strain_g.unsqueeze(1)
    witness_g = witness_g.unsqueeze(1)

    # Jitter the strain and witness copies together (same physical disturbance, so
    # the same time shift) before reweighting.
    glitch = torch.cat([strain_g, witness_g], dim=1)
    glitch, params["time_shift"] = _time_jitter(glitch, config, sample_rate, device)
    strain_g, witness_g = glitch[:, :1], glitch[:, 1:]

    # derive_witness returns unit-RMS glitches. Real strain is ~1e-21, so a unit
    # amplitude glitch makes the SNR integrand |h|^2 / PSD overflow float32 inside
    # ml4gw's compute_ifo_snr (SNR -> inf, then reweight scales the glitch to ~0 and
    # it vanishes from the strain). Rescale each glitch to its background's RMS first
    # so the SNR computation stays well conditioned; reweight_snrs then sets the
    # final amplitude to hit the target SNR regardless of this prefactor.
    strain_g = strain_g * strain_kernel.std(dim=-1, keepdim=True)
    witness_g = witness_g * witness_kernel.std(dim=-1, keepdim=True)

    target_strain_snr = _sample_target_snr(config.glitch.snr.strain, batch_size, device)
    target_witness_snr = _sample_target_snr(config.glitch.snr.witness, batch_size, device)

    strain_g = reweight_snrs(
        responses=strain_g, target_snrs=target_strain_snr, psd=strain_psd_i,
        sample_rate=sample_rate, highpass=f_min,
    )
    witness_g = reweight_snrs(
        responses=witness_g, target_snrs=target_witness_snr, psd=witness_psd_i,
        sample_rate=sample_rate, highpass=f_min,
    )

    strain_kernel = _inject_center(strain_kernel, strain_g, kernel_size, pad)
    witness_kernel = _inject_center(witness_kernel, witness_g, kernel_size, pad)
    # Store the achieved SNRs (a few percent off the targets is acceptable).
    params["strain_snr"] = compute_network_snr(
        strain_g, strain_psd_i, sample_rate, highpass=f_min
    )
    params["witness_snr"] = compute_network_snr(
        witness_g, witness_psd_i, sample_rate, highpass=f_min
    )
    return strain_kernel, witness_kernel, params


def injection(config, data_dir: str, device: str, mode: str):
    """Generate one batch of whitened 2-channel data for the given class.

    Returns ``(data, params)`` where ``data`` has shape (batch, 2, T) with channel
    order [strain, witness], and ``params`` is a dict of tensors (or ``None``).
    """
    assert mode in LABELS, f"unknown mode {mode!r}"

    ifos = config.general.ifos
    assert len(ifos) == 1, "This toy dataset focuses on a single strain detector."

    batch_size = config.general.batch_size
    sample_rate = config.general.sample_rate
    f_min = config.general.f_min
    f_max = config.general.f_max
    kernel_length = config.general.waveform_duration

    fduration = config.whiten.fduration
    fftlength = config.whiten.fftlength
    psd_length = config.whiten.psd_length
    overlap = config.whiten.overlap
    average = config.whiten.average

    psd_size = int(psd_length * sample_rate)
    kernel_size = int(kernel_length * sample_rate)
    pad = int(fduration / 2 * sample_rate)

    window_length = psd_length + fduration + kernel_length
    window_size = int(window_length * sample_rate)
    num_samples = int(kernel_length * sample_rate)
    num_freqs = num_samples // 2 + 1

    # ---- backgrounds -------------------------------------------------------
    fnames = list(Path(data_dir).iterdir())
    dataloader = Hdf5TimeSeriesDataset(
        fnames=fnames,
        channels=ifos,
        kernel_size=window_size,
        batch_size=batch_size,
        batches_per_epoch=1,
        coincident=False,
    )
    strain_bg = [x for x in dataloader][0].to(device)  # (batch, 1, window_size)
    witness_bg = make_witness_noise(
        batch_size, window_size, sample_rate, device, getattr(config.witness, "noise", None)
    )

    spectral_density = SpectralDensity(
        sample_rate=sample_rate, fftlength=fftlength, overlap=overlap, average=average
    ).to(device)
    whiten = Whiten(fduration=fduration, sample_rate=sample_rate, highpass=f_min).to(device)

    strain_psd = spectral_density(strain_bg[..., :psd_size].double())
    witness_psd = spectral_density(witness_bg[..., :psd_size].double())

    strain_kernel = strain_bg[..., psd_size:]
    witness_kernel = witness_bg[..., psd_size:]

    strain_psd_i = _interp_psd(strain_psd, num_freqs)
    witness_psd_i = _interp_psd(witness_psd, num_freqs)

    params = {}

    # ---- class-specific injection -----------------------------------------
    if mode == "signal":
        strain_kernel, params = _inject_signal(
            config, device, strain_kernel, strain_psd_i,
            sample_rate, f_min, kernel_size, pad,
        )

    elif mode == "glitch":
        strain_kernel, witness_kernel, params = _inject_glitch(
            config, device, strain_kernel, witness_kernel, strain_psd_i,
            witness_psd_i, sample_rate, f_min, f_max, kernel_size, pad,
        )

    elif mode == "signal_glitch":
        # Near-coincident signal + blip: the signal lives in the strain only, while
        # the blip appears in both strain and witness. The signal and glitch are
        # jittered independently, so they are not perfectly aligned in time. The
        # witness therefore flags the glitch contamination without responding to the
        # astrophysical signal.
        strain_kernel, sig_params = _inject_signal(
            config, device, strain_kernel, strain_psd_i,
            sample_rate, f_min, kernel_size, pad,
        )
        strain_kernel, witness_kernel, glitch_params = _inject_glitch(
            config, device, strain_kernel, witness_kernel, strain_psd_i,
            witness_psd_i, sample_rate, f_min, f_max, kernel_size, pad,
        )
        # Distinct keys so the two independent time shifts both survive the merge.
        sig_params["signal_time_shift"] = sig_params.pop("time_shift")
        glitch_params["glitch_time_shift"] = glitch_params.pop("time_shift")
        params = {**sig_params, **glitch_params}

    # mode == "background": inject nothing.

    # ---- whiten & stack ----------------------------------------------------
    strain_w = whiten(strain_kernel, strain_psd)
    witness_w = whiten(witness_kernel, witness_psd)
    data = torch.cat([strain_w, witness_w], dim=1)  # (batch, 2, T)

    params["label"] = torch.full((batch_size,), LABELS[mode], dtype=torch.long)
    return data, params


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config(config_path="configs/config_H1.yaml")
    background_dir = Path("./data") / "background_data"
    for m in ("background", "signal", "glitch", "signal_glitch"):
        d, p = injection(config, data_dir=background_dir, device=device, mode=m)
        print(m, d.shape)
