"""Witness-channel synthesis and glitch coupling.

GWOSC open data only contains the strain channel, so the witness channel is
synthesised here. The physical story this models:

* A blip glitch is injected directly into the *strain* channel as a short,
  broadband SineGaussian transient (see ``waveforms.generate_glitch_sources``),
  mimicking the teardrop-shaped blips seen in real interferometer data.
* The *same* terrestrial disturbance is observed by a witness (auxiliary) sensor.
  We therefore *derive* the witness glitch from the injected strain glitch: it is
  the strain blip passed through the witness's own coupling path, modelled as a
  linear time-invariant (LTI) Butterworth filter ``C`` (a limited sensor band),
  optionally with a small propagation ``lag``.
* The coupling is partial: only a fraction ``alpha`` of the witness-glitch power is
  coherent with the strain blip; the remaining ``1 - alpha`` is independent sensor
  noise. ``alpha`` therefore sets the strain<->witness coherence and is the single
  "how useful is the witness" knob.

Crucially, *astrophysical* signals do not couple to a witness, so for the signal
and background classes the witness carries noise only. Only the glitch class
injects a correlated transient into the witness.
"""

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt


def _rms_normalize(x: torch.Tensor) -> torch.Tensor:
    """Normalise each row of (batch, time) to unit RMS."""
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True)) + 1e-30
    return x / rms


def make_witness_noise(batch_size, length, sample_rate, device, noise_cfg=None):
    """Synthesise a raw witness background of shape (batch, 1, length).

    ``noise_cfg.color`` selects the spectral shape:
      * ``white`` (default): flat ASD.
      * ``powerlaw``: ASD proportional to ``f ** index`` (with a low-frequency
        floor), giving e.g. red noise for a negative index.
    """
    color = getattr(noise_cfg, "color", "white") if noise_cfg is not None else "white"
    white = torch.randn(batch_size, 1, length, device=device)

    if color == "white":
        return white

    index = float(getattr(noise_cfg, "index", -1.0))
    num_freqs = length // 2 + 1
    freqs = torch.fft.rfftfreq(length, d=1.0 / sample_rate).to(device)
    asd = torch.ones(num_freqs, device=device)
    nonzero = freqs > 0
    asd[nonzero] = freqs[nonzero] ** index
    asd[~nonzero] = asd[nonzero][0] if nonzero.any() else 1.0

    spectrum = torch.fft.rfft(white, dim=-1) * asd
    colored = torch.fft.irfft(spectrum, n=length, dim=-1)
    return _rms_normalize(colored.squeeze(1)).unsqueeze(1)


def _butter_filter(x: torch.Tensor, sample_rate, filt_cfg) -> torch.Tensor:
    """Apply a zero-phase Butterworth coupling filter along the time axis."""
    btype = getattr(filt_cfg, "btype", "bandpass")
    order = int(getattr(filt_cfg, "order", 4))
    cutoff = getattr(filt_cfg, "cutoff", [20.0, 400.0])
    if isinstance(cutoff, (list, tuple)):
        wn = [c / (sample_rate / 2) for c in cutoff]
    else:
        wn = cutoff / (sample_rate / 2)

    sos = butter(order, wn, btype=btype, output="sos")
    arr = x.detach().cpu().numpy().astype(np.float64)
    filtered = sosfiltfilt(sos, arr, axis=-1).copy()
    return torch.from_numpy(filtered).to(x.device, x.dtype)


def bandlimit(x: torch.Tensor, sample_rate, f_min, f_max) -> torch.Tensor:
    """Confine a transient to the detector's sensitive band ``[f_min, f_max]``.

    A low-Q blip is broadband, but a real blip only has measurable power within the
    detector band. Confining it here keeps the strain glitch physical *and* avoids
    putting power in frequency bins where the estimated PSD is ~0 / unreliable (e.g.
    near DC or Nyquist), which would otherwise make the SNR reweighting blow up.
    """
    nyq = sample_rate / 2.0
    lo = max(float(f_min), 0.0) / nyq
    hi = min(float(f_max), 0.999 * nyq) / nyq
    if lo > 0.0 and hi < 1.0:
        sos = butter(8, [lo, hi], btype="bandpass", output="sos")
    elif lo > 0.0:
        sos = butter(8, lo, btype="highpass", output="sos")
    elif hi < 1.0:
        sos = butter(8, hi, btype="lowpass", output="sos")
    else:
        return x
    arr = x.detach().cpu().numpy().astype(np.float64)
    filtered = sosfiltfilt(sos, arr, axis=-1).copy()
    return torch.from_numpy(filtered).to(x.device, x.dtype)


def derive_witness(strain_blip, witness_indep, sample_rate, coupling_cfg):
    """Derive the witness glitch from the blip injected into the strain channel.

    The blip ``strain_blip`` is what the strain channel actually records. The
    witness sensor observes the same disturbance through its own coupling path: a
    band-limited transfer function (Butterworth filter ``C``), an optional small
    propagation ``lag``, plus independent sensor noise. The coherent fraction is
    set by ``alpha``.

    Parameters
    ----------
    strain_blip : (batch, time) tensor
        The SineGaussian blip as injected into the strain channel.
    witness_indep : (batch, time) tensor
        An independent glitch realisation supplying the incoherent sensor
        component (so the witness still looks glitch-shaped where it is not
        coherent with the strain).
    coupling_cfg : namespace
        ``type`` (only ``lti`` supported), ``filter`` (btype/cutoff/order),
        ``alpha`` (coherent fraction in [0, 1]) and optional ``lag`` (coupling
        delay in seconds).

    Returns
    -------
    strain_glitch, witness_glitch : (batch, time) tensors
        Unit-RMS-normalised; absolute amplitude is set later via SNR reweighting.
    """
    coupling_type = getattr(coupling_cfg, "type", "lti")
    if coupling_type != "lti":
        raise ValueError(f"Unsupported coupling type: {coupling_type!r} (only 'lti').")

    alpha = float(getattr(coupling_cfg, "alpha", 0.8))
    alpha = min(max(alpha, 0.0), 1.0)

    # The strain keeps the clean blip; the witness is a band-limited copy of it.
    strain_glitch = _rms_normalize(strain_blip)
    coupled = _rms_normalize(_butter_filter(strain_glitch, sample_rate, coupling_cfg.filter))

    # Optional propagation delay between the strain and the witness sensor.
    lag = float(getattr(coupling_cfg, "lag", 0.0))
    if lag:
        coupled = torch.roll(coupled, shifts=int(round(lag * sample_rate)), dims=-1)

    indep = _rms_normalize(_butter_filter(witness_indep, sample_rate, coupling_cfg.filter))

    # Power-weighted mix so that the fraction of witness-glitch power coherent with
    # the strain blip is ~alpha.
    witness_glitch = (alpha**0.5) * coupled + ((1.0 - alpha) ** 0.5) * indep
    return strain_glitch, _rms_normalize(witness_glitch)
