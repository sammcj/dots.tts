"""Audio helpers used by the retained train/infer pipeline."""

from __future__ import annotations

import torch
import torchaudio.compliance.kaldi as Kaldi
import torchaudio.functional as AF


def high_quality_resample(x, orig_sr, target_sr):
    return AF.resample(
        x,
        orig_freq=orig_sr,
        new_freq=target_sr,
        lowpass_filter_width=64,
        rolloff=0.95,
        resampling_method="sinc_interp_kaiser",
    )


def extract_fbank(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    n_mels: int,
    dither: float = 0.0,
    mean_norm: bool = False,
) -> torch.Tensor:
    if waveform.ndim == 1:
        feature_input = waveform.unsqueeze(0)
    elif waveform.ndim == 2:
        feature_input = waveform if waveform.size(0) == 1 else waveform[0:1, :]
    else:
        raise ValueError(
            f"FBank expects a 1D or 2D waveform, got shape {tuple(waveform.shape)}."
        )
    # Kaldi.fbank runs an rfft, and Metal has no fp16/bf16 FFT kernel, so this
    # must be fp32 even when the caller is inside a reduced-precision autocast
    # region (the MPS path). Forcing the input to fp32 and disabling autocast
    # here keeps the fbank (and the speaker x-vector net it feeds) in fp32 while
    # the matmul-heavy backbone still benefits from autocast elsewhere.
    feature_input = feature_input.float()
    with torch.autocast(device_type=feature_input.device.type, enabled=False):
        features = Kaldi.fbank(
            feature_input,
            num_mel_bins=n_mels,
            sample_frequency=sample_rate,
            dither=dither,
        )
    if mean_norm:
        features = features - features.mean(dim=0, keepdim=True)
    return features
