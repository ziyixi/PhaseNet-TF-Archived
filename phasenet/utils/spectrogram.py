import torch
import torch.nn.functional as F
from phasenet.conf.load_conf import SpectrogramConfig

window_fn_collections = {
    "hann": torch.hann_window
}


def spectrogram(x: torch.Tensor, cfg_spec: SpectrogramConfig) -> torch.Tensor:
    """spectrogram calculation for the input waveform data

    Args:
        x (torch.Tensor): input data with last dimension as nt
        cfg_spec (SpectrogramConfig): spectrogram conversion configuration

    Raises:
        Exception: window function is not supported!

    Returns:
        torch.Tensor: generated spectrogram, size is  ...,nf,n_frame,2. If magnitude is True, the last dimension will be 1 (only magnite) or 2 (with phase)
    """
    if cfg_spec.window_fn in window_fn_collections:
        window_fn = window_fn_collections[cfg_spec.window_fn]
    else:
        raise Exception(
            f"window function {cfg_spec.window_fn} is not supported!")

    with torch.set_grad_enabled(cfg_spec.grad):
        window = window_fn(cfg_spec.n_fft).to(x.device)
        # sfft input: batch*channel,nt; output: batch*channel,nf,n_frame,2 (2: real, complex)
        stft = torch.stft(x, n_fft=cfg_spec.n_fft, window=window,
                          hop_length=cfg_spec.hop_length, center=True, return_complex=False)
        # as origional n_frame=x.shape[-1] // cfg_spec.hop_length+1
        stft = stft[..., : x.shape[-1] // cfg_spec.hop_length, :]
        # discard the zero frequency
        if cfg_spec.discard_zero_freq:
            stft = stft.narrow(dim=-3, start=1, length=stft.shape[-3] - 1)
        # custome the frequency range selection
        # ! note, here the original implementation seems to have possible problem
        # ! the frequency size is floor of n_fft//2, which is not n_fft
        if cfg_spec.select_freq:
            fmax = 1 / 2 / cfg_spec.dt
            freq = torch.linspace(0, fmax, cfg_spec.n_fft)
            idx = torch.arange(
                cfg_spec.n_fft)[(freq > cfg_spec.fmin) & (freq < cfg_spec.fmax)]
            stft = stft.narrow(dim=-3, start=idx[0].item(), length=idx.numel())
        if cfg_spec.magnitude:
            stft_mag = torch.norm(stft, dim=-1)
            if cfg_spec.log_transform:
                stft_mag = torch.log(1 + F.relu(stft_mag))
            if cfg_spec.phase:
                components = stft.split(1, dim=-1)
                stft_phase = torch.atan2(
                    components[1].squeeze(-1), components[0].squeeze(-1))
                stft = torch.stack([stft_mag, stft_phase], dim=-1)
            else:
                stft = stft_mag
        else:
            if cfg_spec.log_transform:
                stft = torch.log(1 + F.relu(stft)) - \
                    torch.log(1 + F.relu(-stft))
        return stft
