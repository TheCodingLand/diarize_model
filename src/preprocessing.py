
import torchaudio
import torch

def preprocess_audio(file_path: str, target_sample_rate: int = 16000) -> torch.Tensor:
    """
    Load and preprocess an audio file.

    Parameters:
        file_path (str): Path to the audio file.
        target_sample_rate (int): The desired sample rate.

    Returns:
        torch.Tensor: Normalized audio waveform.
    """
    waveform: torch.Tensor
    sample_rate: int
    waveform, sample_rate = torchaudio.load(file_path)  # type: ignore
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    # Normalize the waveform: zero mean and unit variance
    waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-5)
    return waveform