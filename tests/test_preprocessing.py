import torch
import torchaudio
import pytest
# Adjust the import to match your project structure.
from preprocessing import preprocess_audio

def test_preprocess_audio(tmp_path):
    # Create a dummy audio tensor: 1 second of random audio at 16kHz.
    sample_rate = 16000
    waveform = torch.randn(1, sample_rate)  # shape: (channels, samples)
    
    # Save this waveform to a temporary WAV file.
    temp_file = tmp_path / "temp.wav"
    torchaudio.save(str(temp_file), waveform, sample_rate)
    
    # Preprocess the audio file.
    processed_waveform = preprocess_audio(str(temp_file), target_sample_rate=16000)
    
    # Check that the output is a tensor.
    assert isinstance(processed_waveform, torch.Tensor)
    
    # Check that the waveform has been normalized.
    mean = processed_waveform.mean().item()
    std = processed_waveform.std().item()
    # Allowing some tolerance for random data.
    assert abs(mean) < 0.1
    assert abs(std - 1) < 0.1
