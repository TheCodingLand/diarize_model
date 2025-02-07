import torch
import pytest
# Adjust the import to match your project structure.
from diarization import TransformerDiarizationModel
from settings import config

def test_model_forward():
    # Define parameters matching the model/dataset configuration.
    num_speakers = 6
    num_events = 4
    num_moods = 5
    batch_size = 2

    # Create a dummy input with shape (batch, seq_length, input_features)
    dummy_input = torch.randn(batch_size, config.seq_length, config.input_features)

    # Instantiate the model.
    model = TransformerDiarizationModel(num_speakers, num_events, num_moods)
    model.eval()  # Set to evaluation mode

    with torch.no_grad():
        speaker_logits, event_logits, mood_logits = model(dummy_input)

    # Verify that the output shapes match (batch, seq_length, num_classes)
    assert speaker_logits.shape == (batch_size, config.seq_length, num_speakers)
    assert event_logits.shape == (batch_size, config.seq_length, num_events)
    assert mood_logits.shape == (batch_size, config.seq_length, num_moods)
