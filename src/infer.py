
import torch
from diarization import TransformerDiarizationModel
from settings import config

def test_inference():
    model = TransformerDiarizationModel(config.num_speakers, config.num_events, config.num_moods)
    model.load_state_dict(torch.load("diarization_model.pth")) # type: ignore
    model.eval()
    # Generate synthetic input features.
    inputs = torch.randn(1, config.seq_length, config.input_features)
    speaker_logits, event_logits, mood_logits = model(inputs)
    print("Inference completed!") 
    print(speaker_logits.shape, event_logits.shape, mood_logits.shape)  
    print(speaker_logits[0, :10, :])  # Print first 10 time steps of speaker logits.