import torch
from diarization import TransformerDiarizationModel
from settings import config

# Set the device to CUDA if available, otherwise CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_inference():
    # Initialize the model with the configuration parameters.
    model = TransformerDiarizationModel(config.num_speakers, config.num_events, config.num_moods)
    
    # Load the checkpoint with the model parameters.
    model.load_state_dict(torch.load("diarization_model.pth", map_location=device))
    
    # Move the model to the appropriate device.
    model.to(device)
    model.eval()
    
    # Generate synthetic input features and move them to the device.
    inputs = torch.randn(1, config.seq_length, config.input_features).to(device)
    
    # Perform inference without tracking gradients.
    with torch.no_grad():
        speaker_logits, event_logits, mood_logits = model(inputs)
    
    print("Inference completed!")
    print("Speaker logits shape:", speaker_logits.shape)
    print("Event logits shape:", event_logits.shape)
    print("Mood logits shape:", mood_logits.shape)
    print("First 10 time steps of speaker logits:")
    print(speaker_logits[0, :10, :])
    
if __name__ == "__main__":
    test_inference()