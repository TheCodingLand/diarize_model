import torch
import pytest
# Adjust the import to match your project structure.
from diarization import TransformerDiarizationModel
from settings import config
from diarization import ArcFaceLoss  # Adjust the import to your project structure
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_model_forward():
    # Define parameters matching the model/dataset configuration.
    num_speakers = 6
    num_events = 4
    num_moods = 5
    batch_size = 2

    # Set device to CUDA if available, otherwise CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a dummy input with shape (batch, seq_length, input_features) and move it to the device.
    dummy_input = torch.randn(batch_size, config.seq_length, config.input_features).to(device)

    # Instantiate the model and move it to the device.
    model = TransformerDiarizationModel(num_speakers, num_events, num_moods).to(device)
    model.eval()  # Set to evaluation mode

    with torch.no_grad():
        speaker_logits, event_logits, mood_logits = model(dummy_input)

    # Verify that the output shapes match (batch, seq_length, num_classes)
    assert speaker_logits.shape == (batch_size, config.seq_length, num_speakers)
    assert event_logits.shape == (batch_size, config.seq_length, num_events)
    assert mood_logits.shape == (batch_size, config.seq_length, num_moods)
    
    print("Test passed: model forward output shapes are correct.")


def test_arcface_loss_output_and_gradient():
    batch_size = 8
    num_classes = 6

    # Create dummy logits with requires_grad=True so we can check gradients.
    logits = torch.randn(batch_size, num_classes, requires_grad=True)
    # Create dummy labels as integers in the range [0, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))

    # Initialize the ArcFaceLoss.
    loss_fn = ArcFaceLoss(scale=30.0, margin=0.50)
    loss = loss_fn(logits, labels)

    # Check that the loss is a scalar.
    assert loss.dim() == 0, "Expected loss to be a scalar."

    # Perform a backward pass to check gradient computation.
    loss.backward()
    assert logits.grad is not None, "Expected gradients for logits to be computed."

    # Optionally, check that the gradient has the same shape as logits.
    assert logits.grad.shape == logits.shape, "Gradient shape should match logits shape."