# Configuration
import torch
from diarization import ArcFaceLoss, SyntheticDiarizationDataset, TransformerDiarizationModel
from torch.utils.data import DataLoader
from settings import config
import torch.nn as nn
import torch.nn.functional as F
from utils import mixup_data

# Set device to CUDA if available, otherwise CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
def train():
    # Initialize dataset and model. (Dataset parameters override num_speakers, etc.)
    dataset = SyntheticDiarizationDataset(num_samples=1000, num_speakers=6, num_events=4, num_moods=5)
    model = TransformerDiarizationModel(dataset.num_speakers, dataset.num_events, dataset.num_moods)
    model.to(device)
    
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Move ArcFace loss (which has registered buffers) to device.
    criterion_speaker = ArcFaceLoss().to(device)
    criterion_event = nn.CrossEntropyLoss()
    criterion_mood = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Training Loop
    for epoch in range(config.num_epochs):
        for batch_idx, (inputs, (speaker_labels, event_labels, mood_labels)) in enumerate(dataloader):
            # Move inputs and labels to the device.
            inputs = inputs.to(device)
            speaker_labels = speaker_labels.to(device)
            event_labels = event_labels.to(device)
            mood_labels = mood_labels.to(device)
            

            # Apply mixup only to the inputs.
            inputs, _ = mixup_data(inputs, speaker_labels)
            
            # Forward pass
            speaker_logits, event_logits, mood_logits = model(inputs)

            # Compute losses.
            speaker_loss = criterion_speaker(
                F.log_softmax(speaker_logits.reshape(-1, dataset.num_speakers), dim=-1),
                speaker_labels.view(-1)
            )
            event_loss = criterion_event(
                event_logits.contiguous().view(-1, dataset.num_events),
                event_labels.view(-1)
            )
            mood_loss = criterion_mood(
                mood_logits.contiguous().view(-1, dataset.num_moods),
                mood_labels.view(-1)
            )

            total_loss = speaker_loss + event_loss + mood_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{config.num_epochs}, Batch {batch_idx}, Loss: {total_loss.item():.4f}")

    config.num_moods = dataset.num_moods
    config.num_events = dataset.num_events
    config.num_speakers = dataset.num_speakers
    # Save the model to disk.
    config.save()

    print("Training completed!")
    torch.save(model.state_dict(), "diarization_model.pth")

if __name__ == "__main__":
    train()
    from infer import test_inference
    test_inference()
