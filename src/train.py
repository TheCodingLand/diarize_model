

# Configuration
import torch
from diarization import ArcFaceLoss, SyntheticDiarizationDataset, TransformerDiarizationModel

from torch.utils.data import DataLoader
from settings import config
import torch.nn as nn
import torch.nn.functional as F
from utils import mixup_data
# ArcFace Loss (expects hard integer labels)

    
def train():
    # Initialize dataset and model. (Dataset parameters override num_speakers, etc.)
    dataset = SyntheticDiarizationDataset(num_samples=1000, num_speakers=6, num_events=4, num_moods=5)
    model = TransformerDiarizationModel(dataset.num_speakers, dataset.num_events, dataset.num_moods)
    dataloader: DataLoader[SyntheticDiarizationDataset] = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    criterion_speaker = ArcFaceLoss()
    criterion_event = nn.CrossEntropyLoss()
    criterion_mood = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Training Loop
    for epoch in range(config.num_epochs):
        for batch_idx, (inputs, (speaker_labels, event_labels, mood_labels)) in enumerate(dataloader):
            # Apply mixup only to the inputs.
            inputs, _ = mixup_data(inputs, speaker_labels)
            speaker_logits, event_logits, mood_logits = model(inputs)

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

    print("Training completed!")
    torch.save(model.state_dict(), "diarization_model.pth")




if __name__ == "__main__":
    train()
    from infer import test_inference
    test_inference()