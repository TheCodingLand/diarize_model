import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import Wav2Vec2Model

from typing import Tuple

try: 
    from clearml import Task
    try:
        task = Task.init(project_name="my_project", task_name="experiment_1")
    except Exception as e:
        task = Task.current_task()

except ImportError:
    pass

# Configuration
num_speakers: int = 5  # Overridden by the dataset below.
num_events: int = 3
num_moods: int = 4
input_features: int = 40  # Synthetic input dimension.
seq_length: int = 1000    # Increased from 100 to ensure waveform length is long enough.
hidden_size: int = 128
batch_size: int = 32
num_epochs: int = 2
learning_rate: float = 0.001

# Mixup Augmentation (applied only to inputs; speaker labels are kept intact for ArcFace)
def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor]:
    lam: float = np.random.beta(alpha, alpha)
    batch_size: int = x.size(0)
    index: torch.Tensor = torch.randperm(batch_size)
    mixed_x: torch.Tensor = lam * x + (1 - lam) * x[index, :]
    return mixed_x, y  # Note: y remains unchanged

# Label Smoothing (not used with ArcFace, kept here for completeness)
def label_smoothing(y: torch.Tensor, classes: int, smoothing: float = 0.1) -> torch.Tensor:
    y_smoothed: torch.Tensor = torch.full((y.size(0), classes), smoothing / (classes - 1))
    y_smoothed.scatter_(1, y.unsqueeze(1), 1.0 - smoothing)
    return y_smoothed

# ArcFace Loss (expects hard integer labels)
class ArcFaceLoss(nn.Module):
    def __init__(self, scale: float = 30.0, margin: float = 0.50) -> None:
        super(ArcFaceLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.cos_m = torch.cos(torch.tensor(margin))
        self.sin_m = torch.sin(torch.tensor(margin))

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine: torch.Tensor = F.normalize(logits, dim=-1)
        sine: torch.Tensor = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi: torch.Tensor = cosine * self.cos_m - sine * self.sin_m  # Apply margin
        one_hot: torch.Tensor = F.one_hot(labels, num_classes=logits.shape[-1]).float()
        logits_modified = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits_modified *= self.scale
        return F.cross_entropy(logits_modified, labels)

# Synthetic Dataset: generates synthetic waveforms and per–time–step labels.
class SyntheticDiarizationDataset(Dataset):
    def __init__(self, num_samples: int = 1000, num_speakers: int = 5, num_events: int = 3, num_moods: int = 4) -> None:
        self.num_samples = num_samples
        self.num_speakers = num_speakers
        self.num_events = num_events
        self.num_moods = num_moods

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        # Create synthetic features with shape (seq_length, input_features).
        features: torch.Tensor = torch.randn(seq_length, input_features)
        speaker_labels: torch.Tensor = torch.randint(0, self.num_speakers, (seq_length,))
        event_labels: torch.Tensor = torch.randint(0, self.num_events, (seq_length,))
        mood_labels: torch.Tensor = torch.randint(0, self.num_moods, (seq_length,))
        return features, (speaker_labels, event_labels, mood_labels)

# Model combining Wav2Vec2 & Transformer for diarization.
class TransformerDiarizationModel(nn.Module):
    def __init__(self, num_speakers: int, num_events: int, num_moods: int, num_layers: int = 4, nhead: int = 8) -> None:
        super(TransformerDiarizationModel, self).__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        # Project synthetic 40-dimensional input to a single channel waveform.
        self.feature_projection = nn.Linear(input_features, 1)
        # The Wav2Vec2 model outputs hidden states of size 768.
        self.embedding = nn.Linear(768, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.speaker_head = nn.Linear(hidden_size, num_speakers)
        self.event_head = nn.Linear(hidden_size, num_events)
        self.mood_head = nn.Linear(hidden_size, num_moods)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (batch, seq_length, input_features)
        # Project to shape (batch, seq_length, 1) and then squeeze to (batch, seq_length)
        x = self.feature_projection(x).squeeze(-1)
        # Feed the raw waveform to Wav2Vec2. It expects shape (batch, audio_length).
        outputs = self.wav2vec(x)
        # Get the last hidden state with shape (batch, T, 768)
        x = outputs.last_hidden_state
        x = self.embedding(x)
        x = self.transformer(x)
        # Obtain logits for each task (each of shape (batch, T, num_classes)).
        speaker_logits = self.speaker_head(x)
        event_logits = self.event_head(x)
        mood_logits = self.mood_head(x)
        # Interpolate logits along the time axis to match the original seq_length.
        speaker_logits = F.interpolate(speaker_logits.transpose(1, 2),
                                       size=seq_length,
                                       mode='linear',
                                       align_corners=False).transpose(1, 2)
        event_logits = F.interpolate(event_logits.transpose(1, 2),
                                     size=seq_length,
                                     mode='linear',
                                     align_corners=False).transpose(1, 2)
        mood_logits = F.interpolate(mood_logits.transpose(1, 2),
                                    size=seq_length,
                                    mode='linear',
                                    align_corners=False).transpose(1, 2)
        return speaker_logits, event_logits, mood_logits
    
def train():
    # Initialize dataset and model. (Dataset parameters override num_speakers, etc.)
    dataset = SyntheticDiarizationDataset(num_samples=1000, num_speakers=6, num_events=4, num_moods=5)
    model = TransformerDiarizationModel(dataset.num_speakers, dataset.num_events, dataset.num_moods)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion_speaker = ArcFaceLoss()
    criterion_event = nn.CrossEntropyLoss()
    criterion_mood = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    for epoch in range(num_epochs):
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
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {total_loss.item():.4f}")

    print("Training completed!")
    torch.save(model.state_dict(), "diarization_model.pth")


def test_inference():
    model = TransformerDiarizationModel(num_speakers, num_events, num_moods)
    model.load_state_dict(torch.load("diarization_model.pth"))
    model.eval()
    # Generate synthetic input features.
    inputs = torch.randn(1, seq_length, input_features)
    speaker_logits, event_logits, mood_logits = model(inputs)
    print("Inference completed!") 
    print(speaker_logits.shape, event_logits.shape, mood_logits.shape)  
    print(speaker_logits[0, :10, :])  # Print first 10 time steps of speaker logits.

if __name__ == "__main__":
    train()
    test_inference()