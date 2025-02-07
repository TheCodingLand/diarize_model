
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import Wav2Vec2Model  
from settings import config 
from typing import Any, Tuple

try: 
    from clearml import Task # type: ignore 
    try:
        task = Task.init(project_name="my_project", task_name="experiment_1") # type: ignore
    except Exception as e:
        task = Task.current_task() # type: ignore

except ImportError:
    pass

device: str = "cuda" if torch.cuda.is_available() else "cpu"

class ArcFaceLoss(nn.Module):
    def __init__(self, scale: float = 30.0, margin: float = 0.50) -> None:
        super(ArcFaceLoss, self).__init__() # type: ignore
        self.scale = scale
        self.margin = margin
        self.register_buffer('cos_m', torch.cos(torch.tensor(margin)))
        self.register_buffer('sin_m', torch.sin(torch.tensor(margin)))

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine: torch.Tensor = F.normalize(logits, dim=-1)
        sine: torch.Tensor = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi: torch.Tensor = cosine * self.cos_m - sine * self.sin_m  # Apply margin
        one_hot: torch.Tensor = F.one_hot(labels, num_classes=logits.shape[-1]).float()
        logits_modified = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits_modified *= self.scale
        return F.cross_entropy(logits_modified, labels)

# Synthetic Dataset: generates synthetic waveforms and per–time–step labels.
class SyntheticDiarizationDataset(Dataset[Any]):
    def __init__(self, num_samples: int = 1000, num_speakers: int = 5, num_events: int = 3, num_moods: int = 4) -> None:
        self.num_samples = num_samples
        self.num_speakers = num_speakers
        self.num_events = num_events
        self.num_moods = num_moods
        

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        # Create synthetic features with shape (seq_length, input_features).
        features: torch.Tensor = torch.randn(config.seq_length, config.input_features)
        speaker_labels: torch.Tensor = torch.randint(0, self.num_speakers, (config.seq_length,))
        event_labels: torch.Tensor = torch.randint(0, self.num_events, (config.seq_length,))
        mood_labels: torch.Tensor = torch.randint(0, self.num_moods, (config.seq_length,))
        # update config data and save it
        
        
        return features, (speaker_labels, event_labels, mood_labels)

# Model combining Wav2Vec2 & Transformer for diarization.
class TransformerDiarizationModel(nn.Module):
    def __init__(self, num_speakers: int, num_events: int, num_moods: int, num_layers: int = 4, nhead: int = 8) -> None:
        super(TransformerDiarizationModel, self).__init__() # type: ignore
        self.wav2vec : Wav2Vec2Model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h") # type: ignore
        # Project synthetic 40-dimensional input to a single channel waveform.
        self.feature_projection = nn.Linear(config.input_features, 1)
        # The Wav2Vec2 model outputs hidden states of size 768.
        self.embedding = nn.Linear(768, config.hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.speaker_head = nn.Linear(config.hidden_size, num_speakers)
        self.event_head = nn.Linear(config.hidden_size, num_events)
        self.mood_head = nn.Linear(config.hidden_size, num_moods)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (batch, seq_length, input_features)
        # Project to shape (batch, seq_length, 1) and then squeeze to (batch, seq_length)
        x = self.feature_projection(x).squeeze(-1)
        # Feed the raw waveform to Wav2Vec2. It expects shape (batch, audio_length).
        x = x.to(device)
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
        speaker_logits: torch.Tensor = F.interpolate(speaker_logits.transpose(1, 2),    # type: ignore
                                       size=config.seq_length,
                                       mode='linear',
                                       align_corners=False).transpose(1, 2)
        event_logits: torch.Tensor = F.interpolate(event_logits.transpose(1, 2),        # type: ignore
                                     size=config.seq_length,
                                     mode='linear',
                                     align_corners=False).transpose(1, 2) # type: ignore
        mood_logits: torch.Tensor = F.interpolate(mood_logits.transpose(1, 2),  # type: ignore
                                    size=config.seq_length,
                                    mode='linear',
                                    align_corners=False).transpose(1, 2) 
        logging.critical(type(mood_logits))
        return speaker_logits, event_logits, mood_logits  #type: ignore