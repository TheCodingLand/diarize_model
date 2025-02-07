from pydantic_settings import BaseSettings, SettingsConfigDict

import torch

class Config(BaseSettings):
    #load from config.json if available
    model_config = SettingsConfigDict(json_file="config.json")

    num_speakers: int = 5  # Overridden by the dataset below.
    num_events: int = 3
    num_moods: int = 4
    input_features: int = 40  # Synthetic input dimension.
    seq_length: int = 1000    # Increased from 100 to ensure waveform length is long enough.
    hidden_size: int = 128
    batch_size: int = 32
    num_epochs: int = 2
    learning_rate: float = 0.001
    
    def save(self):
        file_path = self.model_config.get("json_file", "config.json")
        with open(file_path, "w") as f:
            f.write(self.model_dump_json(indent=4))
    

config = Config()