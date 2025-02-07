import os
from pathlib import Path
from typing import Any, Dict
from pydantic import model_validator, validator
from pydantic_settings import BaseSettings, SettingsConfigDict

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
    save_model_path: str = f"{Path(__file__).parent.parent}/model/diarization_model.pth"
    
    def save(self):
        file_path = self.model_config.get("json_file", "config.json")
        with open(file_path, "w") as f:
            f.write(self.model_dump_json(indent=4))

    @model_validator(mode="before")
    def validate_save_model_path(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        value = values.get("save_model_path", f"{Path(__file__).parent.parent}/model/diarization_model.pth")
        if not value.endswith(".pth"):
            raise ValueError("Model path must end with .pth")
        print(f"creating directory {Path(value).parent}")
        os.makedirs(os.path.dirname(Path(value)), exist_ok=True)
        values["save_model_path"] = value
        return values
    

config = Config()