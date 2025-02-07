from pydantic_settings import BaseSettings


class Config(BaseSettings):

    num_speakers: int = 5  # Overridden by the dataset below.
    num_events: int = 3
    num_moods: int = 4
    input_features: int = 40  # Synthetic input dimension.
    seq_length: int = 1000    # Increased from 100 to ensure waveform length is long enough.
    hidden_size: int = 128
    batch_size: int = 32
    num_epochs: int = 2
    learning_rate: float = 0.001


config = Config()