# Adjust the import path as necessary.
from diarization import SyntheticDiarizationDataset
from settings import config

def test_dataset_item_shape():
    num_samples = 10
    num_speakers = 6
    num_events = 4
    num_moods = 5
    dataset = SyntheticDiarizationDataset(num_samples, num_speakers, num_events, num_moods)
    
    # Get a sample from the dataset.
    features, (speaker_labels, event_labels, mood_labels) = dataset[0]
    
    # Check that the features have the expected shape.
    assert features.shape == (config.seq_length, config.input_features)
    
    # Check that each label tensor has shape (seq_length,)
    assert speaker_labels.shape == (config.seq_length,)
    assert event_labels.shape == (config.seq_length,)
    assert mood_labels.shape == (config.seq_length,)
