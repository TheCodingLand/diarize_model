# Diarization Model with Wav2Vec2 & Transformer

This repository contains a multi-task diarization model built with PyTorch. The model leverages a pre-trained [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base-960h) backbone combined with a Transformer encoder to perform the following tasks on audio inputs:
- **Speaker Diarization**
- **Event Detection**
- **Mood Detection**

Additionally, a synthetic dataset generator is provided for rapid prototyping, along with data preprocessing utilities and unit tests.


## Features

- **Multi-Task Learning:**  
  Simultaneously performs speaker, event, and mood classification on per–time–step basis.

- **Pre-trained Wav2Vec2 Backbone:**  
  Uses a robust pre-trained model to extract features from raw audio.

- **Transformer Encoder:**  
  Captures temporal dependencies in the extracted features.

- **ArcFace Loss for Speaker Diarization:**  
  Enhances class separability with an additive angular margin.

- **Mixup Data Augmentation:**  
  Applies mixup augmentation to the input data for better generalization (only on inputs; hard targets are preserved for ArcFace).

- **Time-Dimension Interpolation:**  
  Interpolates downsampled outputs back to the original sequence length to match per–time–step labels.

## Dataset Structure

### SyntheticDiarizationDataset
A synthetic dataset is provided to simulate real-world inputs:
- **Input Features:**  
  Each sample is a tensor of shape `(seq_length, input_features)` (default: `(1000, 40)`).
  
- **Labels:**  
  For each time step, the dataset generates:
  - **Speaker Labels:** Integers in `[0, num_speakers)`  
  - **Event Labels:** Integers in `[0, num_events)`  
  - **Mood Labels:** Integers in `[0, num_moods)`

*Custom Datasets:*  
You can replace or extend the synthetic dataset with your own data. Simply ensure your dataset returns audio features and corresponding per–time–step labels.

## Requirements

- Python 3.8+
- [PyTorch](https://pytorch.org/) 1.7+
- [Transformers](https://huggingface.co/transformers/)
- [ClearML](https://clear.ml/) (optional, for experiment tracking)
- NumPy
- (Optional) [torchaudio](https://pytorch.org/audio/stable/) for data preprocessing

Install the dependencies via:

```bash
pip install -r requirements.txt
```

## Installation

Clone the repository and set up your virtual environment:

```bash
git clone https://github.com/yourusername/diarization-model.git
cd diarization-model

python -m venv venv
# Activate the environment (Linux/Mac)
source venv/bin/activate
# or on Windows
venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

### Training the Model

Run the training script to train the model on the synthetic dataset:

```bash
python train.py
```

This will:
- Initialize the synthetic dataset
- Build the model with Wav2Vec2 and Transformer components
- Train using mixup augmentation and ArcFace loss for speaker classification
- Save the model weights as `diarization_model.pth`

### Inference

To run inference on a sample input:

```bash
python infer.py
```

*(Make sure your `infer.py` loads the model weights and applies the same preprocessing as in training.)*

## Data Preprocessing

For real-world applications, you might want to preprocess your audio files. Below is an example of a preprocessing function (`preprocessing.py`) that loads an audio file, resamples it to a target sample rate, and normalizes it:

```python
import torchaudio
import torch

def preprocess_audio(file_path: str, target_sample_rate: int = 16000) -> torch.Tensor:
    """
    Load and preprocess an audio file.

    Parameters:
        file_path (str): Path to the audio file.
        target_sample_rate (int): The desired sample rate.

    Returns:
        torch.Tensor: Normalized audio waveform.
    """
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    # Normalize the waveform: zero mean and unit variance
    waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-5)
    return waveform
```

Integrate this function into your data loading pipeline when working with real audio files.

## Tests

Unit tests are provided in the `tests/` directory. They include:

- **Model Forward Pass Test:**  
  Ensures the model correctly processes input data and returns outputs with the expected shapes.

- **Data Preprocessing Test:**  
  Validates that the preprocessing function outputs tensors with the correct shape and normalized values.

- **Dataset Test:**  
  Checks that the synthetic dataset yields data in the proper format.

Run all tests using [pytest](https://docs.pytest.org/en/latest/):

```bash
pytest tests/
```

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for improvements and new features.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.md) file for details.

