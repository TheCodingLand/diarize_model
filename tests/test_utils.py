import torch
import pytest
import numpy as np
# Adjust the import path as necessary.
from utils import mixup_data, label_smoothing

def test_mixup_data():
    batch_size = 4
    seq_length = 1000
    input_features = 40
    dummy_input = torch.randn(batch_size, seq_length, input_features)
    # Create dummy labels (e.g., for speaker classification) with shape (batch_size, seq_length)
    dummy_labels = torch.randint(0, 6, (batch_size, seq_length))
    
    mixed_input, mixed_labels = mixup_data(dummy_input, dummy_labels)
    
    # Ensure input shape is unchanged and labels remain unmodified.
    assert mixed_input.shape == dummy_input.shape
    assert torch.equal(mixed_labels, dummy_labels)

def test_label_smoothing():
    batch_size = 4
    num_classes = 6
    dummy_labels = torch.randint(0, num_classes, (batch_size,))
    smoothed = label_smoothing(dummy_labels, num_classes, smoothing=0.1)
    
    # The smoothed labels should have shape (batch_size, num_classes)
    assert smoothed.shape == (batch_size, num_classes)
    
    # Check that each label position has value near (1 - smoothing) and the others near (smoothing / (num_classes - 1)).
    for i, label in enumerate(dummy_labels):
        target_val = 1 - 0.1
        off_val = 0.1 / (num_classes - 1)
        assert abs(smoothed[i, label].item() - target_val) < 1e-5
        # You can also check one of the non-target positions.
        non_target_idx = (label + 1) % num_classes
        assert abs(smoothed[i, non_target_idx].item() - off_val) < 1e-5
