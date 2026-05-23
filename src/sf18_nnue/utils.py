import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import data_loader

class ClippedReLU(nn.Module):
    def __init__(self, clip_value=1.0):
        super().__init__()
        self.clip_value = clip_value
        
    def forward(self, input):
        return torch.clamp(input, 0, self.clip_value)
    
class SqrClippedReLU(nn.Module):
    def __init__(self, scale_factor=127/128, clip_value=1.0):
        super().__init__()
        self.scale_factor = scale_factor
        self.clip_value = clip_value

    def forward(self, input):
        return torch.clamp((input ** 2) * self.scale_factor, 0, self.clip_value)
    
def make_data_loaders(
    train_filenames,
    feature_name: str,
    num_workers,
    batch_size,
    config: data_loader.DataloaderSkipConfig,
    epoch_size,
    pin_memory,
    queue_size_limit,
    prefetch_device=None,
):
    # Epoch and validation sizes are arbitrary
    features_name = feature_name
    train_infinite = data_loader.SparseBatchDataset(
        features_name,
        train_filenames,
        batch_size,
        num_workers=num_workers,
        config=config,
    )
    # num_workers has to be 0 for sparse, and 1 for dense
    # it currently cannot work in parallel mode but it shouldn't need to
    train = DataLoader(
        data_loader.FixedNumBatchesDataset(
            train_infinite,
            (epoch_size + batch_size - 1) // batch_size,
            pin_memory=pin_memory,
            queue_size_limit=queue_size_limit,
            device=prefetch_device,
        ),
        batch_size=None,
        batch_sampler=None,
        num_workers=0,
    )
    
    return train