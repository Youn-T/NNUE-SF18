import torch.nn as nn
import torch

class ClippedReLU(nn.Module):
    def __init__(self, clip_value=127):
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