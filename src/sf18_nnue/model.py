
import torch.nn as nn

class SinglePerspectiveSubnet(nn.Module):
    def __init__(self):
        pass
    
    def forward(input):
        pass

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.single_perspective_subnet = SinglePerspectiveSubnet()
    
    def forward(self, input):
        
        input_our, input_their = input
        
        our_subnet, our_psq = SinglePerspectiveSubnet(input_our)
        their_subnet, their_psq = SinglePerspectiveSubnet(input_their)
        