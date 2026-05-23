
import torch.nn as nn
import torch
from sf18_nnue.utils import ClippedReLU, SqrClippedReLU

class SinglePerspectiveSubnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_embeddings = nn.EmbeddingBag(82673, 1024, padding_idx=0)
        self.input_bias = nn.Parameter(torch.zeros(1024))
        
        self.psq_embeddings = nn.EmbeddingBag(82673, 8, padding_idx=0)
        self.psq_bias = nn.Parameter(torch.zeros(8))
        
        self.clipped_relu = ClippedReLU()

    def forward(self, input):
        Z1 = self.input_embeddings(input) + self.input_bias
        A1 = self.clipped_relu(Z1)
        
        out = A1[:512] * A1[512:]
        
        psq_Z1 = self.psq_embeddings(input) + self.psq_bias
        psq_A1 = self.clipped_relu(psq_Z1)
        
        return out, psq_A1
    
class MainSubnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(1024, 32)
        self.sqr_clipped_relu = SqrClippedReLU()
        self.clipped_relu = ClippedReLU()
        self.linear_2 = nn.Linear(62,32)
        self.linear_3 = nn.Linear(32,1)
    
    def forward(self, input):
        Z1 = self.linear_1(input)
        Z1_31 = Z1[:31]
        Z1_skip = Z1[31:]
        
        A1_sqr = self.sqr_clipped_relu(Z1_31)
        A1_classic = self.clipped_relu(Z1_31)
        
        A1_concaneted = torch.cat([A1_sqr, A1_classic], dim=1)
        
        Z2 = self.linear_2(A1_concaneted)
        A2 = self.clipped_relu(Z2)
        
        out = self.linear_3(A2) + Z1_skip
        
        return out

class NNUE(nn.Module):
    def __init__(self):
        super().__init__()
        self.single_perspective_subnet = SinglePerspectiveSubnet()
        self.main_subnets = nn.ModuleList([ MainSubnet() for _ in range(8) ])
    
    def forward(self, input):
        
        input_our, input_their, piece_count = input
        
        our_subnet, our_psq = self.single_perspective_subnet(input_our)
        their_subnet, their_psq = self.single_perspective_subnet(input_their)
        
        average_psq = (our_psq - their_psq) / 2
        
        concaneted_subnet = torch.cat([our_subnet, their_subnet], dim=1)
        
        element_indexes = torch.floor((piece_count - 1) // 4)
        
        output = torch.zeros(input_our.size(0), 1)
        
        for i in range(8):
            mask = (element_indexes == i)
            if mask.any():
                output[mask] = self.main_subnets[i](concaneted_subnet[mask]) + average_psq[i]
                
        return output * 600
        
        