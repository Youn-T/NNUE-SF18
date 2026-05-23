import data_loader
import torch
import torch.nn as nn
from sf18_nnue.utils import make_data_loaders
from sf18_nnue.model import NNUE

EPOCHS = 10

def training_loop(dataloader, model):
    print("Training...", flush=True)
    model.train()
    
    for i, x in enumerate(data_loader):
        print(f"Batch: {i}")

if __name__ == "__main__":
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print("Using {} device".format(device))
    is_cuda = device == "cuda"
    
    data_loader = make_data_loaders(        
        ["D:/Projects/NNUE_SF_13/T60T70wIsRightFarseer.binpack"],
        feature_name="Full_Threats+HalfKAv2_hm",
        num_workers=8,
        batch_size=1024 * 8,
        config= data_loader.DataloaderSkipConfig(
            filtered=False,
            random_fen_skipping=0,
            wld_filtered=False,
            early_fen_skipping=False
        ),
        epoch_size=500_000_000,
        pin_memory=True,
        queue_size_limit=16
    )
    
    model = NNUE().to(device)
    
    for epoch in range(EPOCHS):
        training_loop(data_loader, model)