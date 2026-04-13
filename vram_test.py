# Test to see if 16 gb vram is enough to run 4 models (2 of them is 3d)

import torch
from models.encoder_xray import XrayEncoder
from models.encoder_ct import CTEncoder

def test_vram():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_xray = XrayEncoder().to(device)
    model_ct = CTEncoder().to(device)
    
    batch_size = 2
    
    dummy_xray = torch.randn(batch_size, 1, 256, 256, device=device)
    dummy_ct = torch.randn(batch_size, 1, 128, 256, 256, device=device)
    
    out_xray = model_xray(dummy_xray)
    out_ct = model_ct(dummy_ct)
    
    loss = out_xray.sum() + out_ct.sum()
    loss.backward()
    
    allocated_vram = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved_vram = torch.cuda.memory_reserved() / (1024 ** 3)
    
    print(f"Batch Size: {batch_size}")
    print(f"Allocated VRAM: {allocated_vram:.2f} GB")
    print(f"Reserved VRAM: {reserved_vram:.2f} GB")
    print("Test completed successfully.")

if __name__ == "__main__":
    test_vram()