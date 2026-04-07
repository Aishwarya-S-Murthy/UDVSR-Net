# command to run this .py file:  python -m basicsr.test_autograd

import torch
from basicsr.models.archs.UVSR_temporal_simple_arch import VideoSRNet

def test_autograd():
    print("1. Initializing Model...")
    # Use the exact settings from your YAML
    model = VideoSRNet(
        embed_dim=48,
        num_spatial_blocks=2,
        num_temporal_blocks=1,
        num_refinement_blocks=2,
        num_heads=4,
        scale_factor=2  # <--- ADD THIS LINE
    ).cuda()
    
    # Create a dummy input: Batch=1, Frames=8, Channels=3, H=64, W=64
    input_tensor = torch.randn(1, 8, 3, 64, 64).cuda()
    gt_tensor = torch.randn(1, 8, 3, 128, 128).cuda() # Output is 2x scale
    
    print("2. Running Forward Pass...")
    try:
        output = model(input_tensor)
        print(f"   Forward Success! Output shape: {output.shape}")
    except Exception as e:
        print(f"   Forward Failed: {e}")
        return

    print("3. Calculating Loss...")
    loss = torch.nn.functional.mse_loss(output, gt_tensor)
    print(f"   Loss: {loss.item()}")

    print("4. Running Backward Pass (Autograd)...")
    try:
        loss.backward()
        print("   Backward Success! Autograd is working.")
    except Exception as e:
        print(f"   Backward Failed! The error is: {e}")
        return

if __name__ == "__main__":
    # Disable cuDNN benchmark for stability during test
    torch.backends.cudnn.benchmark = False
    test_autograd()