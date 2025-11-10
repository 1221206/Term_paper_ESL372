
import torch

if torch.cuda.is_available():
    print("Success! Your GPU is available to PyTorch.")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("PyTorch cannot find a CUDA-enabled GPU. It will use the CPU instead.")
