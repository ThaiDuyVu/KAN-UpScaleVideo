import torch
from efficient_kan import KAN

print(f"PyTorch Version: {torch.__version__}")
print(f"GPU Available: {torch.cuda.is_available()}")

# Test thử một lớp KAN đơn giản
model = KAN([2, 5, 1])
test_input = torch.randn(1, 2)
output = model(test_input)
print("KAN layer test successful!")