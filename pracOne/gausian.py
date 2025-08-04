import torch
import numpy as np
import matplotlib.pyplot as plt

print("PyTorch Version:", torch.__version__)

# Device configuration for apple metal on scilly 
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"processing hardware: {device}")

# grid for computing image, subdivide the space
X, Y = np.mgrid[-4.0:4:0.01, -4.0:4:0.01]

# load into PyTorch tensors
x = torch.Tensor(X)
y = torch.Tensor(Y)

# transfer to the GPU device
x = x.to(device)
y = y.to(device)

# Compute Gaussian
z = torch.exp(-(x**2 + y**2) / 2.0)


plt.imshow(z.cpu().numpy())  # Updated!
plt.tight_layout()
plt.show()
