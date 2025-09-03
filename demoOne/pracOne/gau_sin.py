import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

X, Y = np.mgrid[-4.0:4.0:0.01, -4.0:4.0:0.01]
x = torch.tensor(X, dtype=torch.float32).to(device)
y = torch.tensor(Y, dtype=torch.float32).to(device)

# gausian
#z = torch.exp(-(x**2 + y**2) / 2.0)

# Compute 2D sine function
#z = torch.sin(2 * (x + y))  

#both
z = torch.sin(2 * (x + y)) * torch.exp(-(x**2 + y**2) / 2.0)

plt.imshow(z.cpu().numpy(), cmap='viridis', extent=[-4, 4, -4, 4])
plt.colorbar(label='Sine Intensity')
plt.title('2D Sine Function (Stripes)')
plt.tight_layout()
plt.show()
