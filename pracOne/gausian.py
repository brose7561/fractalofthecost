import torch
import numpy as np
import matplotlib.pyplot as plt

print("PyTorch Version:", torch.__version__)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"processing hardware: {device}")

X, Y = np.mgrid[-4.0:4:0.01, -4.0:4:0.01]

x = torch.Tensor(X)
y = torch.Tensor(Y)

x = x.to(device)
y = y.to(device)

z = torch.exp(-(x**2 + y**2) / 2.0)


plt.imshow(z.cpu().numpy()) 
plt.tight_layout()
plt.show()
