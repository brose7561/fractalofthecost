import numpy as np
import matplotlib.pyplot as plt
import time
import torch


# Set parameters for the signal
N = 2048                # Number of sample points
T = 1.0                  # Duration of the signal in seconds
f0 = 1                   # fundamental frequency of the square wave in Hz

# list of harmonic numbers when constructing square wave
harmonics = [1, 3, 5, 20, 100]

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE_REAL = torch.float32   
DTYPE_COMPLEX = torch.complex64

# tecnically uses tensors... not sure why. 
def torch_square_wave(t):
    t = torch.as_tensor(t, dtype=DTYPE_REAL, device=DEVICE)
    return torch.sign(torch.sin(2.0 * torch.pi * f0 * t)).cpu().numpy()


def torch_square_wave_fourier(t, f0, N):
    t = torch.as_tensor(t, dtype=DTYPE_REAL, device=DEVICE)
    n = torch.arange(1, 2*N, 2, dtype=DTYPE_REAL, device=DEVICE)  
    terms = torch.sin(2 * torch.pi * f0 * n[:, None] * t) / n[:, None]

    result = (4 / torch.pi) * terms.sum(dim=0)
    return  result.cpu().numpy()


def np_square_wave(t):
 return np.sign(np.sin(2.0 * np.pi * f0 * t))

# Fourier series approximation of square wave
def np_square_wave_fourier(t, N):
    result = np.zeros_like(t)
    for k in range(N):
        n = 2 * k + 1  # Fourier series of a square wave only contains odd harmonics
        result += np.sin(2 * np.pi * n * f0 * t) / n # add harmonics to reconstruct square function
    return (4 / np.pi) * result


# Create the time vector
# np.linspace generates evenly spaced numbers over a specified interval.
t = np.linspace(0.0, T, N, endpoint=False)

# generate original square wave
torch_square = torch_square_wave(t)

plt.figure(figsize=(12, 8))
# Plot original square wave
plt.subplot(2, 3, 1)
plt.plot(t, torch_square, 'k', label="Square wave")
plt.title("Original Square Wave")
plt.ylim(-1.5, 1.5)
plt.grid(True)
plt.legend()

# Plot Fourier reconstructions
for i, Nh in enumerate(harmonics, start=2):
    plt.subplot(2, 3, i)
    y = torch_square_wave_fourier(t, f0, Nh)
    plt.plot(t, y, label=f"N={Nh} harmonics")
    plt.plot(t, torch_square, 'k--', alpha=0.5, label="Square wave")
    plt.title(f"Fourier Approximation with N={Nh}")
    plt.ylim(-1.5, 1.5)
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()


