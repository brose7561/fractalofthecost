import numpy as np
import matplotlib.pyplot as plt
import time

import torch


def python_loops_dft(x):
    """
    Computes the Discrete Fourier Transform (DFT) of a 1D signal.

    This is a "naïve" implementation that directly follows the DFT formula,
    which has a time complexity of O(N^2).

    Args:
        x (np.ndarray): The input signal, a 1D NumPy array.

    Returns:
        np.ndarray: The complex-valued DFT of the input signal.
    """
    N = len(x)
    # Create an empty array of complex numbers to store the DFT results
    X = np.zeros(N, dtype=np.complex128)

    # Iterate through each frequency bin (k)
    for k in range(N):
        # For each frequency bin, sum the contributions from all input samples (n)
        for n in range(N):
            # The core DFT formula: x[n] * e^(-2j * pi * k * n / N)
            angle = -2j * np.pi * k * n / N
            X[k] += x[n] * np.exp(angle)

    return X

def torch_dft(x):
    """
    Computes the Discrete Fourier Transform (DFT) of a 1D signal.

    This is a "naïve" implementation that directly follows the DFT formula,
    which has a time complexity of O(N^2).

    Args:
        x (np.ndarray): The input signal, a 1D NumPy array.

    Returns:
        np.ndarray: The complex-valued DFT of the input signal.
    """
    # Select Metal (MPS) if available, otherwise fallback to CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Convert input to a complex tensor on the selected device
    X = torch.as_tensor(x, dtype=torch.complex64, device=device)

    # Signal length
    N = X.shape[0]

    # Create index arrays for n (time) and k (frequency bins)
    n = torch.arange(N, device=device)
    k = n.view(-1, 1)

    # Construct the DFT matrix W using the core formula: e^(-2j * pi * k * n / N)
    W = torch.exp(-2j * torch.pi * k * n / N)

    # Perform matrix multiplication to compute the DFT
    result = W @ X

    # Return result as NumPy array on CPU
    return result.to("cpu").numpy()


# --- Main Script ---

# 1. Generate the Signal
# Parameters for the signal


N_values = [256, 512, 1024, 2048, 4096]


SAMPLE_RATE = 800.0    # Sampling rate in Hz
FREQUENCY = 50.0       # Frequency of the sine wave in Hz

# Calculate sample spacing
T = 1.0 / SAMPLE_RATE

# Create the time vector

methods = [
    ("python loops DFT", python_loops_dft),
    ("torch DFT", torch_dft),
    ]



for N in N_values:
    time.sleep(2)

    print(f"--- DFT/FFT Performance Comparison for data size: {N} ---")

    t = np.linspace(0.0, N * T, N, endpoint=False)
    y = np.sin(FREQUENCY * 2.0 * np.pi * t)

    # Time NumPy's FFT implementation
    start_time_fft = time.time()
    fft_result = np.fft.fft(y)
    end_time_fft = time.time()
    fft_duration = end_time_fft - start_time_fft

    for name, funct in methods:
        start_time = time.time()
        dft_result = funct(y)
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"{name} Execution Time: {duration:.6f} seconds")

        if fft_duration > 0:
            print(f"Numpy FFT is approximately {duration / fft_duration:.2f} times faster than {name}.")
        else:
            print("Numpy FFT was too fast to measure a significant duration difference.")
        print(f"{name} implementation is close to NumPy's FFT: {np.allclose(dft_result, fft_result)}")
        
    print(f"NumPy FFT Execution Time: {fft_duration:.6f} seconds\n\n")
    

    



