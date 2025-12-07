"""
SVD Analysis: Information Retained vs Compression Ratio.
Based on the formulas provided:
1. Compression Ratio = (k(m + n + 1)) / (m * n)
2. Information Retained = sum(sigma_i^2 from 1 to k) / sum(sigma_i^2 from 1 to r)
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.util import img_as_float


def main():
    # Load Data
    img = data.camera()
    imgFloat = img_as_float(img)
    m, n = imgFloat.shape
    print(f"Image shape: {m} x {n}")

    # Compute Full SVD
    # We only need S (singular values) for the chart,
    U, S, Vt = np.linalg.svd(imgFloat, full_matrices=False)

    # Calculate "Information Retained" for all possible k
    # Formula: sum(sigma^2 from 1 to k) / sum(total sigma^2)
    singular_values_squared = S ** 2
    total_variance = np.sum(singular_values_squared)

    # cumsum get the sum for k=1, k=2, ... k=N instantly
    explained_variance_ratio = np.cumsum(singular_values_squared) / total_variance

    # 4. Calculate Data Points for k = 1 to 200
    k_max = 200
    k_values = np.arange(1, k_max + 1)

    compression_ratios = []
    info_retained_values = []

    total_pixels = m * n

    for k in k_values:
        # Formula from your image: Compressed / Original
        # Compressed size = k vectors for U + k vectors for V + k singular values
        compressed_size = k * (m + n + 1)
        ratio = compressed_size / total_pixels

        compression_ratios.append(ratio)

        # S arrays are 0-indexed, so k=1 is index 0
        info = explained_variance_ratio[k - 1]
        info_retained_values.append(info)

    # 5. Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(compression_ratios, info_retained_values, linewidth=2, label='k = 1 to 200')

    # Formatting the chart
    plt.title(f"Information Retained vs Compression Ratio (k=1 to {k_max})")
    plt.xlabel("Compression Ratio (Lower is better)")
    plt.ylabel("Information Retained (Higher is better)")
    plt.grid(True, linestyle='--', alpha=0.6)

    # Highlight specific points
    idx_50 = 49  # k=50 is index 49
    plt.scatter(compression_ratios[idx_50], info_retained_values[idx_50], color='red', zorder=5)
    plt.text(compression_ratios[idx_50], info_retained_values[idx_50] - 0.05,
             f' k=50\n Info retained: {info_retained_values[idx_50] * 100:.1f}%', color='red')
    idx_100 = 99  # k=100 is index 99
    plt.scatter(compression_ratios[idx_100], info_retained_values[idx_100], color='red', zorder=5)
    plt.text(compression_ratios[idx_100], info_retained_values[idx_100] - 0.05,
             f' k=100\n Info retained: {info_retained_values[idx_100] * 100:.1f}%', color='red')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()