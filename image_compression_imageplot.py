import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.util import img_as_float

# This code was written by David Noboa.
# The purpose of this code is to visualize the effect of SVD compression on the image.
# I used Numpy SVD function since the testing image size is too large for our handwritten
# SVD and would take too much time
def compressionStats(m, n, k):
    original = m * n
    compressed = (m * k) + (n * k) + k
    reductionRatio = compressed / original
    return original, compressed, reductionRatio

def main():
    # Load the grayscale test image
    img = data.camera()           # 512x512
    imgFloat = img_as_float(img)  # convert to float in range [0,1]

    m, n = imgFloat.shape
    print(f"Loaded image shape: {m} x {n}")

    # Compute full SVD
    U, S, Vt = np.linalg.svd(imgFloat, full_matrices=False)
    rank = np.sum(S > 1e-12)
    print(f"Estimated rank (numerical): {rank}")
    print(f"Top 5 singular values: {S[:5]}")

    # k values to test
    kList = [5, 20, 50, 100]

    # Prepare plotting
    nCols = len(kList) + 1
    fig, axes = plt.subplots(1, nCols, figsize=(4 * nCols, 5.5))
    # Show original
    axes[0].imshow(imgFloat, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title("Original")
    axes[0].axis('off')

    # For each k reconstruct and show stats
    for i, k in enumerate(kList, start=1):
        # Reconstruct using first k singular values / vectors
        Uk = U[:, :k]
        Sk = S[:k]
        Vtk = Vt[:k, :]
        recon = Uk @ np.diag(Sk) @ Vtk

        # Error (Frobenius norm) and storage
        froError = np.linalg.norm(imgFloat - recon, ord='fro')
        originalSize, compressedSize, ratio = compressionStats(m, n, k)

        axes[i].imshow(recon, cmap='gray', vmin=0, vmax=1)
        axes[i].set_title(f"k={k}\nerr={froError:.3f}\nsize={compressedSize}/{originalSize}\nratio={ratio:.3f}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
