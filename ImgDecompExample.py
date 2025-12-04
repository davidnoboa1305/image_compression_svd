# svd_image_compression.py
"""
SVD image compression demo using skimage.data.camera().
Shows original grayscale image and reconstructed images
for several choices of k (rank).
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float

def computeSvdCompression(imageMatrix, k):
    """
    Compute rank-k SVD approximation of imageMatrix.
    Returns reconstructed matrix and singular values used.
    """
    # SVD (economy size)
    U, S, Vt = np.linalg.svd(imageMatrix, full_matrices=False)
    # keep top-k
    Uk = U[:, :k]
    Sk = S[:k]
    Vtk = Vt[:k, :]
    # reconstruct
    A_k = Uk @ np.diag(Sk) @ Vtk
    return A_k, S

def compressionStats(m, n, k):
    """
    Compute approximate storage (number of scalars) for rank-k representation:
    mk + nk + k  (U_k, V_k, and k singular values)
    """
    original = m * n
    compressed = m * k + n * k + k
    reductionRatio = compressed / original
    return original, compressed, reductionRatio

def main():
    # 1) Load the built-in grayscale test image (camera)
    img = data.camera()           # uint8 512x512
    imgFloat = img_as_float(img)  # convert to float in range [0,1]

    m, n = imgFloat.shape
    print(f"Loaded image shape: {m} x {n}")

    # 2) Compute full SVD once so we can slice for many k cheaply
    U, S, Vt = np.linalg.svd(imgFloat, full_matrices=False)
    rank = np.sum(S > 1e-12)
    print(f"Estimated rank (numerical): {rank}")
    print(f"Top 5 singular values: {S[:5]}")

    # 3) Choose k values to test
    kList = [5, 20, 50, 100]

    # Prepare plotting: original + each compressed
    nCols = len(kList) + 1
    fig, axes = plt.subplots(1, nCols, figsize=(4 * nCols, 4))
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

        # Clip numerical jitter
        recon = np.clip(recon, 0.0, 1.0)

        # Error (Frobenius norm) and storage
        froError = np.linalg.norm(imgFloat - recon, ord='fro')
        originalSize, compressedSize, ratio = compressionStats(m, n, k)

        axes[i].imshow(recon, cmap='gray', vmin=0, vmax=1)
        axes[i].set_title(f"k={k}\nerr={froError:.3f}\nsize={compressedSize}/{originalSize}\nratio={ratio:.3f}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

    # Optional: save one compressed image to disk
    kSave = 50
    Uk = U[:, :kSave]
    Sk = S[:kSave]
    Vtk = Vt[:kSave, :]
    recon50 = np.clip(Uk @ np.diag(Sk) @ Vtk, 0.0, 1.0)
    # Save using matplotlib
    plt.imsave("Assets/camera_svd_k50.png", recon50, cmap='gray', vmin=0, vmax=1)
    print("Saved compressed image camera_svd_k50.png")

if __name__ == "__main__":
    main()

