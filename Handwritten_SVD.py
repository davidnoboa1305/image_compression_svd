#Pseudocode for SVD
from numpy import eye, diag, sqrt, argsort, zeros
import numpy as np

def _vector_norm(x):
    #Compute Euclidean norm of vector x without np.linalg.norm.
    return np.sqrt((x * x).sum())

def householder_qr(A):
    """
    Compute the QR decomposition of matrix A (m x n) using Householder reflections.
    Does not use any functions from np.linalg.

    Returns:
        Q (m x m) orthogonal matrix
        R (m x n) upper-triangular (R's lower part may be nonzero if m>n, but R[:n,:] is triangular)
    Guarantee:
        A_original ≈ Q @ R
    """
    A = np.array(A, dtype=float)  # make a copy and ensure float
    m, n = A.shape

    # Initialize Q as identity (m x m). We will form full m x m Q.
    Q = np.eye(m)

    # Work on each column k up to min(m, n) - 1
    for k in range(min(m, n)):
        # Extract x: the k-th column from row k to m-1
        x = A[k:, k].copy()           # shape (m-k,)

        # Compute norm of x
        norm_x = _vector_norm(x)

        if norm_x == 0.0:
            # Nothing to do for this column
            continue

        # sign to avoid cancellation: sign = 1 if x0 >=0 else -1
        sign = 1.0 if x[0] >= 0.0 else -1.0

        # Form v = x + sign * ||x|| * e1
        e1 = np.zeros_like(x)
        e1[0] = 1.0
        v = x + sign * norm_x * e1

        # If v is (almost) zero, skip (rare)
        v_norm_sq = (v * v).sum()
        if v_norm_sq == 0.0:
            continue

        # Householder submatrix: H_sub = I - 2 * v v^T / (v^T v)
        # Instead of forming H_full explicitly for whole matrix, we will build H_full and apply.
        H_sub = np.eye(v.shape[0]) - 2.0 * np.outer(v, v) / v_norm_sq

        # Build full-size Householder H_k (m x m), identity except lower-right block replaced
        H_k = np.eye(m)
        H_k[k:, k:] = H_sub

        # Apply H_k on left to A (A <- H_k @ A)
        A = H_k @ A

        # Accumulate Q = Q @ H_k (so final Q = H1 H2 ... H_{n-1})
        Q = Q @ H_k

    # After all reflections, A has become R
    R = A

    # Optional: zero-out tiny values below the diagonal (numerical cleanup)
    eps = 1e-12
    for i in range(m):
        for j in range(min(i, n)):
            if abs(R[i, j]) < eps:
                R[i, j] = 0.0

    return Q, R


def handwritten_svd(A, num_iterations):
    # --- Step 1: Form the symmetric matrix ---
    S = A.T @ A
    n = S.shape[0]           # <<-- FIX: use integer size, not the tuple
    # --- Step 2: Compute eigenvalues and eigenvectors of S ---
    A_k = S.copy()           # The matrix in the QR iteration
    V = np.eye(n)            # Accumulator for eigenvectors (use numpy's eye)

    for i in range(num_iterations):
        Q, R = householder_qr(A_k)  # A_k = Q @ R
        A_k = R @ Q                 # A_(k+1) = R @ Q
        V = V @ Q                   # accumulate eigenvectors

    # After iteration, A_k is (approximately) diagonal of eigenvalues
    eigenvalues = np.diag(A_k)      # use numpy diag to extract the diagonal

    # --- Step 3: Get singular values and sort ---
    singular_values = sqrt(np.abs(eigenvalues))

    # Sort in descending order
    sort_indices = argsort(singular_values)[::-1]
    singular_values = singular_values[sort_indices]
    V = V[:, sort_indices]  # Sort V columns to match

    # --- Step 4: Form the Sigma matrix ---
    m, nA = A.shape
    Sigma = zeros((m, nA))
    minmn = min(m, nA)
    Sigma[:minmn, :minmn] = np.diag(singular_values)

    # --- Step 5: Compute U ---
    U = zeros((m, m))
    for i in range(minmn):
        if singular_values[i] > 1e-12:
            U[:, i] = (A @ V[:, i]) / singular_values[i]
        else:
            # leave as zeros; to get full U you'd fill remaining with orthonormal basis
            U[:, i] = 0.0

    return U, Sigma, V.T


def main():
    A = np.array([
        [3.0, 1.0],
        [1.0, 3.0],
        [1.0, 1.0]
    ])

    print("Original matrix A:\n", A)

    num_iterations = 100
    U, Sigma, VT = handwritten_svd(A, num_iterations)

    print("\nMatrix U (first few cols):\n", np.round(U[:, :min(4, U.shape[1])], 6))
    print("\nMatrix Σ:\n", np.round(Sigma, 6))
    print("\nMatrix V^T:\n", np.round(VT, 6))

    A_reconstructed = U @ Sigma @ VT
    print("\nReconstructed A:\n", np.round(A_reconstructed, 6))

    error = _vector_norm(A - A_reconstructed)
    print(f"\nReconstruction error (‖A - UΣVᵀ‖): {error:.6e}")

    # Compare singular values with NumPy's SVD
    U_np, S_np, VT_np = np.linalg.svd(A)
    print("\nNumPy singular values:", np.round(S_np, 6))


if __name__ == "__main__":
    main()
