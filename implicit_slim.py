import numpy as np
from scipy.sparse.linalg import eigsh


def implicit_slim(W, X, λ, α, thr):
    A = W.copy().astype(np.float16)
    
    D = 1 / (np.array(X.sum(0)) + λ)
    
    ind = (np.array(X.sum(axis=0)) < thr).squeeze()
    A[:, ind.nonzero()[0]] = 0
    
    M = (λ * A + A @ X.T @ X) * D * D
    
    AinvC = λ * M + M @ X.T @ X
    AinvCAt = AinvC @ A.T

    AC = AinvC - AinvCAt @ np.linalg.inv(np.eye(A.shape[0]) / α + AinvCAt) @ AinvC
    
    return α * W @ A.T @ AC


def slim_lle_1(X, λ):
    B = np.array((X.T * X).toarray())
    diagIndices = np.diag_indices(B.shape[0])
    B[diagIndices] += λ
    B = np.linalg.inv(B)
    B1 = B.sum(1, keepdims=True, dtype=np.float64)
    B = B - ((B1 / np.sqrt(B1.sum())) @ (B1.T / np.sqrt(B1.sum())))
    B = B / (-np.diag(B))
    B[diagIndices] = 0
    
    return B


def slim_lle_2(B, dim):
    W = B.copy().T
    diagIndices = np.diag_indices(W.shape[0])
    W[diagIndices] -= 1

    M = W.T @ W

    return eigsh(M, k=dim+1, sigma=0.0)[1][:,1:]