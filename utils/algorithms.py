import torch


# Fast Largest Eigenvector
def FLE(A, iters=500, eigvec=False):
    v = torch.ones(A.shape[0]) #.to(device)
    for _ in range(iters):
        m = torch.einsum("ij, j -> i",A, v)
        lamb = torch.norm(m)
        v = m/lamb
    if eigvec:
        return torch.norm(torch.einsum("ij, j -> i",A, v)), v
    else:
        return torch.norm(torch.einsum("ij, j -> i",A, v))

# Slow Largest Eigenvector
def SLE(A):
    eig = torch.linalg.eig(A)[0]
    eig = eig.unsqueeze(1)
    return torch.max(torch.norm(eig, dim=1))


def lin_sys(x, A, b):
    y = torch.einsum("ik, k -> i", A, x)
    y = y + b
    return y