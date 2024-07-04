
# Nathan Olah

import numpy as np

def print_step(k, R):
    min = 1e-12
    R[np.abs(R) < min] = 0
    print(f"Transformation for column {k}:\n{R}")

def householder(A):
    m, n = A.shape
    R = A.copy()
    Q = np.eye(m)

    for k in range(n):
        x = R[k:m, k]
        
        # Calculate the norm of x
        norm_x = np.linalg.norm(x)
        
        # Create the vector e1
        e1 = np.zeros_like(x)
        e1[0] = 1
        
        # Calculate the vector
        u = x + np.sign(x[0]) * norm_x * e1
        u = u / np.linalg.norm(u)
        
        # Calculate the Householder matrix H
        H_k = np.eye(m)
        H_k[k:m, k:m] -= 2 * np.outer(u, u)
        
        # Apply the transformation to R and Q
        R = np.dot(H_k, R)
        Q = np.dot(Q, H_k)

        print_step(k+1, R)

    return Q, R

A = np.array([[1, -1, 4],
              [1,  4, -2],
              [1,  4,  2],
              [1, -1,  0]], dtype=float)

# Apply Householder transformation
Q, R = householder(A)

# Adjust small values that should be zero 
min = 1e-12
R[np.abs(R) < min] = 0

print("Orthogonal matrix Q:\n", Q)
print("Upper triangular matrix R:\n", R)
