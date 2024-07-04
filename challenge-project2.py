
# Nathan Olah

import numpy as np

def backwardSubstitution(U, y):
    # Ux = y
    n = len(y)
    x = np.zeros(n)

    for i in range(n-1, -1, -1):
        x[i] = y[i]
        for j in range(i+1, n):
            x[i] -= U[i, j] * x[j] 
        x[i] /= U[i, i]
    
    return x

def householder(A):
    m, n = A.shape
    R = A.copy()
    Q = np.eye(m)

    for k in range(n):
        # compute the Householder vector
        x = R[k:, k]
        e1 = np.zeros_like(x)
        e1[0] = np.sign(x[0]) * np.linalg.norm(x)
        v = x + e1
        v = v / np.linalg.norm(v)

        # update matrix R
        R[k:, k:] -= 2 * np.outer(v, np.dot(v, R[k:, k:]))

        # update the orthogonal matrix Q
        Q_k = np.eye(m)
        Q_k[k:, k:] -= 2 * np.outer(v, v)
        Q = np.dot(Q, Q_k)

    return Q, R

#
A = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [1, -1, 0, 0],
    [1, 0, -1, 0],
    [1, 0, 0, -1],
    [0, 1, -1, 0],
    [0, 1, 0, -1],
    [0, 0, 1, -1]
], dtype=float)

b = np.array([2.95, 1.74, -1.45, 1.32, 1.23, 4.45, 1.61, 3.21, 0.45, -2.75])

# Solve least squares problem
Q, R = householder(A)

# Adjust small values that should be zero 
min = 1e-12
R[np.abs(R) < min] = 0

Qb = np.dot(Q.T, b)
x_hat = backwardSubstitution(R, Qb[:R.shape[1]])

# Print results
print("Matrix A after applying the Householder transformations:")
print(R)

print("\nCalculated values of the altitudes (x_hat):")
print(x_hat)

# Calculate differences between calculated values and direct measurements
direct_measurements = np.array([2.95, 1.74, -1.45, 1.32])
delta_x = x_hat - direct_measurements
print("\nDifference between direct measurements and calculated values (Δx):")
print(delta_x)
