import numpy as np

def householder_qr(A):
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

def backward_substitution(R, b):
    n = len(b)
    x = np.zeros_like(b)
    for i in reversed(range(n)):
        x[i] = (b[i] - np.dot(R[i, i+1:], x[i+1:])) / R[i, i]
    return x

def solve_least_squares(A, b):
    Q, R = householder_qr(A)
    Qb = np.dot(Q.T, b)
    return backward_substitution(R, Qb[:R.shape[1]])

# Constructing the matrix A and vector b
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

# Solving the least squares problem
x = solve_least_squares(A, b)

# Printing the results
print("Matrix A after applying all the Householder transformations:")
Q, R = householder_qr(A)
print(R)
print("\nCalculated values of the altitudes:")
print(x)

# Comparing the calculated values with the direct measurements
direct_measurements = np.array([2.95, 1.74, -1.45, 1.32])
# delta_h = direct_measurements - x
delta_h = x - direct_measurements
print("\nDifference between direct measurements and calculated values (Δh):")
print(delta_h)

# def householder_qr(A):
#     m, n = A.shape
#     R = A.copy()
#     Q = np.eye(m)

#     for k in range(n):
#         # compute the Householder vector
#         x = R[k:, k]
#         e1 = np.zeros_like(x)
#         e1[0] = np.sign(x[0]) * np.linalg.norm(x)
#         v = x + e1
#         v = v / np.linalg.norm(v)

#         # update the matrix R
#         R[k:, k:] -= 2 * np.outer(v, np.dot(v, R[k:, k:]))

#         # update the orthogonal matrix Q
#         Q_k = np.eye(m)
#         Q_k[k:, k:] -= 2 * np.outer(v, v)
#         Q = np.dot(Q, Q_k)

#     return Q, R


# # Example usage
# # A = np.array([
# #     [12, -51, 4],
# #     [6, 167, -68],
# #     [-4, 24, -41]
# # ], dtype=float)
# A = np.array([
#     [1, 0, 0, 0],
#     [0, 1, 0, 0],
#     [0, 0, 1, 0],
#     [0, 0, 0, 1],
#     [1, -1, 0, 0],
#     [1, 0, -1, 0],
#     [1, 0, 0, -1],
#     [0, 1, -1, 0],
#     [0, 1, 0, -1],
#     [0, 0, 1, -1]
# ], dtype=float)

# Q, R = householder_qr(A)
# print("Q:\n", Q)
# print("R:\n", R)