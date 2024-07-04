

import numpy as np
from tabulate import tabulate

def rayleigh_quotient_iteration(A, e, tolerance=0.0001):
    n = len(e)
    k = 0
    lambda_old = np.dot(e.T, np.dot(A, e))
    
    while True:
        k += 1

        w = np.linalg.solve(A - lambda_old * np.eye(n), e)
        e = w / np.linalg.norm(w) # normalize
        lambda_new = np.dot(e.T, np.dot(A, e)) # calculate new lambda
        
        # check tolerance
        if abs(lambda_new - lambda_old) < tolerance:
            break
        
        lambda_old = lambda_new

    return lambda_old, k

def gram_schmidt(A):
    n = A.shape[0]
    Q = np.zeros((n, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j]
        
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]
        
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j] # normalize v to get the j-th column of Q

    # returns orthogonal matrix Q and upper triangular matrix R
    return Q, R 

def QR_iteration(A, tolerance=0.0001):
    A_new = A.copy()
    iterations = 0

    while True:
        Q, R = gram_schmidt(A_new)
        A_new = np.dot(R, Q)
        iterations += 1

        if iterations > 1:
            # get max difference between the diagonal elements of current and previous A matrix
            eigenvalue_diff = np.max(np.abs(np.diag(A_new) - np.diag(A_prev)))
            if eigenvalue_diff < tolerance:
                break
        
        A_prev = A_new.copy()

    # get the diagonal elements of the converged matrix as eigenvalues
    eigenvalues = np.diag(A_new)
    
    return eigenvalues, iterations

def main():
    A = np.array([
        [2.9766, 0.3945, 0.4198, 1.1159],
        [0.3945, 2.7328, -0.3097, 0.1129],
        [0.4198, -0.3097, 2.5675, 0.6079],
        [1.1159, 0.1129, 0.6079, 1.7231]
    ])

    # starting vectors e1, e2, e3, e4 
    starting_vectors = [
        np.array([1, 0, 0, 0], dtype=float),
        np.array([0, 1, 0, 0], dtype=float),
        np.array([0, 0, 1, 0], dtype=float),
        np.array([0, 0, 0, 1], dtype=float)
    ]

    # Algorithm 1
    results = []
    for i, e in enumerate(starting_vectors, 1):
        eigenvalue, iterations = rayleigh_quotient_iteration(A, e)
        results.append([i, e.astype(int), eigenvalue, iterations])

    for result in results:
        result[1] = str(result[1])

    headers = ["#", "Starting Eigen vector", "Eigen value", "Number of iterations"]
    table = tabulate(results, headers, tablefmt="simple")
    print(table)

    # Algorithm 2
    eigenvalues, iterations = QR_iteration(A)

    print("\nThe eigen values are:")
    print(eigenvalues)
    print(f"The number of iterations for the convergence is: {iterations}")

if __name__ == "__main__":
    main()
