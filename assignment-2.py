# Assignment 2 - Question 1
# Nathan Olah

import numpy as np

def forward_substitution(L, b):
    # Lx = b
    n = len(b)
    x =  np.zeros(n)

    for i in range(n):
        x[i] = b[i]
        for j in range(i):
            x[i] -= L[i, j] * x[j]
        x[i] / L[i, i]
    
    return x


def backward_substitution(U, b, n):
    # Ux = b
    x = np.zeros(n)

    # x[n-1] = b[n-1] / U[n-1, n-1]
    # for i in range(n-2, -1, -1):
    #     sum_j = 0
    #     for j in range(i+1, n):
    #         sum_j += U[i, j] * x[j]
    #     x[i] = (b[i] - sum_j) / U[i, i]

    for i in range(n-1, -1, -1):
        x[i] = b[i]
        for j in range(i+1, n):
            x[i] -= U[i, j] * x[j] 
        x[i] /= U[i, i]
    
    return x

def gauss_elimination(A, b, n):
    # Solving the system Ax = b
    # Forward Elimination

    # a_ij = a_ij - a_ik / a_kk * a_kj
    # b_i = b_i - a_ik / a_kk * b_k
    
    # k: diagonals -- 0 to n-2
    # i: rows -- k+1 to n-1
    # j: cols -- k to n-1

    for k in range(n-1):
        for i in range(k+1, n):
            ratio = A[i, k] / A[k, k]
            for j in range(k, n):
                A[i, j] -= ratio * A[k, j]
            b[i] -= ratio * b[k]
        
        print(f"Step {k+1}: Matrix A:\n{A}\nVector b:\n{b}\n")
    
    x = backward_substitution(A, b, n)

    return x

def main():
    # System of equations
    A = np.array([[1, 2, 1, -1],
                    [3, 2, 4, 4],
                    [4, 4, 3, 4],
                    [2, 0, 1, 5]])
    b = np.array([5, 16, 22, 15])

    # x = np.linalg.solve(A, b)
    # print("np.linalg Solve Solution (x):", x)

    n = len(b)
    x = gauss_elimination(A, b, n)
    print("Gauss Elimination Solution (x):", x) # rewrite the output text

if __name__ == "__main__":
    main()