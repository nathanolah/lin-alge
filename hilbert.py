import numpy as np

def generatorHb(n):
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            H[i, j] = 1 / (i + j + 1)
    x = np.ones(n)
    b = H @ x
    return H, b, x

def gaussElimination(A):
    n = A.shape[0]
    for i in range(n):
        # Pivoting
        max_row = i + np.argmax(np.abs(A[i:, i]))
        A[[i, max_row]] = A[[max_row, i]]
        for j in range(i+1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
    return A

def forwardSubstitution(L, b):
    n = b.size
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]
    return y

def backwardSubstitution(U, y):
    n = y.size
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - U[i, i+1:] @ x[i+1:]) / U[i, i]
    return x

def solveHilbert(n):
    H, b, x = generatorHb(n)
    U = gaussElimination(H.copy())
    y = forwardSubstitution(U, b)
    x_hat = backwardSubstitution(U, y)
    
    r = b - H @ x_hat
    dx = x_hat - x
    residual_norm = np.linalg.norm(r, np.inf)
    error_norm = np.linalg.norm(dx, np.inf)
    cond_H = np.linalg.cond(H, np.inf)
    
    return x_hat, dx, r, residual_norm, error_norm, cond_H

# Testing for different values of n
results = []
for n in range(2, 13):
    x_hat, dx, r, residual_norm, error_norm, cond_H = solveHilbert(n)
    results.append((n, x_hat, dx, r, residual_norm, error_norm, cond_H))

# Printing the results in a table format
print("n  | x_hat           | dx             | r              | Residual Norm | Error Norm  | Condition Number")
print("---|-----------------|----------------|----------------|---------------|-------------|-----------------")
for res in results:
    n, x_hat, dx, r, residual_norm, error_norm, cond_H = res
    print(f"{n:2} | {x_hat} | {dx} | {r} | {residual_norm:.2e}       | {error_norm:.2e}     | {cond_H:.2e}")

# Determine the maximum n where error is less than 100%
max_n = max(n for n, x_hat, dx, r, residual_norm, error_norm, cond_H in results if error_norm < 1)
print(f"The largest n before the error reaches 100% is: {max_n}")