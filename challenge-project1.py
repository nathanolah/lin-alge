# SFWRTECH 4MA3 - Challenge Project - 1
# Nathan Olah

import numpy as np
from tabulate import tabulate

def generatorHb(n):
    H = np.zeros((n, n))
    x = np.ones(n)

    for i in range(n):
        for j in range(n):
            H[i, j] = 1 / (i + j + 1)

    # Matrix Vector multiplication
    b = np.dot(H, x)

    return H, b, x

def forwardSubstitution(L, b):
    # Ly = b
    n = len(b)
    y =  np.zeros(n)

    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= L[i, j] * y[j]
        y[i] / L[i, i]
    
    return y

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

def LU_decomposition(A):
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    
    for i in range(n):
        L[i, i] = 1
        for j in range(i, n):
            U[i, j] = A[i, j] - sum(L[i, k] * U[k, j] for k in range(i))
        for j in range(i + 1, n):
            L[j, i] = (A[j, i] - sum(L[j, k] * U[k, i] for k in range(i))) / U[i, i]
    
    return L, U

def gaussElimination(H, b):
    L, U = LU_decomposition(H)
    y = forwardSubstitution(L, b)
    x_hat = backwardSubstitution(U, y)

    return x_hat

def format_array(arr):
    # Convert numpy array to a list
    if isinstance(arr, np.ndarray):
        arr = arr.tolist()

    if len(arr) > 3:
        return f"[ {arr[0]}, {arr[1]}, {arr[2]}, ... {len(arr) - 3} more items ]"
    else:
        return str(arr)

def main():
    
    results = []
    output_vectors = []
    n = 2

    while True:
        H, b, x = generatorHb(n)
        H_copy = H.copy()
        x_hat = gaussElimination(H, b)

        # infinity norm of the residual
        r = np.linalg.norm(b - np.dot(H_copy, x_hat), np.inf)

        # delta x
        delta_x = x_hat - x
        delta_x_inf_norm = np.linalg.norm(delta_x, np.inf)

        # Condition Number of H
        cond_H = np.linalg.cond(H_copy)

        # Collect results
        results.append([
            n,
            format_array(x_hat),
            format_array(delta_x),
            np.linalg.norm(x_hat),
            np.linalg.norm(delta_x),
            np.linalg.norm(delta_x) / np.linalg.norm(x), # relative error 
            f"{(np.linalg.norm(delta_x) / np.linalg.norm(x)) * 100:.2e}%",
            r,
            format_array(b),
            cond_H
        ])

        output_vectors.append([n, format_array(x_hat), format_array(delta_x), r])

        if delta_x_inf_norm >= 1:
            error_msg = f"Error is greater than or equal to 100% at n={n}"
            print(error_msg)
            break

        n += 1
    
    # For each n, print the following vectors: 𝒙2, 𝚫𝒙, r
    output_headers = ["n", "x̂", "Δx", "r"]
    print(tabulate(output_vectors, headers=output_headers, tablefmt="fancy_outline"))

    # Save all results to a text file
    headers = ["n", "x̂", "Δx", "ǁxǁ_2", "ǁΔxǁ_2", "xRE", "xRE%", "ǁrǁ_∞", "b", "cond(H)"]
    with open("results.txt", "w", encoding="utf-8") as f:
        if error_msg:
            f.write(error_msg + "\n\n")
        f.write(tabulate(results, headers=headers, tablefmt="fancy_outline", numalign="right", colalign="right"))

if __name__ == "__main__":
    main()