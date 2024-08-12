# SFWRTECH 4MA3 - Assignment - 6
# Nathan Olah

import numpy as np
from tabulate import tabulate

def f1(x):
    return x**2 - 2*x + 1

def df1(x):
    return 2*x - 2

def ddf1(x):
    return 2

# Algorithm 1
def newton_method_one_dimensional(f, df, ddf, x0, tol=1e-5, max=100):
    results = []
    x = x0

    for k in range(max):
        fx = f(x)
        dfx = df(x)
        ddfx = ddf(x)

        results.append([k, f"{x:.6f}", f"{fx:.6f}", f"{dfx:.6f}", f"{ddfx:.6f}"])
        x_new = x - dfx / ddfx

        if abs(x_new - x) < tol:
            break
        x = x_new

    headers = ["k", "x", "f(x)", "f'(x)", "f''(x)"]
    table = tabulate(results, headers, tablefmt="simple")
    print(table + "\n")

    return x

def f2(x, y):
    return 0.5*x**2 + 2.5*y**2

def gradient_f2(x, y):
    return np.array([x, 5*y])

def hessian_f2(x, y):
    return np.array([[1, 0], [0, 5]])

# Algorithm 2
def newton_method_multi_dimensional(f, gradient_f, hessian_f, x0, tol=1e-5, max=100):
    results = []
    x = np.array(x0, dtype=float)

    for k in range(max):
        fx = f(x[0], x[1])
        gradient_fx = np.array(gradient_f(x[0], x[1]))
        hessian_fx = np.array(hessian_f(x[0], x[1]))

        # Format values
        gradient_fx_str = f"[{gradient_fx[0]:.5f}, {gradient_fx[1]:.5f}]"
        xy_str = f"({x[0]:.5f}, {x[1]:.5f})"
        results.append([k, f"{xy_str}", f"{fx:.6f}", f"{gradient_fx_str}"])

        hessian_inv = np.linalg.inv(hessian_fx)
        x_new = x - hessian_inv.dot(gradient_fx)
    
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    
    headers = ["k", "(x, y)", "f(x, y)", "∇f(x, y)"]
    table = tabulate(results, headers, tablefmt="simple")
    print(table)
    
    return x

def main():
    # Algorithm 1
    # Starting point and interval [-0.5, 1.5]
    print("Algorithm 1\n")
    end = 1.5
    x0_1 = -0.5
    while x0_1 <= end:
        print(f"x0: {x0_1}")
        newton_method_one_dimensional(f1, df1, ddf1, x0_1)
        x0_1 += 0.5

    # Algorithm 2
    # Starting point
    print("Algorithm 2\n")
    x0_2 = [-1, 1]
    newton_method_multi_dimensional(f2, gradient_f2, hessian_f2, x0_2)

if __name__ == "__main__":
    main()