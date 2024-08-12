# SFWRTECH 4MA3 - Assignment - 5
# Nathan Olah

import numpy as np
from tabulate import tabulate

def f(x):
    return (x - 2)**2 + 4*x - 8

# derivative of f()
def df(x):
    return 2*(x - 2) + 4

def newtons_method_single(f, df, x0, tol=1e-5, max=100):
    results = []
    print(f"Initial Guess x0: {x0}")
    
    xk = x0
    for k in range(max):
        f_xk = f(xk)
        df_xk = df(xk)

        if df_xk == 0:
            print(f"Derivative is zero at iteration {k}. Division by zero error.")
            break

        hk = -f_xk / df_xk
        xk_new = xk + hk
        results.append([k, f"{xk:.6f}", f"{f_xk:.6f}", f"{df_xk:.6f}", f"{hk:.6f}"])

        if abs(hk) < tol:
            break
        
        xk = xk_new
    
    headers = ["k", "xk", "f(xk)", "f'(xk)", "hk"]
    table = tabulate(results, headers, tablefmt="simple")
    print(table)
    
    return xk

def f1(x1, x2):
    return x1 + 2*x2 - 2

def f2(x1, x2):
    return x1**2 + 4*x2**2 - 4

# Jacobian matrix of the system 
def jacobian(x1, x2):
    return np.array([[1, 2], [2*x1, 8*x2]])

def newtons_method_system(f1, f2, jacobian, x0, tol=1e-4, max=100):
    results = []
    xk = np.array(x0, dtype=float)

    for k in range(max):
        # Evaluate functions at current guess
        Fk = np.array([f1(*xk), f2(*xk)], dtype=float)
        
        # Evaluate Jacobian matrix at current guess
        Jk = jacobian(*xk)

        # Solve system (Jk * hk = -Fk)
        hk = np.linalg.solve(Jk, -Fk)
        xk_new = xk + hk
        
        results.append([f"{k}", f"{xk[0]:.6f}", f"{xk[1]:.6f}"])
        
        if np.linalg.norm(hk) < tol:
            break

        xk = xk_new

    headers = ["k", "x1_k", "x2_k"]
    table = tabulate(results, headers, tablefmt="simple")
    print(table)

    return xk

def main():
    # Algorithm 1
    # Initial guess within the range [0, 4]
    x0 = 1.0
    for i in range(4):        
        root = newtons_method_single(f, df, x0)
        print(f"Root found: {root:.6f}\n")
        x0 += 1.0
    print("\n")

    # Algorithm 2
    x0 = [0.5, 0.5]
    root = newtons_method_system(f1, f2, jacobian, x0)
    print(f"Solution: [{root[0]:.6f}, {root[1]:.6f}]")
   

if __name__ == "__main__":
    main()