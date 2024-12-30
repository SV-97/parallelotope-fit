import marimo

__generated_with = "0.10.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pymanopt
    from pymanopt.manifolds import Stiefel
    from pymanopt.optimizers.steepest_descent import SteepestDescent
    from pymanopt import Problem
    import numpy as np
    import autograd.numpy as anp

    # Example: Define the Stiefel manifold
    manifold = Stiefel(3, 3)


    # Define the cost function to minimize
    @pymanopt.function.numpy(manifold)
    def cost(A):
        # Example: Minimize the Frobenius norm of Ax - b
        x = np.random.randn(3)  # Fixed vector in R^p
        b = np.random.randn(3)  # Target vector in R^n
        residual = A @ x - b
        return np.linalg.norm(residual) ** 2


    # Define the problem on the manifold
    problem = Problem(manifold=manifold, cost=cost)

    # Solve the problem using Riemannian gradient descent
    optimizer = SteepestDescent()
    A_optimized = optimizer.run(problem)

    # Output results
    print("Optimized matrix A:")
    print(A_optimized)
    return (
        A_optimized,
        Problem,
        SteepestDescent,
        Stiefel,
        anp,
        cost,
        manifold,
        mo,
        np,
        optimizer,
        problem,
        pymanopt,
    )


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
