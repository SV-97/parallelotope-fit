import marimo

__generated_with = "0.10.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    import scipy as scp
    return go, mo, np, scp


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Given a convex set $C$ (in our case $C = [0,1]^N$ the unit cube) its squared distance function $\varphi_C(x) = \frac{1}{2} \min_{y \in C} \lVert x - y \rVert^2_2$ is differentiable and $\nabla \varphi_C(x) = x - P_C(x)$ where $P_C(x)$ denotes the (well defined) projection onto $C$.

        Since $C$ is a product in our case we can project componentwise onto $[0,1]$ which is a simple clamping.

        Given data $x_1, ..., x_N$ we determine the *optimal cube* $A^{-1} C \subseteq \R^3$ by solving

        \[
        \min_{A \in O(3)} \sum_{n=1}^N d_{A^{-1} C}(x_n)^2
        \]

        which by the above is equivalent to

        \[
        \min_{A \in O(3)} \sum_{n=1}^N \varphi_C(Ax_n)^2
        \]

        which we can solve using projected gradient methods. Note that $\nabla_A (A \mapsto \varphi_C (Ax_n))|_A = (Ax_n - P_C(Ax_n)) \otimes x_n \in \R^{3,3}$ where $\otimes$ denotes the outer product of vectors.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The orthogonal projection (w.r.t. Frobenius inner product) onto the set of orthogonal matrices is given by $A \mapsto UV^T$ where $A=USV^T$ is an SVD of $A$.""")
    return


@app.cell
def _(np):
    def proj_unit_cube(points: np.ndarray, out=None) -> np.ndarray:
        return np.clip(points, 0, 1, out=None)


    def make_objective_val_and_grad(points: np.ndarray):
        """Points is d \times N array of cartesian point coordinates
        where d is the dimensions of space and N is the number of point.

        Call these x_1, ..., x_N
        """

        def objective_val_and_grad(matrix: np.ndarray):
            # Call the matrix A
            # This is A x_n
            transformed_points = matrix @ points
            proj = proj_unit_cube(transformed_points)
            # gradient of squared distance function at Ax_n
            dist_grad = transformed_points - proj
            # "gradient" w.r.t matrix; this compute the outer products (Ax_n - P_C(Ax_n)) \otimes x_n = (Ax_n - P_C(Ax_n)) x_n^(T) for each n
            # so this is morally [np.outer(dist_grad[:, i], points[:, i]) for i in range(num_points)]
            mat_grad = np.einsum("ik,jk->ijk", dist_grad, points)
            # to obtain the full gradient we simply sum over all points
            full_grad = np.sum(mat_grad, axis=2)
            # lets also return the objective value since we can easily compute it here
            objective_value = (
                # this should be np.linalg.vecdot(dist_grad, dist_grad, axis=0).sum()) but can't use current numpy due to scipy
                np.einsum("ij,ij->j", dist_grad, dist_grad).sum()
            )
            return objective_value, full_grad

        return objective_val_and_grad


    def make_objective_val(points: np.ndarray):
        """Points is d \times N array of cartesian point coordinates
        where d is the dimensions of space and N is the number of point.

        Call these x_1, ..., x_N
        """

        def objective_val(matrix: np.ndarray):
            # Call the matrix A
            # This is A x_n
            transformed_points = matrix @ points
            proj = proj_unit_cube(transformed_points)
            # gradient of squared distance function at Ax_n
            dist_grad = transformed_points - proj
            objective_value = np.einsum(
                "ij,ij->j", dist_grad, dist_grad
            ).sum()  # np.linalg.vecdot(dist_grad, dist_grad, axis=0).sum()
            return objective_value

        return objective_val
    return make_objective_val, make_objective_val_and_grad, proj_unit_cube


@app.cell
def _(make_objective_val_and_grad, np):
    def fit_cube_naive(
        points: np.ndarray,
        starting_guess: np.ndarray,
        max_iters: int = 10_000,
        abs_tol: float = 1e-3,
        step_size: float = 1e-4,
    ) -> np.ndarray:
        """Fits a cube to the given points by running a projected gradient method"""
        matrix = starting_guess.copy()
        obj_val_and_grad = make_objective_val_and_grad(points)
        # super stupid gradient descent
        for _ in range(max_iters):
            obj_val, obj_grad = obj_val_and_grad(matrix)
            descent_direction = -obj_grad
            grad_norm = np.linalg.norm(
                descent_direction
            )  # frobenious norm of gradient
            if grad_norm < abs_tol:
                break
            matrix += step_size * descent_direction
            # okay maybe not so stupid: lets do a projected gradient method
            # we project the matrix onto the set of orthogonal matrices
            svd = np.linalg.svd(matrix, full_matrices=True)
            matrix = svd.U @ svd.Vh
        return matrix
    return (fit_cube_naive,)


@app.cell
def _(make_objective_val, make_objective_val_and_grad, np):
    def fit_cube_armijo(
        points: np.ndarray,
        starting_guess: np.ndarray,
        max_iters: int = 1_000,
        abs_tol: float = 1e-3,
        initial_step_size: float = 1e-2,
        beta: float = 0.5,  # Step size reduction factor
        armijo_c: float = 1e-4,  # Armijo constant
    ) -> np.ndarray:
        """Fits a cube to the given points by running a projected gradient method with Armijo line search"""
        matrix = starting_guess.copy()
        obj_val_and_grad = make_objective_val_and_grad(points)
        obj_val = make_objective_val(points)

        for _ in range(max_iters):
            cur_obj_val, cur_obj_grad = obj_val_and_grad(matrix)
            descent_direction = -cur_obj_grad
            grad_norm = np.linalg.norm(
                descent_direction
            )  # Frobenius norm of gradient
            if grad_norm < abs_tol:
                break

            # Armijo line search
            step_size = initial_step_size
            while True:
                # Test potential step
                new_matrix = matrix + step_size * descent_direction
                svd = np.linalg.svd(new_matrix, full_matrices=True)
                new_matrix = svd.U @ svd.Vh  # Projection onto orthogonal matrices
                new_value = obj_val(new_matrix)

                # Check Armijo condition
                if new_value <= cur_obj_val + armijo_c * step_size * np.sum(
                    descent_direction * cur_obj_val
                ):
                    break  # Accept the step
                step_size *= beta  # Reduce step size

            # Update matrix after line search
            matrix += step_size * descent_direction

            # Project onto orthogonal matrices
            svd = np.linalg.svd(matrix, full_matrices=True)
            matrix = svd.U @ svd.Vh

        return matrix
    return (fit_cube_armijo,)


@app.cell
def _(make_objective_val_and_grad, np):
    def fit_cube_conjugate_gradient(
        points: np.ndarray,
        starting_guess: np.ndarray,
        max_iters: int = 10_000,
        abs_tol: float = 1e-3,
        step_size: float = 1e-4,
    ) -> np.ndarray:
        """Fits a cube to the given points using projected conjugate gradient method"""
        matrix = starting_guess.copy()
        obj_val_and_grad = make_objective_val_and_grad(points)

        # Initialize conjugate gradient variables
        obj_val, obj_grad = obj_val_and_grad(matrix)
        descent_direction = -obj_grad  # Start with the steepest descent direction
        grad_norm = np.linalg.norm(descent_direction)  # Frobenius norm

        for _ in range(max_iters):
            if grad_norm < abs_tol:
                break

            # Perform a line search along the descent direction
            alpha = (
                step_size  # Use a fixed step size or implement a line search here
            )
            matrix += alpha * descent_direction

            # Project back onto the set of orthogonal matrices
            svd = np.linalg.svd(matrix, full_matrices=True)
            matrix = svd.U @ svd.Vh

            # Compute new gradient
            obj_val_new, obj_grad_new = obj_val_and_grad(matrix)

            # Compute conjugate direction update (Polak-Ribiere formula)
            grad_diff = obj_grad_new - obj_grad
            beta = np.sum(obj_grad_new * grad_diff) / (
                np.linalg.norm(obj_grad) ** 2 + 1e-10
            )
            beta = max(beta, 0)  # Ensure beta >= 0 for stability

            # Update descent direction
            descent_direction = -obj_grad_new + beta * descent_direction
            obj_grad = obj_grad_new  # Update gradient
            grad_norm = np.linalg.norm(obj_grad)  # Update gradient norm

        return matrix
    return (fit_cube_conjugate_gradient,)


@app.cell
def _(make_objective_val, make_objective_val_and_grad, np, scp):
    def fit_cube_conjugate_gradient_with_momentum_linesearch(
        points: np.ndarray,
        starting_guess: np.ndarray,
        max_iters: int = 10_000,
        abs_tol: float = 1e-3,
        step_size: float = 1e-4,
        momentum_beta: float = 0.9,  # Momentum coefficient
    ) -> np.ndarray:
        """Fits a cube to the given points using projected conjugate gradient method with momentum"""
        matrix = starting_guess.copy()
        obj_val_and_grad = make_objective_val_and_grad(points)
        obj_val_fun = make_objective_val(points)

        # Initialize conjugate gradient variables
        obj_val, obj_grad = obj_val_and_grad(matrix)
        descent_direction = -obj_grad  # Start with steepest descent
        grad_norm = np.linalg.norm(descent_direction)  # Frobenius norm

        # Initialize momentum variable (velocity)
        momentum = np.zeros_like(matrix)  # Same shape as the matrix

        for _ in range(max_iters):
            if grad_norm < abs_tol:
                break

            # Update momentum term: Incorporate previous momentum and current gradient direction
            momentum = (
                momentum_beta * momentum + (1 - momentum_beta) * descent_direction
            )

            # Perform a line search along the descent direction (momentum-adjusted direction)
            alph, *_ = scp.optimize.line_search(
                lambda x: obj_val_fun(x.reshape(3, 3)),
                lambda x: obj_val_and_grad(x.reshape(3, 3))[1].flatten(),
                matrix.flatten(),
                momentum.flatten(),
                amax=step_size,
            )
            alpha = alph if alph is not None else step_size
            matrix += alpha * momentum

            # Project back onto the set of orthogonal matrices
            svd = np.linalg.svd(matrix, full_matrices=True)
            matrix = svd.U @ svd.Vh

            # Compute new gradient
            obj_val_new, obj_grad_new = obj_val_and_grad(matrix)

            # Compute conjugate direction update (Polak-Ribiere formula)
            grad_diff = obj_grad_new - obj_grad
            beta = np.sum(obj_grad_new * grad_diff) / (
                np.linalg.norm(obj_grad) ** 2 + 1e-10
            )
            beta = max(beta, 0)  # Ensure beta >= 0 for stability

            # Update descent direction with conjugate gradient formula
            descent_direction = -obj_grad_new + beta * descent_direction
            obj_grad = obj_grad_new  # Update gradient
            grad_norm = np.linalg.norm(obj_grad)  # Update gradient norm

        return matrix
    return (fit_cube_conjugate_gradient_with_momentum_linesearch,)


@app.cell
def _(make_objective_val_and_grad, np):
    def fit_cube_conjugate_gradient_with_momentum(
        points: np.ndarray,
        starting_guess: np.ndarray,
        max_iters: int = 10_000,
        abs_tol: float = 1e-3,
        step_size: float = 1e-4,
        momentum_beta: float = 0.9,  # Momentum coefficient
    ) -> np.ndarray:
        """Fits a cube to the given points using projected conjugate gradient method with momentum"""
        matrix = starting_guess.copy()
        obj_val_and_grad = make_objective_val_and_grad(points)

        # Initialize conjugate gradient variables
        obj_val, obj_grad = obj_val_and_grad(matrix)
        descent_direction = -obj_grad  # Start with steepest descent
        grad_norm = np.linalg.norm(descent_direction)  # Frobenius norm

        # Initialize momentum variable (velocity)
        momentum = np.zeros_like(matrix)  # Same shape as the matrix

        for _ in range(max_iters):
            if grad_norm < abs_tol:
                break

            # Update momentum term: Incorporate previous momentum and current gradient direction
            momentum = (
                momentum_beta * momentum + (1 - momentum_beta) * descent_direction
            )

            # Perform a line search along the descent direction (momentum-adjusted direction)
            alpha = (
                step_size  # Fixed step size, or implement line search if needed
            )
            matrix += alpha * momentum

            if np.any(~np.isfinite(matrix)):
                raise ValueError("Encountered invalid value during iteration")
            svd = np.linalg.svd(matrix, full_matrices=True)
            # Project back onto the set of orthogonal matrices
            # matrix = svd.U @ svd.Vh

            # "project" onto invertible matrices (ish, kind of) by forcing all singular values to be positive and prevent overly large eigenvalues by limiting their maximal value. Note that this imposes limits on the size of the cube
            # matrix = svd.U @ np.diag(np.clip(svd.S, 1e-4, 1e5)) @ svd.Vh
            # print(matrix)
            # print(svd.S)

            # Compute new gradient
            obj_val_new, obj_grad_new = obj_val_and_grad(matrix)

            # Compute conjugate direction update (Polak-Ribiere formula)
            grad_diff = obj_grad_new - obj_grad
            beta = np.sum(obj_grad_new * grad_diff) / (
                np.linalg.norm(obj_grad) ** 2 + 1e-10
            )
            beta = max(beta, 0)  # Ensure beta >= 0 for stability

            # Update descent direction with conjugate gradient formula
            descent_direction = -obj_grad_new + beta * descent_direction
            obj_grad = obj_grad_new  # Update gradient
            grad_norm = np.linalg.norm(obj_grad)  # Update gradient norm

        return matrix
    return (fit_cube_conjugate_gradient_with_momentum,)


@app.cell
def _(np):
    n_dims = 3
    # generate some points in unit cube
    _rng = np.random.default_rng(1)

    # solid cube
    _p = _rng.uniform(0, 1, size=(3, 10_000))

    # cube "shell"
    """
    _p = _rng.uniform(-1, 1, size=(3, 10_000))
    _p = _p / np.linalg.norm(_p, ord=np.inf, axis=0)
    _p = 0.5 * _p + 0.5
    """

    # stretch axes a bit
    # _p2 = np.array([2, 4, 1]).reshape(-1, 1) * _p
    # calc rotation matrix
    _rot_ax = np.array([1, 2, 0.7])
    _rot_ax = _rot_ax / np.linalg.norm(_rot_ax)
    _rot_angle = np.pi * 3 / 4
    _cp_mat = np.array(
        [
            [0, -_rot_ax[2], _rot_ax[1]],
            [_rot_ax[2], 0, -_rot_ax[0]],
            [-_rot_ax[1], _rot_ax[0], 0],
        ]
    )
    rot_mat = (
        np.eye(n_dims)
        + np.sin(_rot_angle) * _cp_mat
        + (1 - np.cos(_rot_angle)) * (_cp_mat @ _cp_mat)
    )
    # rotate points and stretch axes a bit
    transformation_mat = rot_mat  # * np.array([2, 3, 1.5]).reshape(1, -1)
    # _center = np.mean(_p, axis=1)
    _p = _p
    points = transformation_mat @ _p + 0.01 * _rng.normal(size=_p.shape)
    points_unit = _p
    print(points_unit[:, 0])
    print(np.linalg.solve(transformation_mat, points[:, 0]))
    print(np.linalg.inv(transformation_mat) @ points[:, 0])
    print((np.linalg.inv(transformation_mat) @ points)[:, 0])
    print(transformation_mat)
    return n_dims, points, points_unit, rot_mat, transformation_mat


@app.cell
def _(
    fit_cube_conjugate_gradient_with_momentum,
    np,
    points,
    transformation_mat,
):
    _svd = np.linalg.svd(points @ points.T, full_matrices=True)
    _initial_guess = transformation_mat # np.random.default_rng(1).uniform(size=(3, 3))
    (_svd.Vh.T @ _svd.U.T)  # this was just a guess, feel free to change it
    #
    # should be nonsingular, which this randomized one is with probability 1
    """
    fit_trans_mat = fit_cube_conjugate_gradient_with_momentum(
        points.copy(),
        _initial_guess.copy(),
        max_iters=50_000,
        step_size=1e-6,
        abs_tol=1e-3,
        momentum_beta=0.0,
    )
    """
    fit_trans_mat = fit_cube_conjugate_gradient_with_momentum(
        points.copy(),
        _initial_guess.copy(),
        max_iters=50_000,
        step_size=1e-4,
        abs_tol=1e-3,
        # momentum_beta=0.9,
    )
    # _fit_trans_mat = fit_cube_conjugate_gradient_with_momentum_linesearch(
    #    points.copy(),
    #    _initial_guess.copy(),
    #    max_iters=500,
    #    step_size=5e-1,
    #    abs_tol=1e-6,
    # )
    np.linalg.norm(fit_trans_mat - np.linalg.inv(transformation_mat), ord=np.inf)
    return (fit_trans_mat,)


@app.cell
def _(fit_trans_mat, go, np, points, transformation_mat):
    print(transformation_mat)
    _inv_trans_mat = np.linalg.inv(transformation_mat)

    _fig = go.Figure()
    _fig.add_trace(
        go.Scatter3d(
            x=(_inv_trans_mat @ points)[0],
            y=(_inv_trans_mat @ points)[1],
            z=(_inv_trans_mat @ points)[2],
            mode="markers",
            marker=dict(
                size=5,  # Size of markers
                opacity=0.8,  # Opacity of markers
            ),
            name="Ground Truth",
        )
    )
    """
    _fig.add_trace(
        go.Scatter3d(
            x=points_unit[0, :],
            y=points_unit[1, :],
            z=points_unit[2, :],
            mode="markers",
            marker=dict(
                size=5,  # Size of markers
                opacity=0.8,  # Opacity of markers
            ),
            name="Ground Truth 2",
        )
    )
    """

    _fig.add_trace(
        go.Scatter3d(
            x=(fit_trans_mat @ points)[0],
            y=(fit_trans_mat @ points)[1],
            z=(fit_trans_mat @ points)[2],
            mode="markers",
            marker=dict(
                size=5,  # Size of markers
                opacity=0.8,  # Opacity of markers
            ),
            name="fitted",
        )
    )


    # Customize the layout
    _fig.update_layout(
        scene=dict(
            xaxis=dict(
                title="X",
            ),
            yaxis=dict(
                title="Y",
            ),
            zaxis=dict(
                title="Z",
            ),
        ),
        title="Comparing fitted and true cube",
        margin=dict(l=0, r=0, b=0, t=40),  # Margins around the plot
    )

    _fig
    return


@app.cell
def _(center, points):
    from sklearn.decomposition import PCA

    _pca = PCA(n_components=1)
    _centered_points = points - center
    comps = _pca.fit(_centered_points.T).components_.T
    comps
    return PCA, comps


@app.cell
def _(np, points):
    # compute a PCA of the data
    center = np.mean(points, axis=1).reshape(3, -1)
    _centered_points = points - center
    vals, vecs = np.linalg.eigh(_centered_points @ _centered_points.T)
    print(vals)
    print(vecs)
    return center, vals, vecs


@app.cell
def _(mo):
    mo.md(r"""Ignore the "vertex lines" in the plot, I just tried some stuff here""")
    return


@app.cell
def _(center, fit_trans_mat, go, np, points, points_unit, vertices):
    # Create the 3D scatter plot
    _fig = go.Figure()
    _fig.add_trace(
        go.Scatter3d(
            x=points[0],
            y=points[1],
            z=points[2],
            mode="markers",
            marker=dict(
                size=5,  # Size of markers
                opacity=0.8,  # Opacity of markers
            ),
            name="Ground Truth",
        )
    )

    _inv_fitted = np.linalg.inv(fit_trans_mat)
    _fig.add_trace(
        go.Scatter3d(
            x=(_inv_fitted @ points_unit)[0],
            y=(_inv_fitted @ points_unit)[1],
            z=(_inv_fitted @ points_unit)[2],
            mode="markers",
            marker=dict(
                size=5,  # Size of markers
                opacity=0.8,  # Opacity of markers
            ),
            name="Fitted",
        )
    )

    _t = np.linspace(-10, 10, num=50)
    # _line = (1 - _t) * np.zeros_like(d) + _t * d
    """
    _line = center + _t * d
    _fig.add_trace(
        go.Scatter3d(
            x=_line[0], y=_line[1], z=_line[2], mode="lines", name="Iterthingy"
        )
    )
    """

    for i, vertex in enumerate(vertices):
        _line = center + _t * vertex
        _fig.add_trace(
            go.Scatter3d(
                x=_line[0],
                y=_line[1],
                z=_line[2],
                mode="lines",
                name=f"Vertex #{i}",
            )
        )

    # Add a cone trace for the vectors
    """
    _fig.add_trace(
    go.Cone(
            x=np.repeat(center[0], 3),  # Starting x-coordinates
            y=np.repeat(center[1], 3),  # Starting y-coordinates
            z=np.repeat(center[2], 3),  # Starting z-coordinates
            u=vecs[0],  # x-components of vectors
            v=vecs[1],  # y-components of vectors
            w=vecs[2],  # z-components of vectors
            sizeref=1,
            sizemode="absolute",  # Scale arrow size proportionally
            anchor="tail",
        )
    )
    """

    # Customize the layout
    _fig.update_layout(
        scene=dict(
            xaxis=dict(title="X", range=[-5, 5]),
            yaxis=dict(title="Y", range=[-5, 5]),
            zaxis=dict(title="Z", range=[-5, 5]),
        ),
        title="Point cloud",
        margin=dict(l=0, r=0, b=0, t=40),  # Margins around the plot
    )

    _fig
    return i, vertex


@app.cell
def _(mo):
    mo.md(r"""Just trying some other things below""")
    return


@app.cell
def _(center, np, points):
    d = points[:, 0]
    _n_iters = 100
    for _ in range(_n_iters):
        _dots = np.einsum("i,ij", d, points)
        _s = np.sum(points[:, _dots > 0], axis=1)
        d = _s / np.linalg.norm(_s)
    d = d.reshape(3, -1) - center

    """
    # deflate: remove component of points that's parallel to the vertex we just determined
    _p = points - np.einsum("ij,i->j", points, d[:, 0]) * d
    d2 = np.random.default_rng(0).uniform(-1, 1, size=3)
    d2 = d2 / np.linalg.norm(d2)
    for _ in range(_n_iters):
        _dots = np.einsum("i,ij", d2, _p)
        _s = np.sum(points[:, _dots > 0], axis=1)
        d2 = _s / np.linalg.norm(_s)
    d2 = d2.reshape(3, -1)

    _p = _p - np.einsum("ij,i->j", _p, d2[:, 0]) * d2
    d3 = np.random.default_rng(0).uniform(-1, 1, size=3)
    d3 = d3 / np.linalg.norm(d3)
    for _ in range(_n_iters):
        _dots = np.einsum("i,ij", d3, _p)
        _s = np.sum(points[:, _dots > 0], axis=1)
        d3 = _s / np.linalg.norm(_s)
    d3 = d3.reshape(3, -1)
    """
    d2 = np.cross(np.random.default_rng(0).uniform(-1, 1, size=3), d[:, 0])
    d3 = np.cross(d2, d[:, 0])
    vertices = [d, d2.reshape(3, -1), d3.reshape(3, -1)]
    return d, d2, d3, vertices


@app.cell
def _(vertices):
    vertices
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
