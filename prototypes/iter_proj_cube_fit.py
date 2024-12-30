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


@app.cell
def _(np):
    n_dims = 3
    # generate some points in unit cube
    _rng = np.random.default_rng(1)

    # solid cube
    _p = _rng.uniform(0, 1, size=(3, 10_000))
    _p -= 0.5

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
    # rotate points and skew axes a bit
    transformation_mat = rot_mat @ np.array(
        [[2, 1, 2.85], [0, 3, 0.25], [0, 0, 1.5]]
    )
    # _center = np.mean(_p, axis=1)
    _p = _p
    points = transformation_mat @ _p  # + 0.01 * _rng.normal(size=_p.shape)
    points_unit = _p
    print(points_unit[:, 0])
    print(np.linalg.solve(transformation_mat, points[:, 0]))
    print(np.linalg.inv(transformation_mat) @ points[:, 0])
    print((np.linalg.inv(transformation_mat) @ points)[:, 0])
    print(transformation_mat)
    return n_dims, points, points_unit, rot_mat, transformation_mat


@app.cell
def _(np):
    def fibonacci_sphere(samples=1000):
        """Generate points that are approximately uniformly distributed on sphere"""
        # Generate indices array [0, 1, ..., samples-1]
        i = np.arange(samples)
        phi = np.pi * (np.sqrt(5.0) - 1.0)
        y = np.linspace(1, -1, samples)
        radius = np.sqrt(1 - y * y)
        theta = phi * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        return np.row_stack((x, y, z))
    return (fibonacci_sphere,)


@app.cell
def _(mo):
    mo.md(r"""## Step 1: Determine first diagonal""")
    return


@app.cell
def _(n_dims, np, points):
    center = np.mean(points, axis=1)
    centered_points = points - center.reshape(n_dims, -1)
    return center, centered_points


@app.cell
def _(centered_points, np, points):
    norms = np.linalg.norm(centered_points, axis=0)
    best_dir = points[:, np.argmax(norms)]
    best_dir_unit = best_dir / np.linalg.norm(best_dir)
    return best_dir, best_dir_unit, norms


@app.cell
def _(mo):
    mo.md(r"""## Step 2: Project sample onto plane orthogonally to first diagonal""")
    return


@app.cell
def _(best_dir_unit, centered_points, np):
    _par1 = np.einsum(
        "i,ij", best_dir_unit, centered_points
    ) * best_dir_unit.reshape(3, -1)
    projected_points_1 = centered_points - _par1
    return (projected_points_1,)


@app.cell
def _(mo):
    mo.md(r"""## Step 3: Find second diagonal""")
    return


@app.cell
def _(np, points, projected_points_1):
    proj_norms_1 = np.linalg.norm(
        projected_points_1, axis=0
    )  # we could probably also compute this from the dot products used in the orthogonal projection but eh
    idx_max_2 = np.argmax(proj_norms_1)
    best_dir_2 = points[:, idx_max_2]
    best_dir_2_unit = best_dir_2 / np.linalg.norm(best_dir_2)
    best_dir_2_unit_subspace = projected_points_1[:, idx_max_2] / np.linalg.norm(
        projected_points_1[:, idx_max_2]
    )
    return (
        best_dir_2,
        best_dir_2_unit,
        best_dir_2_unit_subspace,
        idx_max_2,
        proj_norms_1,
    )


@app.cell
def _(mo):
    mo.md(r"""## Step 4: Repeat projection etc. to determine third diagonal""")
    return


@app.cell
def _(best_dir_2_unit_subspace, np, points, projected_points_1):
    _par2 = np.einsum(
        "i,ij", best_dir_2_unit_subspace, projected_points_1
    ) * best_dir_2_unit_subspace.reshape(3, -1)
    projected_points_2 = projected_points_1 - _par2

    proj_norms_2 = np.linalg.norm(
        projected_points_2, axis=0
    )  # we could probably also compute this from the dot products used in the orthogonal projection but eh
    idx_max_3 = np.argmax(proj_norms_2)
    best_dir_3 = points[:, idx_max_3]
    best_dir_3_unit = best_dir_3 / np.linalg.norm(best_dir_3)
    best_dir_3_unit_subspace = projected_points_2[:, idx_max_3] / np.linalg.norm(
        projected_points_2[:, idx_max_3]
    )
    return (
        best_dir_3,
        best_dir_3_unit,
        best_dir_3_unit_subspace,
        idx_max_3,
        proj_norms_2,
        projected_points_2,
    )


@app.cell
def _(best_dir, best_dir_2, best_dir_3, center):
    # we fix best_dir as reference vertex. We now consider the vertices we determined that are not on the same diagonal as the reference
    reference_vertex = best_dir
    determined_vertices = [
        best_dir_2,
        center - best_dir_2,
        best_dir_3,
        center - best_dir_3,
    ]
    # each such vertex connects with out reference by a line on the surface of the cube
    some_edge_vecs = [vert - reference_vertex for vert in determined_vertices]
    return determined_vertices, reference_vertex, some_edge_vecs


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Step 5: Repeat the whole thing one last time for the final diagonal
        (I think this can also be directly computed from the other diagonals, I just didn't wanna think about how exactly)
        """
    )
    return


@app.cell
def _(best_dir, best_dir_2, best_dir_3, np):
    """
    _par3 = np.einsum(
        "i,ij", best_dir_3_unit_subspace, projected_points_2
    ) * best_dir_3_unit_subspace.reshape(3, -1)
    projected_points_3 = projected_points_2 - _par3

    proj_norms_3 = np.linalg.norm(
        projected_points_3, axis=0
    )  # we could probably also compute this from the dot products used in the orthogonal projection but eh
    idx_max_4 = np.argmax(proj_norms_3)
    best_dir_4 = points[:, idx_max_4]
    """

    # best_dir_4 = best_dir_3 + best_dir_2 - best_dir
    best_dir_4 = best_dir - best_dir_2 - best_dir_3
    best_dir_4_unit = best_dir_4 / np.linalg.norm(best_dir_4)
    return best_dir_4, best_dir_4_unit


@app.cell
def _(best_dir, best_dir_2, best_dir_3, best_dir_4, np):
    edge_vectors = [
        best_dir_2 - best_dir,
        best_dir_3 - best_dir,
        best_dir_4 - best_dir,
    ]
    # this transformation assumes that the best_dir vertex is at the origin
    fitted_transformation_mat = np.column_stack(edge_vectors)
    return edge_vectors, fitted_transformation_mat


@app.cell
def _(centered_points, fibonacci_sphere, np):
    # This cell could be used for another approach: it turns the data into a sort of wireframe that could then potentially be more easily processed. However it requires computing quite a few dot products.
    _num_points_on_sphere = 500
    d = fibonacci_sphere(_num_points_on_sphere)
    dots = np.einsum("ij,ik", d, centered_points)
    argmax_dots = np.argmax(
        dots, axis=1
    )  # find the sample that "points most towards the point on the sphere"
    max_dots = dots[np.arange(_num_points_on_sphere), argmax_dots]
    best_dirs = centered_points[:, argmax_dots]

    # best_dir = best_dirs[:, np.argmax(max_dots)]
    return argmax_dots, best_dirs, d, dots, max_dots


@app.cell(hide_code=True)
def _(
    best_dir,
    best_dir_2,
    best_dir_3,
    best_dir_4,
    best_dirs,
    center,
    d,
    go,
    max_dots,
    np,
    points,
    projected_points_1,
    projected_points_2,
    reference_vertex,
    some_edge_vecs,
):
    # Create the 3D scatter plot
    _points = points

    _fig = go.Figure()
    _fig.add_trace(
        go.Scatter3d(
            x=_points[0],
            y=_points[1],
            z=_points[2],
            mode="markers",
            marker=dict(
                size=5,  # Size of markers
                opacity=0.6,  # Opacity of markers
            ),
            name="Sample",
        )
    )

    _sphere_to_plot = np.linalg.norm(best_dir) * d

    """
    _fig.add_trace(
        go.Scatter3d(
            x=_sphere_to_plot[0],
            y=_sphere_to_plot[1],
            z=_sphere_to_plot[2],
            mode="markers",
            marker=dict(
                size=5,  # Size of markers
                opacity=0.8,  # Opacity of markers
            ),
            name="Sphere",
        )
    )
    """

    _fig.add_trace(
        go.Scatter3d(
            x=best_dirs[0],
            y=best_dirs[1],
            z=best_dirs[2],
            mode="markers",
            marker=dict(
                size=5,  # Size of markers
                opacity=0.8,  # Opacity of markers
            ),
            name="'Wireframe'",
            customdata=max_dots,  # Attach additional data
            hovertemplate="X: %{x}<br>Y: %{y}<br>Z: %{z}<br>Max Dot: %{customdata}<extra></extra>",
            visible="legendonly",
        )
    )

    _fig.add_trace(
        go.Scatter3d(
            x=[best_dir[0], (center - best_dir)[0]],
            y=[best_dir[1], (center - best_dir)[1]],
            z=[best_dir[2], (center - best_dir)[2]],
            mode="markers",
            marker=dict(
                size=5,  # Size of markers
                opacity=0.8,  # Opacity of markers
            ),
            name="Best dir",
        )
    )

    _fig.add_trace(
        go.Scatter3d(
            x=[best_dir_2[0], (center - best_dir_2)[0]],
            y=[best_dir_2[1], (center - best_dir_2)[1]],
            z=[best_dir_2[2], (center - best_dir_2)[2]],
            mode="markers",
            marker=dict(
                size=5,  # Size of markers
                opacity=0.8,  # Opacity of markers
            ),
            name="Best dir 2",
        )
    )

    _fig.add_trace(
        go.Scatter3d(
            x=[best_dir_3[0], (center - best_dir_3)[0]],
            y=[best_dir_3[1], (center - best_dir_3)[1]],
            z=[best_dir_3[2], (center - best_dir_3)[2]],
            mode="markers",
            marker=dict(
                size=5,  # Size of markers
                opacity=0.8,  # Opacity of markers
            ),
            name="Best dir 3",
        )
    )

    _fig.add_trace(
        go.Scatter3d(
            x=[best_dir_4[0], (center - best_dir_4)[0]],
            y=[best_dir_4[1], (center - best_dir_4)[1]],
            z=[best_dir_4[2], (center - best_dir_4)[2]],
            mode="markers",
            marker=dict(
                size=5,  # Size of markers
                opacity=0.8,  # Opacity of markers
            ),
            name="Best dir 4",
        )
    )

    _fig.add_trace(
        go.Scatter3d(
            x=projected_points_1[0],
            y=projected_points_1[1],
            z=projected_points_1[2],
            mode="markers",
            marker=dict(
                size=2.5,  # Size of markers
                opacity=0.3,  # Opacity of markers
            ),
            name="Once projected Sample",
        )
    )

    _fig.add_trace(
        go.Scatter3d(
            x=projected_points_2[0],
            y=projected_points_2[1],
            z=projected_points_2[2],
            mode="markers",
            marker=dict(
                size=1.25,  # Size of markers
                opacity=0.3,  # Opacity of markers
            ),
            name="Twice projected Sample",
        )
    )

    for _v in some_edge_vecs:
        _fig.add_trace(
            go.Scatter3d(
                x=[reference_vertex[0], (reference_vertex + _v)[0]],
                y=[reference_vertex[1], (reference_vertex + _v)[1]],
                z=[reference_vertex[2], (reference_vertex + _v)[2]],
                mode="markers+lines",
                marker=dict(
                    size=8,  # Size of markers
                    opacity=0.3,  # Opacity of markers
                ),
                line=dict(
                    width=4,
                ),
            )
        )

    for _v in some_edge_vecs:
        _fig.add_trace(
            go.Scatter3d(
                x=[
                    (center - reference_vertex)[0],
                    (center - reference_vertex - _v)[0],
                ],
                y=[
                    (center - reference_vertex)[1],
                    (center - reference_vertex - _v)[1],
                ],
                z=[
                    (center - reference_vertex)[2],
                    (center - reference_vertex - _v)[2],
                ],
                mode="markers+lines",
                marker=dict(
                    size=8,  # Size of markers
                    opacity=0.3,  # Opacity of markers
                ),
                line=dict(
                    width=4,
                ),
            )
        )
    """
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
    """
    """
    _t = np.linspace(-10, 10, num=50)
    # _line = (1 - _t) * np.zeros_like(d) + _t * d
    for _d in princip_comps.T:
        _line = center.reshape(3, -1) + _t.reshape(1, -1) * _d.reshape(3, -1)
        _fig.add_trace(
            go.Scatter3d(
                x=_line[0],
                y=_line[1],
                z=_line[2],
                mode="lines",
                name="principal direction",
            )
        )
    """

    # Customize the layout
    """
    _fig.update_layout(
        scene=dict(
            xaxis=dict(title="X", range=[-5, 5]),
            yaxis=dict(title="Y", range=[-5, 5]),
            zaxis=dict(title="Z", range=[-5, 5]),
        ),
        title="Point cloud",
        margin=dict(l=0, r=0, b=0, t=40),  # Margins around the plot
    )
    """

    _fig
    return


@app.cell
def _(best_dir, fitted_transformation_mat, go, np, points):
    # Create the 3D scatter plot
    _points = points

    _fig = go.Figure()
    _fig.add_trace(
        go.Scatter3d(
            x=_points[0],
            y=_points[1],
            z=_points[2],
            mode="markers",
            marker=dict(
                size=5,  # Size of markers
                opacity=0.8,  # Opacity of markers
            ),
            name="Sample",
        )
    )

    _tets = np.array(
        [
            (0, 1, 3),
            (0, 1, 5),
            (0, 2, 3),
            (0, 2, 6),
            (0, 4, 6),
            (0, 4, 5),
            (1, 3, 7),
            (1, 5, 7),
            (2, 3, 7),
            (2, 6, 7),
            (4, 5, 7),
            (4, 6, 7),
        ],
        dtype=int,
    ).T

    _unit_cube_verts = []
    # _prev_vert = np.array([0,0,0])
    for _i in range(0, 0b111 + 1):
        _unit_cube_verts.append(np.array([1 & _i, (2 & _i) >> 1, (4 & _i) >> 2]))
    _unit_cube_verts = np.array(_unit_cube_verts).T
    _cube_verts = fitted_transformation_mat @ _unit_cube_verts + best_dir.reshape(
        3, -1
    )


    """
    unit_interval = np.linspace(0, 1, num=20)
    _X, _Y, _Z = np.meshgrid(unit_interval, unit_interval, unit_interval, indexing="ij")
    unit_cube = np.vstack([_X.ravel(), _Y.ravel(), _Z.ravel()])
    print(unit_cube.shape)
    cube = fitted_transformation_mat @ unit_cube + best_dir.reshape(3,-1)
    """
    _fig.add_trace(
        go.Mesh3d(
            x=_cube_verts[0],
            y=_cube_verts[1],
            z=_cube_verts[2],
            i=_tets[0],
            j=_tets[1],
            k=_tets[2],
            name="Fitted Cube",
            opacity=0.5,
        )
    )

    _fig
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
            # svd = np.linalg.svd(matrix, full_matrices=True)
            # matrix = svd.U @ svd.Vh
        return matrix
    return (
        fit_cube_naive,
        make_objective_val,
        make_objective_val_and_grad,
        proj_unit_cube,
    )


@app.cell
def _(best_dir, fit_cube_naive, fitted_transformation_mat, points):
    _initial_guess = fitted_transformation_mat
    fitted_transformation_mat2 = fit_cube_naive(
        (points - best_dir.reshape(3, -1)).copy(),
        _initial_guess.copy(),
        max_iters=50_000,
        step_size=1e-5,
        abs_tol=1e-6,
    )
    return (fitted_transformation_mat2,)


@app.cell
def _(best_dir, fitted_transformation_mat2, go, np, points):
    # Create the 3D scatter plot
    _points = points

    _fig = go.Figure()
    _fig.add_trace(
        go.Scatter3d(
            x=_points[0],
            y=_points[1],
            z=_points[2],
            mode="markers",
            marker=dict(
                size=5,  # Size of markers
                opacity=0.8,  # Opacity of markers
            ),
            name="Sample",
        )
    )

    _tets = np.array(
        [
            (0, 1, 3),
            (0, 1, 5),
            (0, 2, 3),
            (0, 2, 6),
            (0, 4, 6),
            (0, 4, 5),
            (1, 3, 7),
            (1, 5, 7),
            (2, 3, 7),
            (2, 6, 7),
            (4, 5, 7),
            (4, 6, 7),
        ],
        dtype=int,
    ).T

    _unit_cube_verts = []
    # _prev_vert = np.array([0,0,0])
    for _i in range(0, 0b111 + 1):
        _unit_cube_verts.append(np.array([1 & _i, (2 & _i) >> 1, (4 & _i) >> 2]))
    _unit_cube_verts = np.array(_unit_cube_verts).T
    _cube_verts = fitted_transformation_mat2 @ _unit_cube_verts + best_dir.reshape(
        3, -1
    )


    """
    unit_interval = np.linspace(0, 1, num=20)
    _X, _Y, _Z = np.meshgrid(unit_interval, unit_interval, unit_interval, indexing="ij")
    unit_cube = np.vstack([_X.ravel(), _Y.ravel(), _Z.ravel()])
    print(unit_cube.shape)
    cube = fitted_transformation_mat @ unit_cube + best_dir.reshape(3,-1)
    """
    _fig.add_trace(
        go.Mesh3d(
            x=_cube_verts[0],
            y=_cube_verts[1],
            z=_cube_verts[2],
            i=_tets[0],
            j=_tets[1],
            k=_tets[2],
            name="Fitted Cube",
            opacity=0.5,
        )
    )

    _fig
    return


@app.cell
def _():
    bin(3)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
