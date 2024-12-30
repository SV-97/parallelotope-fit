import marimo

__generated_with = "0.10.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    import scipy as scp
    import itertools as itt
    return go, itt, mo, np, scp


@app.cell
def _(np):
    n_dims = 3
    # generate some points in unit cube for testing
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
        [[2, 1, 2.85], [0, 3, 0.25], [10, 0, 1.5]]
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


    def orthogonal_vector(v):
        """
        Returns a vector that is orthogonal to the given nonzero 3D vector vv.

        This function is (necessarily) discontinuous.
        """
        if np.allclose(v, 0):
            raise ValueError("Input vector must be nonzero.")

        # Choose an arbitrary vector to cross with
        if abs(v[0]) > abs(v[1]):
            temp = np.array([0, 1, 0])  # Prefer y-axis if x-component dominates
        else:
            temp = np.array([1, 0, 0])  # Prefer x-axis otherwise

        # Compute the orthogonal vector using the cross product
        orthogonal = np.cross(v, temp)

        # Normalize the result to ensure it's a unit vector
        return orthogonal / np.linalg.norm(orthogonal)
    return fibonacci_sphere, orthogonal_vector


@app.cell
def _(np):
    def scale_along_vector(v):
        """
        Returns a linear transformation matrix that "normalizes the length of vv"
        without affecting orthogonal directions.

        So it maps v to v/||v|| and span{v}^\bot is invariant under the map.
        """
        if np.allclose(v, 0):
            raise ValueError("Input vector must be nonzero.")

        v_norm = np.linalg.norm(v)
        v_unit = v / v_norm
        projection_matrix = np.outer(v_unit, v_unit)

        return np.eye(3) + (1 / v_norm - 1) * projection_matrix


    def proj_onto_orthogonal_complement(v):
        """
        Returns a linear transformation matrix that projects onto the orthogonal complement of a given vector vv.

        So it maps v to 0 and span{v}^\bot is invariant under the map.
        """
        v_unit = v / np.linalg.norm(v)
        projection_matrix = np.outer(v_unit, v_unit)
        return np.eye(3) - projection_matrix
    return proj_onto_orthogonal_complement, scale_along_vector


@app.cell
def _(np):
    def mat_from_eigenpairs(*eigenpairs: tuple[float, np.ndarray]):
        """Computes a matrix A such that Av = tv for a given collection of pairs (t,v).

        This solves systems Q^(T) a_i^(T) = Lambda Q^(T) where Q and Lambda are matrices as in the eigendecomposition and a_i is the i-th row of A.
        """
        eigenvals = np.array([val for (val, vec) in eigenpairs])
        eigenvecs = np.row_stack([vec for (val, vec) in eigenpairs])
        if eigenvecs.shape[0] != eigenvecs.shape[1]:
            raise ValueError(
                "Need n eigenpairs to determine matrix acting on R^n."
            )
        scaled = eigenvals.reshape(-1, 1) * eigenvecs
        print(eigenvecs)
        mat = np.row_stack(
            [
                np.linalg.solve(eigenvecs, scaled[:, i])
                for i in range(scaled.shape[1])
            ]
        )
        return mat
    return (mat_from_eigenpairs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Step 1: Center points

        **We assume in the following that the points themselves are centered i.e. `points == centered_points`!**
        """
    )
    return


@app.cell
def _(n_dims, np, points):
    center = np.mean(points, axis=1)
    centered_points = points - center.reshape(n_dims, -1)
    return center, centered_points


@app.cell
def _(mo):
    mo.md(r"""## Step 2: Determine first (longest) diagonal""")
    return


@app.cell
def _(centered_points, np, points):
    norms_1 = np.linalg.norm(centered_points, axis=0)
    diag_1 = points[:, np.argmax(norms_1)]
    return diag_1, norms_1


@app.cell
def _(mo):
    mo.md(r"""## Step 3: Project points such that the found diagonal is zero and determine second diagonal""")
    return


@app.cell
def _(
    centered_points,
    diag_1,
    np,
    points,
    proj_onto_orthogonal_complement,
):
    projected_points_1 = proj_onto_orthogonal_complement(diag_1) @ centered_points
    norms_2 = np.linalg.norm(projected_points_1, axis=0)
    diag_2 = points[:, np.argmax(norms_2)]
    return diag_2, norms_2, projected_points_1


@app.cell
def _(mo):
    mo.md(r"""## Step 4: Project the two already determined diagonals to zero and determine third diagonal""")
    return


@app.cell
def _(centered_points, diag_1, diag_2, mat_from_eigenpairs, np, points):
    projected_points_2 = (
        mat_from_eigenpairs(
            (0, diag_1), (0, diag_2), (1, np.cross(diag_1, diag_2))
        )
        @ centered_points
    )
    norms_3 = np.linalg.norm(projected_points_2, axis=0)
    diag_3 = points[:, np.argmax(norms_3)]
    return diag_3, norms_3, projected_points_2


@app.cell(hide_code=True)
def _(center, diags, go, points, projected_points_1, projected_points_2):
    # Create the 3D scatter plot
    _points = points

    _fig = go.Figure()
    _fig.update_layout(dict(title="Sample with determined diagonals"))
    _fig.add_trace(
        go.Scatter3d(
            x=_points[0],
            y=_points[1],
            z=_points[2],
            mode="markers",
            marker=dict(
                size=2,  # Size of markers
                opacity=0.6,  # Opacity of markers
            ),
            name="Sample",
        )
    )

    for i, _diag in enumerate(diags):
        _vertex = _diag
        _opposite_vertex = center - _diag
        _fig.add_trace(
            go.Scatter3d(
                x=[_vertex[0], _opposite_vertex[0]],
                y=[_vertex[1], _opposite_vertex[1]],
                z=[_vertex[2], _opposite_vertex[2]],
                mode="markers",
                marker=dict(
                    size=10,  # Size of markers
                    opacity=0.8,  # Opacity of markers
                ),
                name=f"Diagonal {i+1}",
            )
        )

    _fig.add_trace(
        go.Scatter3d(
            x=projected_points_1[0],
            y=projected_points_1[1],
            z=projected_points_1[2],
            mode="markers",
            marker=dict(
                size=2,  # Size of markers
                opacity=0.6,  # Opacity of markers
            ),
            name="Projection 1",
        )
    )


    _fig.add_trace(
        go.Scatter3d(
            x=projected_points_2[0],
            y=projected_points_2[1],
            z=projected_points_2[2],
            mode="markers",
            marker=dict(
                size=2,  # Size of markers
                opacity=0.6,  # Opacity of markers
            ),
            name="Projection 2",
        )
    )

    _fig
    return (i,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We now determine a matrix that orthonormalizes the determined diagonals:
        we want a transformation $A$ such that $\lVert Ad_i \rVert = 1$ for all $i=1,...,3$ and further it should "unskew" the diagonals which we achieve by positing that they be pairwise orthogonal post-transformation.
        Hence concisely we want to solve:

        \[
        \text{Find}~A: (A d_i)^T (A d_j) = \delta_{ij} \quad \text{for}~i,j=1,...,3\\
        \text{i.\,e.} ~ \langle d_i, d_j \rangle_A = \delta_{ij}.
        \]

        Let $A=QR$ be a QR-decomposition of $A$, then we easily check that $\langle d_i, d_j \rangle_A = \langle d_i, d_j \rangle_R$ independent of the orthogonal matrix $Q$. Letting $D := (d1d2d3)(d1d2d3)\begin{pmatrix} d_1 & d_2 & d_3 \end{pmatrix}$ be the matrix with the given diagonals as columns our orthonormality equation is equivalent to $D^T R^T R D = I$ where $I \in \R^{3,3}$ denotes the identity matrix. If our parallelotope is nondegenerate (which we have to assume) the matrix $D$ is invertible and hence $R^T R = (D^T)^{-1} D^{-1} = (D D^T)^{-1}$. Letting $L = R^T$ we find that $L L^T = (D D^T)^{-1}$ i.e. $LL^T$ is a Cholesky decomposition of $(D D^T)^{-1}$.

        We don't really care about the rotation $Q$. For aesthetic reasons we take $Q$ to be a matrix that aligns the three diagonals with the coordinate axes which is simply the transpose of $(Rd1Rd2Rd3)(Rd1Rd2Rd3)\begin{pmatrix} R d_1 & R d_2 & R d_3 \end{pmatrix}$. Analyzing this we find that in this case $A = D^{-1}$ so we can forego the whole decomposition.
        """
    )
    return


@app.cell
def _(np):
    def cross_prod_mat(v):
        """Computes the cross product matrix for a given vector v, i.e. the matrix K such that Kw = v \times w."""
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return (cross_prod_mat,)


@app.cell
def _(diag_1, diag_2, diag_3, np):
    diags = [diag_1, diag_2, diag_3]
    diags_mat = np.column_stack(diags)
    _cov = diags_mat @ diags_mat.T

    _skewing = np.linalg.cholesky(np.linalg.inv(_cov)).T
    _rotation = (_skewing @ diags_mat).T

    estimated_transformation_mat = _rotation @ _skewing
    # this is equivalent to the following line:
    # estimated_transformation_mat = np.linalg.inv(diags_mat)
    print(
        "Diagonals are numerically A-orthonormal: ",
        np.allclose(
            np.eye(3),
            np.array(
                [
                    [
                        np.dot(
                            estimated_transformation_mat @ d_i,
                            estimated_transformation_mat @ d_j,
                        )
                        for d_i in diags
                    ]
                    for d_j in diags
                ]
            ),
        ),
    )
    return diags, diags_mat, estimated_transformation_mat


@app.cell(hide_code=True)
def _(center, centered_points, diags, estimated_transformation_mat, go):
    _fig = go.Figure()
    _fig.update_layout(
        dict(title="Transformed sample under estimated transformation")
    )

    _points = estimated_transformation_mat @ centered_points

    _fig.add_trace(
        go.Scatter3d(
            x=_points[0],
            y=_points[1],
            z=_points[2],
            mode="markers",
            marker=dict(
                size=2,  # Size of markers
                opacity=0.6,  # Opacity of markers
            ),
            name="Sample",
        )
    )

    centered_diags = [_d - center for _d in diags]

    for _i, _diag in enumerate(centered_diags):
        _vertex = estimated_transformation_mat @ _diag
        _opposite_vertex = estimated_transformation_mat @ (-_diag)
        _fig.add_trace(
            go.Scatter3d(
                x=[_vertex[0], _opposite_vertex[0]],
                y=[_vertex[1], _opposite_vertex[1]],
                z=[_vertex[2], _opposite_vertex[2]],
                mode="markers",
                marker=dict(
                    size=10,  # Size of markers
                    opacity=0.8,  # Opacity of markers
                ),
                name=f"Diagonal {_i+1}",
            )
        )

    _fig.update_layout(
        scene=dict(
            xaxis=dict(title="X", range=[-3, 3]),
            yaxis=dict(title="Y", range=[-3, 3]),
            zaxis=dict(title="Z", range=[-3, 3]),
        ),
    )

    _fig.update_layout(scene_aspectmode="cube")
    _fig.layout.scene.camera.projection.type = "orthographic"
    # _fig.update_xaxes(scaleanchor="z", scaleratio=1)
    # _fig.update_yaxes(scaleanchor="z", scaleratio=1)

    _fig
    return (centered_diags,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The last vertex could be $d_1 + (d_2 - d_1) + (d_3 - d_1) = d_2 + d_3 - d_1$, but it could also be $d_1 + d_2 + d_3$ or indeed any one element of the set $\{ \sum_{i=1}^3 s_i d_i : s_i \in \{-1,1\}\}$. We have no way of telling these apart purely from the three diagonals and one convinces themself that these all yield valid parallelotopes. Hence we need to "touch" the pointcloud again: we compute the 9 different options for what the point might be and simply pick the one closest to our data.""")
    return


@app.cell(hide_code=True)
def _(
    center,
    centered_points,
    diags,
    estimated_transformation_mat,
    go,
    itt,
    np,
):
    _fig = go.Figure()
    _fig.update_layout(
        dict(title="Transformed sample under estimated transformation")
    )

    _points = estimated_transformation_mat @ centered_points

    _fig.add_trace(
        go.Scatter3d(
            x=_points[0],
            y=_points[1],
            z=_points[2],
            mode="markers",
            marker=dict(
                size=2,  # Size of markers
                opacity=0.6,  # Opacity of markers
            ),
            name="Sample",
        )
    )

    for _i, _diag in enumerate(diags):
        # because of how we constructed the vertices this is always a unit vector aligned with the axes, i.e. e_1, e_2 or e_3
        _vertex = estimated_transformation_mat @ _diag
        _fig.add_trace(
            go.Scatter3d(
                x=[-_vertex[0], _vertex[0]],
                y=[-_vertex[1], _vertex[1]],
                z=[-_vertex[2], _vertex[2]],
                mode="markers",
                marker=dict(
                    size=10,  # Size of markers
                    opacity=0.8,  # Opacity of markers
                ),
                name=f"Diagonal {_i+1}",
            )
        )

    _fig.add_trace(
        go.Scatter3d(
            x=[(estimated_transformation_mat @ center)[0]],
            y=[(estimated_transformation_mat @ center)[1]],
            z=[(estimated_transformation_mat @ center)[2]],
            mode="markers",
            marker=dict(
                size=10,  # Size of markers
                opacity=0.8,  # Opacity of markers
            ),
            name=f"Center",
        )
    )

    # _dir = 4*np.mean(estimated_transformation_mat @ diags_mat, axis=1)
    _dirs = [2 * np.array([1, *_s]) for _s in itt.product([1, -1], repeat=2)]
    for _i, _dir in enumerate(_dirs):
        _trace = go.Scatter3d(
            x=[-_dir[0], _dir[0]],
            y=[-_dir[1], _dir[1]],
            z=[-_dir[2], _dir[2]],
            mode="lines",
            marker=dict(
                size=10,  # Size of markers
                opacity=0.8,  # Opacity of markers
            ),
            name="Possible fourth diagonal",
            legendgroup="Possible fourth diagonal",
            # legendgrouptitle_text="Possible fourth diagonal",
            showlegend=_i == 0,
        )
        _fig.add_trace(_trace)


    _fig.update_layout(scene_aspectmode="cube")
    _fig.layout.scene.camera.projection.type = "orthographic"
    # _fig.update_xaxes(scaleanchor="z", scaleratio=1)
    # _fig.update_yaxes(scaleanchor="z", scaleratio=1)

    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We now consider only those points where the first coordinate is positive (since each diagonal corresponds to a pair of vertices any diagonal has one where the first entry is 1!), then the other two coordinates must each be either $+1$ or $-1$. To determine which "corner" we need, we determine which way "our cloud points the most".

        We let this point be $\hat{d}_4$. This point is connected by and edge of the cube to either $e_i$ or $-e_i$ for each $i=1,...,3$. We determine which ones we need --- call these $\hat{d}_i, i=1,...,3$ respectively --- an then map them all back to the space with the original point cloud using $A^{-1}$ to get $d_i' := A^{-1} \hat{d}_i$ for $i=1,...,4$.
        """
    )
    return


@app.cell
def _(centered_points, estimated_transformation_mat, np):
    transformed_points = estimated_transformation_mat @ centered_points
    positive_first = transformed_points[0, :] > 0
    positive_second = transformed_points[1, positive_first] > 0
    positive_third = transformed_points[2, positive_first] > 0

    # this just computes 2D dot products of various subsets of the data projected down onto a 2D plane. It ignores "negative contributions"
    # assuming enough sampling the negative ones shouldn't matter but I'm not quite sure how it works out if there's not *that* many points to begin with
    _p = transformed_points[1:, positive_first]
    _options = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
    _p2 = _p[0, positive_second].sum()
    _p3 = _p[1, positive_third].sum()
    _n2 = _p[0, ~positive_second].sum()
    _n3 = _p[1, ~positive_third].sum()
    _scores = [
        _p2 + _p3,
        _p2 - _n3,
        -_n2 + _p3,
        -_n2 - _n3,
    ]
    print(_scores)

    best_option = _options[np.argmax(_scores)]
    print(best_option)
    # determine point from cloud that's closest to the vertex we found
    """
    diag_4 = points[
        :,
        np.argmax(
            transformed_points[0, positive_first]
            + np.einsum("ij,i", _p, np.array(best_option))
        ),
    ]
    """
    _transformed_diag_4 = np.array([1, *best_option])
    # for topological reasons the new vertex must connect to exactly these other vertices
    transformed_diagonals = [
        np.array([1, 0, 0]),
        np.array([0, best_option[0], 0]),
        np.array([0, 0, best_option[1]]),
        _transformed_diag_4,
    ]
    diagonals = [
        np.linalg.solve(estimated_transformation_mat, d)
        for d in transformed_diagonals
    ]
    diag_4 = diagonals[-1]
    return (
        best_option,
        diag_4,
        diagonals,
        positive_first,
        positive_second,
        positive_third,
        transformed_diagonals,
        transformed_points,
    )


@app.cell(hide_code=True)
def _(
    center,
    centered_points,
    diagonals,
    estimated_transformation_mat,
    go,
):
    _fig = go.Figure()
    _fig.update_layout(dict(title="The determined fourth diagonal"))

    _points = estimated_transformation_mat @ centered_points

    _fig.add_trace(
        go.Scatter3d(
            x=_points[0],
            y=_points[1],
            z=_points[2],
            mode="markers",
            marker=dict(
                size=2,  # Size of markers
                opacity=0.6,  # Opacity of markers
            ),
            name="Sample",
        )
    )

    _diags = diagonals  # [*diags, diag_4]
    for _i, _diag in enumerate(_diags):
        _vertex = estimated_transformation_mat @ _diag
        _fig.add_trace(
            go.Scatter3d(
                x=[_vertex[0]],
                y=[_vertex[1]],
                z=[_vertex[2]],
                mode="markers",
                marker=dict(
                    size=10,  # Size of markers
                    opacity=0.8,  # Opacity of markers
                ),
                name=f"Diagonal {_i+1}",
            )
        )

    _fig.add_trace(
        go.Scatter3d(
            x=[(estimated_transformation_mat @ center)[0]],
            y=[(estimated_transformation_mat @ center)[1]],
            z=[(estimated_transformation_mat @ center)[2]],
            mode="markers",
            marker=dict(
                size=10,  # Size of markers
                opacity=0.8,  # Opacity of markers
            ),
            name=f"Center",
        )
    )

    _fig.update_layout(scene_aspectmode="cube")
    _fig.layout.scene.camera.projection.type = "orthographic"
    # _fig.update_xaxes(scaleanchor="z", scaleratio=1)
    # _fig.update_yaxes(scaleanchor="z", scaleratio=1)

    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Final Step: the big cubification :D

        We now consider the three edges of our transformed parallelotope pointing from $d'_4$ to $d'_1, d'_2, d'_3$. We construct a new linear map that takes these to $e_1, e_2, e_3$. This is finally the map that "cubifies" our parallelotope!

        The full affine map hence becomes

        \[
        \operatorname{Cubify}(x) = (d'_1 - d'_4, d'_2 - d'_4, d'_3 - d'_4)^{-1} (x - d'_4) \\
         = (A^{-1} (\hat{d}_1 - \hat{d}_4), ..., A^{-1} (\hat{d}_3 - \hat{d}_4))^{-1} (x - d'_4) \\
         = (\hat{d}_1 - \hat{d}_4, ..., \hat{d}_3 - \hat{d}_4)^{-1} A (x - d'_4)
        \]

        Hence it's easily computed from $A$ by inverting that left "edge matrix". Note that there's only a handful of options of how this matrix could look like so it'd be feasible to precompute these and simply pick the one that's needed for the current setup.
        """
    )
    return


@app.cell
def _(centered_points, diag_4, diagonals, np):
    _standard_basis = [
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1]),
    ]
    edge_vectors = np.column_stack([d - diag_4 for d in diagonals[:3]])
    fitted_transformation_mat = np.linalg.inv(edge_vectors)
    print(edge_vectors)

    cubified_points = np.linalg.inv(edge_vectors) @ (
        centered_points - diag_4.reshape(3, -1)
    )
    return cubified_points, edge_vectors, fitted_transformation_mat


@app.cell(hide_code=True)
def _(cubified_points, go):
    _fig = go.Figure()
    _fig.update_layout(dict(title="The cubified data"))

    _points = cubified_points

    _fig.add_trace(
        go.Scatter3d(
            x=_points[0],
            y=_points[1],
            z=_points[2],
            mode="markers",
            marker=dict(
                size=2,  # Size of markers
                opacity=0.6,  # Opacity of markers
            ),
            name="Sample",
        )
    )

    _fig.update_layout(scene_aspectmode="cube")
    _fig.layout.scene.camera.projection.type = "orthographic"

    _fig
    return


@app.cell(hide_code=True)
def _(diag_4, fitted_transformation_mat, go, np, points):
    # Create the 3D scatter plot
    _points = points
    _fig = go.Figure()
    _fig.update_layout(dict(title="And a mesh for the data"))

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
    _cube_verts = np.linalg.inv(
        fitted_transformation_mat
    ) @ _unit_cube_verts + diag_4.reshape(3, -1)


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


if __name__ == "__main__":
    app.run()
