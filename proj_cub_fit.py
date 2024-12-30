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
    from typing import NamedTuple
    return NamedTuple, go, itt, mo, np, scp


@app.cell
def _(NamedTuple, np):
    class AffineMap(NamedTuple):
        """Represents an affine transformation x \mapsto Ax + b on R^(n)
        via its associated linear transformation A and offset b.
        """

        linear_transf: np.ndarray
        offset: np.ndarray

        def apply_to(self, points):
            return self.linear_transf @ points + self.offset.reshape(3, 1)

        def __matmul__(self, other):
            if type(other) is AffineMap:
                return self.after(other)
            else:
                return self.apply_to(other)

        def after(self, other: "AffineMap") -> "AffineMap":
            """Computes the composition self \circ other."""
            return AffineMap(
                self.linear_transf @ other.linear_transf,
                self.linear_transf @ other.offset + self.offset,
            )

        def inv(self) -> "AffineMap":
            inv = np.linalg.inv(self.linear_transf)
            return AffineMap(inv, -inv @ self.offset)
    return (AffineMap,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Some helpers""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let $v_1, ..., v_k$ be $k$ linearly independent vectors in $\R^n$.
        We want to determine a matrix $A$ such that $A v_i = 0$ for all $i$ and $A x = x$ for all $x \in W^\bot$ where $W := \operatorname{span}(v_1, ..., v_k)$.

        Let $V = (v_1, ..., v_k) \in \R^{n,k}$.
        It's a standard result that $P := V V^\dagger$ is the orthogonal projector onto $W$ --- here $V^\dagger$ denotes the moore-penrose pseudoinverse of $V$. Since the $\{v_i\}$ are linearly independent $V$ has full column-rank and hence we have $V^\dagger = (V^T V)^{-1} V^T$ and hence $P = V (V^T V)^{-1} V^T.$

        Now let $V = U \Sigma Q^T$ be a SVD of $V$.

        Then $V (V^T V)^{-1} V^T = V (Q \Sigma^T U^T U \Sigma Q^T)^{-1} V^T = V (Q(\Sigma^T \Sigma) Q^T)^{-1} V^T = V Q (\Sigma^T \Sigma)^{-1} Q^T V^T = U \Sigma (\Sigma^T \Sigma)^{-1} \Sigma^T U^T$. Since $\Sigma^T \Sigma$ is a diagonal matrix it follows that $P = UU^T$ and hence $A := I - UU^T$ works.

        Similarly we can show that for a thin / reduced QR decomposition $V = QR$ we have $A = I - QQ^T$.
        """
    )
    return


@app.cell
def _(np):
    def proj_onto_orthogonal_complement_single(v):
        """
        Similar to projon→orthogonalcomp≤mentproj_onto_orthogonal_complement special-cased to a single vector.

        Note: One could also implement a special version for two vectors in R^3 or more
        generally n-1 vectors in R^n.
        """
        v_unit = v / np.linalg.norm(v)
        projection_matrix = np.outer(v_unit, v_unit)
        return np.eye(v.shape[0]) - projection_matrix


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
        mat = np.row_stack(
            [
                np.linalg.solve(eigenvecs, scaled[:, i])
                for i in range(scaled.shape[1])
            ]
        )
        return mat


    def proj_onto_orthogonal_complement(*vectors: np.ndarray) -> np.ndarray:
        """
        Returns a linear transformation that projects onto the orthogonal complement of the
        span of a given collection of vectors \{v_i\}. The vectors are assumed to be linearly
        independent.

        So it maps all the vectors v_i to 0 and (span{v_i}_i)^\bot is invariant under the map.
        """
        # we'd really want to return the identity if len(vectors) == 0 but in this case we don't know the dimension
        if len(vectors) == 1:
            # this isn't really necessary and the below code works for a single vector but eh
            return proj_onto_orthogonal_complement_single(vectors[0])
        # if len(vectors) == 2: # for 3D
        #     return mat_from_eigenpairs(
        #         (0, vectors[0]),
        #         (0, vectors[1]),
        #         (1, np.cross(vectors[0], vectors[1])),
        #     )
        else:
            vec_mat = np.column_stack(vectors)
            qr_decomp = np.linalg.qr(vec_mat, mode="reduced")
            projection_matrix = qr_decomp.Q @ qr_decomp.Q.T
            return np.eye(vec_mat.shape[0]) - projection_matrix
    return (
        mat_from_eigenpairs,
        proj_onto_orthogonal_complement,
        proj_onto_orthogonal_complement_single,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Central algorithm""")
    return


@app.cell
def _(AffineMap, NamedTuple, np, proj_onto_orthogonal_complement):
    class CubeFit(NamedTuple):
        # remaps the parallelotope in R^d such that d of its diagonals are orthonormal post-transformation (and the center is at the origin).
        diagonal_orthonormalizer: AffineMap
        # maps the parallelotope into the unit cube [0,1]^d.
        cubifier: AffineMap


    def proj_cube_fit(
        points: np.ndarray, points_centered: bool = False
    ) -> AffineMap:
        """

        Remark: for more details on the implementation please refer to the [prototypes/iter_proj_cube_fit_2.py] notebook.
        The basic idea of the algorithm is to successively determine diagonals by determining maximal elements in those subspaces that are orthogonal to the already determines diagonals.
        It utilizes that the image of a parallelotope under a linear map is again a parallelotope, that the preimages of
        vertices in the image are also vertices in the original parallelotope, and that any maximizer of the norm over a
        parallelotope is necessarily at a vertex (as is easily seen by noting that the euclidean norm is strongly convex
        and hence f(x) = ||T(x)||_2 is as well for affine maps T; it also seems geometrically rather self-evident.)

        Args:
        * points: Array of shape (d,n) where n is the number of points and d the dimension of space
        * points_centered: whether the inputs points may be assumed to be centered around the origin.
        """
        dim_space = points.shape[0]
        # center points as needed
        if points_centered:
            center = np.zeros(dim_space)
            centered_points = points.copy()
        else:
            center = np.mean(points, axis=1)
            centered_points = points - center.reshape(dim_space, -1)

        # determine d diagonals of the parallelotope ordered by their lengths
        norms_1 = np.linalg.norm(centered_points, axis=0)
        diag_1 = centered_points[:, np.argmax(norms_1)]
        diags = [diag_1]  # could preallocate for dim_space vectors here
        # we want to / are able to determine n diagonals in total, but we already know one
        for _ in range(dim_space - 1):
            ortho_projector = proj_onto_orthogonal_complement(*diags)
            projected_points = ortho_projector @ centered_points
            norms = np.linalg.norm(projected_points, axis=0)
            new_diag = centered_points[:, np.argmax(norms)]
            diags.append(new_diag)

        diags_mat = np.column_stack(diags)
        # This estimates the transformation by that matrix A that makes the diagonals A-orthonormal
        # and moreover maps the i-th diagonal to the i-th standard unit vector
        estimated_transformation_mat = np.linalg.inv(diags_mat)
        diagonal_orthonormalizer = AffineMap(estimated_transformation_mat, -center)
        # the 2D case would really be done at this point (since we can just pick one diagonal point
        # and we know that it's connected to both points of the other diagonal by an edge; in
        # constrast to the 3D case)
        # and the >3 dimensional one needs a bit more thinking:
        # We can probably again pick the point of maximal norm in the transformed space as next
        # diagonal and then construct connected vertices similar to 3D case. But I haven't thought it out properly and I'm not sure whether that indeed uniquely determines the parallelotope

        transformed_points = estimated_transformation_mat @ centered_points
        positive_first = transformed_points[0, :] > 0
        positive_second = transformed_points[1, positive_first] > 0
        positive_third = transformed_points[2, positive_first] > 0

        p = transformed_points[:, positive_first]
        transformed_diag_4 = p[:, np.argmax(np.linalg.norm(p, axis=0))]
        # for topological reasons the new vertex must connect to exactly these other vertices
        _sgn_d4 = np.sign(transformed_diag_4)
        transformed_diagonals = [
            np.array([1, 0, 0]),
            np.array([0, _sgn_d4[1], 0]),
            np.array([0, 0, _sgn_d4[2]]),
            transformed_diag_4,
        ]
        """
        # black magic or something like that
        # this just computes 2D dot products of various subsets of the data projected down onto a 2D plane. It ignores "negative contributions"
        # assuming enough sampling the negative ones shouldn't matter but I'm not quite sure how it works out if there's not *that* many points to begin with
        
        p = transformed_points[1:, positive_first]
        options = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
        p2 = p[0, positive_second].sum()
        p3 = p[1, positive_third].sum()
        n2 = p[0, ~positive_second].sum()
        n3 = p[1, ~positive_third].sum()
        scores = [
            p2 + p3,
            p2 - n3,
            -n2 + p3,
            -n2 - n3,
        ]
        best_option = options[np.argmax(scores)]
        # determine point from cloud that's closest to the vertex we found
        transformed_diag_4 = np.array([1, *best_option])
        
        # for topological reasons the new vertex must connect to exactly these other vertices
        transformed_diagonals = [
            np.array([1, 0, 0]),
            np.array([0, best_option[0], 0]),
            np.array([0, 0, best_option[1]]),
            transformed_diag_4,
        ]"""
        # construct "cubification" map
        transformed_edge_vectors = [
            d - transformed_diag_4 for d in transformed_diagonals[:3]
        ]
        edge_cubifier = np.linalg.inv(np.column_stack(transformed_edge_vectors))
        fitted_transformation_mat = edge_cubifier @ estimated_transformation_mat

        cubifier = AffineMap(
            fitted_transformation_mat, -edge_cubifier @ transformed_diag_4
        ).after(AffineMap(np.eye(dim_space), -center))
        return CubeFit(diagonal_orthonormalizer, cubifier)
    return CubeFit, proj_cube_fit


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## An example on generated data""")
    return


@app.cell(hide_code=True)
def _(mo):
    num_points = mo.ui.slider(50, 20_000, show_value=True, value=10_000)
    mo.md(f"Number of points in generated sample: {num_points}")
    return (num_points,)


@app.cell
def _(np, num_points):
    n_dims = 3
    # generate some points in unit cube
    _rng = np.random.default_rng(1)

    # solid cube
    _p = _rng.uniform(0, 1, size=(3, num_points.value))
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
        [[2, 1, 2.85], [0, 3, 0.25], [40, 0, 1.5]]
    )
    # _center = np.mean(_p, axis=1)
    _p = _p
    points = (
        transformation_mat @ _p
        + 0.1 * _rng.normal(size=_p.shape)
        + 100 * _rng.normal(size=(3, 1))
    )

    # points = points - np.mean(points, axis=1).reshape(3, 1)

    points_unit = _p
    print(transformation_mat)
    return n_dims, points, points_unit, rot_mat, transformation_mat


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Note that the generated data is quite heavily skewed and a bit noisy""")
    return


@app.cell
def _(go, points):
    _fig = go.Figure()
    _fig.update_layout(dict(title="The generated data sample"))

    _points = points

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


@app.cell
def _(points, proj_cube_fit):
    fit = proj_cube_fit(points, points_centered=False)
    return (fit,)


@app.cell(hide_code=True)
def _(fit, mo, np, points):
    # we want to map the points into [0,1]³ hence we should have an infinity-norm ball of radius 0.5 around (1/2, ..., 1/2). Points that fall outside of this ball are "incorrect"
    algebraic_residual = np.linalg.norm(
        fit.cubifier.apply_to(points) - 0.5 * np.ones(3).reshape(3, 1),
        ord=np.inf,
        axis=0,
    )
    res = np.sum(algebraic_residual - 0.5, where=algebraic_residual > 0.5)
    mo.md(f"### Algebraic residual of fit: {res:.3f}").center()
    return algebraic_residual, res


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Note**: this residual doesn't tell the whole story. Since the distance function $d_C$ of the cube is $0$ on its interior any large enough cube (that encompasses all of the data) trivially has a residual of 0.
        So instead we really want to consider objectives of the form

        \[
            \sum_{i=1}^N d_C(x_i)^2 + \lambda \operatorname{vol}(C)
        \]

        for regularization constants $\lambda > 0$. More on that when we get to the gradient iteration.
        """
    )
    return


@app.cell(hide_code=True)
def _(fit, go, points):
    _fig = go.Figure()
    _fig.update_layout(dict(title="The cubified data"))

    _points = fit.cubifier @ points
    # we "hollow out" the cube a bit to make the plot more performant
    # _points = _points[:, algebraic_residual > 0.3]

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
def _(fit, go, points):
    _fig = go.Figure()
    _fig.update_layout(dict(title="Data with orthonormalized diagonals"))

    _points = fit.diagonal_orthonormalizer @ points
    # we "hollow out" the cube a bit to make the plot more performant
    # _points = _points[:, algebraic_residual > 0.3]

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
def _(mo):
    mo.md(r"""We may be able to improve our estimate by iterating the algorithm:""")
    return


@app.cell
def _(fit, points, proj_cube_fit):
    estimated_map2 = proj_cube_fit(fit.cubifier.apply_to(points)).cubifier.after(
        fit.cubifier
    )

    # or by utilizing the orthonormalized representation
    estimated_map3 = proj_cube_fit(
        fit.diagonal_orthonormalizer.apply_to(points)
    ).cubifier.after(fit.diagonal_orthonormalizer)
    return estimated_map2, estimated_map3


@app.cell(hide_code=True)
def _(estimated_map2, go, points):
    _fig = go.Figure()
    _fig.update_layout(dict(title="The doubly cubified data"))

    _points = estimated_map2 @ points

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
def _(estimated_map2, mo, np, points):
    # we want to map the points into [0,1]³ hence we should have an infinity-norm ball of radius 0.5 around (1/2, ..., 1/2). Points that fall outside of this ball are "incorrect"
    _algebraic_residual = np.linalg.norm(
        estimated_map2.apply_to(points) - 0.5 * np.ones(3).reshape(3, 1),
        ord=np.inf,
        axis=0,
    )
    _res = np.sum(_algebraic_residual - 0.5, where=_algebraic_residual > 0.5)
    mo.md(f"### Algebraic residual of second fit: {_res:.3f}").center()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Projected Gradient Iteration

        There's also the option of improving the map determined in the first step by a subsequent projected gradient iteration. However care has to be taken to control the step size.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Some background

        Given a convex set $C$ (in our case $C = [0,1]^d$ the unit cube) its squared distance function $\varphi_C(x) = \frac{1}{2} \min_{y \in C} \lVert x - y \rVert^2_2$ is differentiable and $\nabla \varphi_C(x) = x - P_C(x)$ where $P_C(x)$ denotes the (well defined) projection onto $C$.

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

        The orthogonal projection (w.r.t. Frobenius inner product) onto the set of orthogonal matrices is given by $A \mapsto UV^T$ where $A=USV^T$ is an SVD of $A$.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This objective has some issues when we consider not just orthogonal matrices: as noted previously, for the general case we really want to minimize a weighted sum of the volume of the cube and the residual error of the fit.

        In our setup this translates to solving

        \[
            \min_{A} \sum_{i=1}^N d_{[0,1]^n}(A x_i)^2 + \lambda \operatorname{det}(A).
        \]

        i.e.


        \[
            \min_{A} \sum_{n=1}^N \varphi_C(Ax_n)^2 + \lambda \operatorname{det}(A).
        \]

        As previously determined we have (with some abuse of notation; we're really taking a derivative w.r.t. the entries of the matrix here) $\nabla_A (A \mapsto \varphi_C (Ax_n))|_A = (Ax_n - P_C(Ax_n)) \otimes x_n$ and by Jacobi's formula we have (assuming an invertible matrix) $\nabla \det|_A = \operatorname{adj}(A)^T = \det(A) A^{-T}$ so that our full gradient becomes

        \[
            (Ax_n - P_C(Ax_n)) \otimes x_n + \lambda \det(A) A^{-T}
        \]
        """
    )
    return


@app.cell
def _():
    from numba import njit
    return (njit,)


@app.cell
def _(njit, np):
    def proj_unit_cube(points: np.ndarray, out=None) -> np.ndarray:
        return np.clip(points, 0, 1, out=None)


    @njit
    def proj_boundary_unit_cube(points: np.ndarray, out=None) -> np.ndarray:
        """
        Project a set of points onto the boundary of the unit cube [0, 1]^n.

        Parameters:
            points: np.ndarray
                An n x N array where n is the dimension of space and N is the number of points.
            out: np.ndarray, optional
                If provided, the output is stored in this array.

        Returns:
            np.ndarray: The projected points, n x N array.
        """
        # Clamp points to the interior of the unit cube
        clamped = np.clip(points, 0, 1, out=out)

        # Compute distances to the boundaries (0 and 1) for each coordinate
        dist_to_0 = clamped
        dist_to_1 = 1 - clamped

        # For each point, identify the coordinate closest to the boundary
        min_dist_to_boundary = np.minimum(dist_to_0, dist_to_1)
        idx_closest = np.argmin(min_dist_to_boundary, axis=0)

        # Set the closest coordinate of each point to the respective boundary (0 or 1)
        for i, idx in enumerate(idx_closest):
            clamped[idx, i] = 0 if dist_to_0[idx, i] <= dist_to_1[idx, i] else 1

        return clamped


    @njit
    def det_gradient(A):
        """
        Compute the gradient of det(A) with respect to the matrix A.

        Parameters:
            A: np.ndarray
                A square matrix A (n x n).

        Returns:
            np.ndarray: The gradient of det(A) with respect to A.
        """
        det_A = np.linalg.det(A)
        inv_A_T = np.linalg.inv(A).T  # Transpose of the inverse
        return det_A * inv_A_T


    def make_objective_val_and_grad(
        points: np.ndarray, regularization_factor: float
    ):
        """Points is d \times N array of cartesian point coordinates
        where d is the dimensions of space and N is the number of point.

        Call these x_1, ..., x_N
        """

        def objective_val_and_grad(matrix: np.ndarray):
            # Call the matrix A
            # This is A x_n
            transformed_points = matrix @ points
            # proj = proj_boundary_unit_cube(transformed_points)
            proj = proj_unit_cube(transformed_points)
            # gradient of squared distance function at Ax_n
            dist_grad = transformed_points - proj
            # "gradient" w.r.t matrix; this compute the outer products (Ax_n - P_C(Ax_n)) \otimes x_n = (Ax_n - P_C(Ax_n)) x_n^(T) for each n
            # so this is morally [np.outer(dist_grad[:, i], points[:, i]) for i in range(num_points)]
            mat_grad = np.einsum("ik,jk->ijk", dist_grad, points)
            # to obtain the full gradient we simply sum over all points
            full_grad = np.sum(
                mat_grad, axis=2
            ) + regularization_factor * det_gradient(matrix)
            # lets also return the objective value since we can easily compute it here
            objective_value = (
                # this should be np.linalg.vecdot(dist_grad, dist_grad, axis=0).sum()) but can't use current numpy due to scipy
                np.einsum("ij,ij->j", dist_grad, dist_grad).sum()
            )
            return objective_value, full_grad

        return objective_val_and_grad


    def fit_cube_naive_subgradient(
        points: np.ndarray,
        starting_guess: np.ndarray,
        max_iters: int = 10_000,
        abs_tol: float = 1e-3,
        step_size: float = 1e-4,
        regularization_factor=10,
    ) -> np.ndarray:
        """Fits a cube to the given points by running a projected gradient method"""
        matrix = starting_guess.copy()
        obj_val_and_grad = make_objective_val_and_grad(
            points, regularization_factor
        )
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
            svd = np.linalg.svd(matrix, full_matrices=True)
            # we "project" (not really) onto the invertible matrices by forcing all singular values to be sliiiightly positive.
            matrix = (svd.U * np.maximum(svd.S, 1e-4)) @ svd.Vh
        return matrix
    return (
        det_gradient,
        fit_cube_naive_subgradient,
        make_objective_val_and_grad,
        proj_boundary_unit_cube,
        proj_unit_cube,
    )


@app.cell
def _(AffineMap, estimated_map2, fit_cube_naive_subgradient, np, points):
    _mat = fit_cube_naive_subgradient(
        estimated_map2.apply_to(points),
        starting_guess=np.identity(3),
        max_iters=50_000,
        abs_tol=1e-4,
        step_size=2e-3,
        regularization_factor=10,
    )
    iterated_estimate = AffineMap(_mat, np.zeros(3)).after(estimated_map2)
    return (iterated_estimate,)


@app.cell(hide_code=True)
def _(iterated_estimate, mo, np, points):
    # we want to map the points into [0,1]³ hence we should have an infinity-norm ball of radius 0.5 around (1/2, ..., 1/2). Points that fall outside of this ball are "incorrect"
    _algebraic_residual = np.linalg.norm(
        iterated_estimate.apply_to(points) - 0.5 * np.ones(3).reshape(3, 1),
        ord=np.inf,
        axis=0,
    )
    _res = np.sum(_algebraic_residual - 0.5, where=_algebraic_residual > 0.5)
    mo.md(f"### Algebraic residual of projected gradient fit: {_res:.3f}").center()
    return


@app.cell(hide_code=True)
def _(go, iterated_estimate, points):
    _fig = go.Figure()
    _fig.update_layout(
        dict(title="The estimate of the map after projected gradient iteration")
    )

    _points = iterated_estimate @ points

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
def _(mo):
    mo.md(
        r"""
        ## Approximating the data

        By inverting the determined maps we can now obtain approximations to the data.

        This meshing example assumes the "cubifier" map to be perfect which might exclude some points that are considered noise. To include these replace the cube vertices by the corresponding ones of the bounding box of the cubified data.
        """
    )
    return


@app.cell(hide_code=True)
def _(go, iterated_estimate, np, points):
    # Create the 3D scatter plot
    _points = points
    _fig = go.Figure()
    _fig.update_layout(dict(title="Finally a mesh approximation for the data"))

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
    _cube_verts = iterated_estimate.inv().apply_to(_unit_cube_verts)


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


if __name__ == "__main__":
    app.run()
