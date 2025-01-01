import marimo

__generated_with = "0.10.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""This generalizes the algorithm from [proj_cube_fit.py](proj_cub_fit.py) to general parallelotopes $P \subseteq \R^3$ (i.e. $P = A ([0,1]^3) + b$ where $A \in \R^{3,3}$ not necessarily invertible) and optimizes some aspects of the implementation. It also provides further theoretical justification for the algorithm.""")
    return


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
            return self.linear_transf @ points + self.offset.reshape(
                self.linear_transf.shape[0], 1
            )

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
    class UpdatableOrthogonalProjection:
        """Given vectors v_1, ..., v_i in R^n this allows to compute the projection operator onto the
        orthogonal complement of W_i := span(v_1, ..., v_i); and to update this projection as additional
        vectors v_{i+1} are added.
        """

        def __init__(self, n_dims: int):
            # span of empty set is trivial subspace and complement is full space
            self.projector = np.identity(n_dims)

        def add_vector_to_generators(self, new_vec: np.ndarray) -> np.ndarray:
            """This adds a vector to the spanning set i.e. updates from W_i to W_{i+1}."""
            projected = self.projector @ new_vec
            norm = np.linalg.norm(projected)
            if np.isclose(norm, 0):
                # new vector is linearly dependent with previous ones, no need to update
                return self.projector
            else:
                proj_unit = projected / norm
                self.projector -= np.outer(proj_unit, proj_unit)
                return self.projector
    return (UpdatableOrthogonalProjection,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In addition to updating the projection we might also be able to update the norm of the projections of all points. Lets look into this:

        Let $u_i = \frac{P_i v_{i+1}}{\lVert P_i v_{i+1} \rVert}$ such that $P_{i+1} = P_i - u_i \otimes u_i$.

        Then

        \[
        \lVert P_{i+1} x \rVert_2^2 = \lVert P_i x - u_i \otimes u_i x \rVert_2^2 = \lVert P_i x - \langle u_i, x \rangle u_i \rVert_2^2 = \lVert P_i x \rVert_2^2 - 2\langle u_i, x \rangle \langle P_i x, u_i \rangle + \lVert \langle u_i, x \rangle u_i \rVert_2^2 \\
        = \lVert P_i x \rVert_2^2 - 2\langle u_i, x \rangle \langle P_i x, u_i \rangle + \langle u_i, x \rangle^2 = \lVert P_i x \rVert_2^2 - \langle u_i, x \rangle (2\langle P_i x, u_i \rangle - \langle u_i, x \rangle) \\
        = \lVert P_i x \rVert_2^2 - \langle u_i, x \rangle (\langle P_i x, u_i \rangle + \langle P_i x - x, u_i \rangle)
        \]

        Since $u_i \in W_i$ and $P_i$ is the orthogonal projection onto $W_i$ we have $\langle P_i x - x, u_i \rangle = 0$ and hence

        \[
        \lVert P_{i+1} x \rVert_2^2 = \lVert P_i x \rVert_2^2 - \langle u_i, x \rangle \langle u_i, P_i x \rangle
        \]

        I don't immediately see how this would give us a cheap update: computing $\langle u_i, x \rangle$ is reasonably cheap, but to compute the other factor in the update term we need to know either $P_i^T u_i$ or $P_i x$.

        This helps though: we easily see that by induction the matrices $P_i$ are symmetric. Hence $P_i^T u_i = P_i u_i$ which by idempotence of $P_i$ is just $u_i$. Hence $\langle u_i, P_i x \rangle = \langle P_i^T u_i, x \rangle = \langle u_i, x \rangle$ and we obtain the cheaply computable update

        \[
        \lVert P_{i+1} x \rVert_2^2 = \lVert P_i x \rVert_2^2 - \langle u_i, x \rangle^2.
        \]
        """
    )
    return


@app.cell
def _(np):
    class UpdatableOrthogonalProjectionWithNormUpdate:
        """Given vectors v_1, ..., v_i in R^n this allows to compute the projection operator P_i onto the
        orthogonal complement of W_i := span(v_1, ..., v_i); and to update this projection as additional
        vectors v_{i+1} are added. Additionally given a second collection of vectors w_1, ..., w_N this
        also maintains the squared norms ||P_i w_j||_2^2 for all j.
        """

        def __init__(self, n_dims: int, points: np.ndarray):
            # span of empty set is trivial subspace and complement is full space
            self.projector = np.identity(n_dims)
            # (yes directly computing the dot product would be better here, blame libraries that still
            # use old numpy versions without proper dot product for me using the norm here)
            self.projected_sq_norms = np.linalg.norm(points, axis=0) ** 2
            self.points = points

        def add_vector_to_generators(self, new_vec: np.ndarray) -> np.ndarray:
            """This adds a vector to the spanning set i.e. updates from W_i to W_{i+1}."""
            projected = self.projector @ new_vec
            norm = np.linalg.norm(projected)

            if np.isclose(norm, 0):
                # new vector is linearly dependent with previous ones, no need to update
                new_proj = self.projector
                # norms don't change either
            else:
                proj_unit = projected / norm
                self.projector -= np.outer(proj_unit, proj_unit)
                new_proj = self.projector

                # update norms (if more precision is needed use a compensated sum here)
                self.projected_sq_norms -= (
                    np.einsum("i,ij", proj_unit, self.points) ** 2
                )
            return new_proj
    return (UpdatableOrthogonalProjectionWithNormUpdate,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Central algorithm""")
    return


@app.cell
def _(
    AffineMap,
    NamedTuple,
    UpdatableOrthogonalProjectionWithNormUpdate,
    np,
):
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
        _proj = UpdatableOrthogonalProjectionWithNormUpdate(
            dim_space, centered_points
        )
        # we want to / are able to determine n diagonals in total, but we already know one
        for _ in range(dim_space - 1):
            ortho_projector = _proj.add_vector_to_generators(diags[-1])
            _max_idx = np.argmax(_proj.projected_sq_norms)
            """ the previous line is equivalent to these ones
            projected_points = ortho_projector @ centered_points
            norms = np.linalg.norm(projected_points, axis=0)
            _max_idx = np.argmax(norms)
            """
            # if we're looking for parallelotopes  P = A [0,1]^n + b with A singular, we want to check
            # whether the maximum of the projected norms is (close to) zero and if it is break out of
            # the loop: at that point we've determined all diagonals of the parallelotope and can construct the desired map
            # however we'll for now assume A to be invertible and continue:
            new_diag = centered_points[:, _max_idx]
            diags.append(new_diag)

        diags_mat = np.column_stack(diags)
        # This estimates the transformation by that matrix A that makes the diagonals A-orthonormal
        # and moreover maps the i-th diagonal to the i-th standard unit vector
        estimated_transformation_mat = np.linalg.inv(diags_mat)
        diagonal_orthonormalizer = AffineMap(estimated_transformation_mat, -center)

        # we may freely assume which diagonals of the unit cube "we have" modulo their signs
        # note that if we replace any such diagonal by its reflection (i.e. we get "the wrong sign")
        # it results in a reflection of the resulting point as well --- which again doesn't matter.
        transformed_points = estimated_transformation_mat @ centered_points
        transformed_norms = np.linalg.norm(transformed_points, axis=0)
        transformed_diag_4 = transformed_points[:, np.argmax(transformed_norms)]
        # for topological reasons the new vertex must connect to exactly these other vertices:
        sgn_d4 = np.sign(transformed_diag_4)
        transformed_diagonals = [
            np.array([sgn_d4[0], 0, 0]),
            np.array([0, sgn_d4[1], 0]),
            np.array([0, 0, sgn_d4[2]]),
            transformed_diag_4,
        ]
        # construct "cubification" map
        transformed_edge_vectors = [
            d - transformed_diag_4 for d in transformed_diagonals[:3]
        ]
        edge_cubifier = np.linalg.inv(np.column_stack(transformed_edge_vectors))
        fitted_transformation_mat = edge_cubifier @ estimated_transformation_mat

        cubifier = AffineMap(
            fitted_transformation_mat, -edge_cubifier @ transformed_diag_4
        ).after(AffineMap(np.identity(dim_space), -center))
        return CubeFit(diagonal_orthonormalizer, cubifier)
    return CubeFit, proj_cube_fit


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Call a set $P \subseteq \R^n$ a centered polytope if its the image of $[-1,1]^n$ under a linear map $A$.
        Call the images of the vertices of $[-1,1]^n$ under $A$ the diagonals of $P$ and note that linear maps preserve diagonals.

        ### Theorem
        Let $d_1, ..., d_3$ be linearly independent diagonals of $P = [-1,1]^3$. Moreover let $T$ be the linear map such that $T d_i = e_i$ for all $i=1,...,3$; i.e. $T = D^{-1}$ where $D = (d_1, ..., d_3)$.
        Then $TP$ has one vertex of the form $(s_1, ..., s_3)$ with all the $s_i \in \{+1, -1\}$ i.e. $TP$ shares a vertex with $[-1,1]^3$.

        **Proof**

        It suffices to check that for any choice of $d_1, ..., d_3$ there is some vector $x$ with all entries in $\{\pm 1\}$ such that $Ds$ also has all entries in $\{\pm 1\}$ (since in that case $s := D^{-1} x$ has the desired property). There are (up to a sign, which doesn't matter here) only four diagonals and one easily checks by exhausting all possibilities that indeed $s=d_4$ and $s=-d_4$ work, where $d_4$ is precisely the "leftover" diagonal of $[-1,1]^3$ that's not among the $d_1,...,d_3$. Explicitly we have:

        \[
        \begin{pmatrix} 1 & 1 & 1 \\ 1 & 1 & -1 \\ 1 & -1 & 1  \end{pmatrix} \begin{pmatrix} 1 \\ -1 \\ -1 \end{pmatrix} = -\begin{pmatrix} 1 \\ -1 \\ -1 \end{pmatrix}
        \]

        \[
        \begin{pmatrix} 1 & 1 & 1 \\ 1 & 1 & -1 \\ 1 & -1 & -1  \end{pmatrix} \begin{pmatrix} 1 \\ -1 \\ 1 \end{pmatrix} = -\begin{pmatrix} 1 \\ -1 \\ 1 \end{pmatrix}
        \]

        \[
        \begin{pmatrix} 1 & 1 & 1 \\ 1 & -1 & 1 \\ 1 & -1 & -1  \end{pmatrix} \begin{pmatrix} 1 \\ 1 \\ -1 \end{pmatrix} = \begin{pmatrix} 1 \\ -1 \\ 1 \end{pmatrix}
        \]

        \[
        \begin{pmatrix} 1 & 1 & -1 \\ 1 & -1 & 1 \\ 1 & -1 & -1  \end{pmatrix} \begin{pmatrix} 1 \\ 1 \\ 1 \end{pmatrix} = \begin{pmatrix} 1 \\ 1 \\ -1 \end{pmatrix}
        \]

        **Remark**

        This theorem essentially shows why the step that determines the final vertex from the three previously determined diagonals works. Note that the analogous theorem isn't generally true in other dimensions: in 2 dimensions it's obviously false and using a CAS we also quickly find it to be wrong in 4 dimensions. In 5 dimensions we find that there's zero, one or two vectors (modulo reflection) satisfying the claim.
        This suggests that some solutions may exist in odd dimensions while there are none in odd dimensions.

        Finally note how we can compute the "unused diagonal" as the pointwise product of the used ones.
        """
    )
    return


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


@app.cell(disabled=True)
def _(estimated_map2, make_objective_val_and_grad, np, points, scp):
    _og = make_objective_val_and_grad(estimated_map2.apply_to(points), 10)
    _obj = lambda x, *args: _og(x.reshape(3, 3))[0]
    _obj_grad = lambda x, *args: _og(x.reshape(3, 3))[1].flatten()

    _mat = scp.optimize.minimize(
        _obj, np.identity(3).flatten(), jac=_obj_grad, method="CG"
    ).x.reshape(3, 3)
    return


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
