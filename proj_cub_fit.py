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
        print(eigenvecs)
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
        if len(vectors) == 2:
            return mat_from_eigenpairs(
                (0, vectors[0]),
                (0, vectors[1]),
                (1, np.cross(vectors[0], vectors[1])),
            )
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
        diag_1 = points[:, np.argmax(norms_1)]
        diags = [diag_1]  # could preallocate for dim_space vectors here
        # we want to / are able to determine n diagonals in total, but we already know one
        for _ in range(dim_space - 1):
            ortho_projector = proj_onto_orthogonal_complement(*diags)
            projected_points = ortho_projector @ centered_points
            norms = np.linalg.norm(projected_points, axis=0)
            new_diag = points[:, np.argmax(norms)]
            diags.append(new_diag)

        diags_mat = np.column_stack(diags)
        # This estimates the transformation by that matrix A that makes the diagonals A-orthonormal
        # and moreover maps the i-th diagonal to the i-th standard unit vector
        estimated_transformation_mat = np.linalg.inv(diags_mat)
        diagonal_orthonormalizer = AffineMap(estimated_transformation_mat, -center)
        # the 2D case would really be done at this point
        # and the >3 dimensional one needs a bit more thinking:
        # (maybe we can again pick the point of maximal norm in the transformed space as the final
        # diagonal and then construct connected vertices similar to 3D case)

        # black magic
        transformed_points = estimated_transformation_mat @ centered_points
        positive_first = transformed_points[0, :] > 0
        positive_second = transformed_points[1, positive_first] > 0
        positive_third = transformed_points[2, positive_first] > 0

        # this just computes 2D dot products of various subsets of the data projected down onto a 2D plane. It ignores "negative contributions"
        # assuming enough sampling the negative ones shouldn't matter but I'm not quite sure how it works out if there's not *that* many points to begin with
        p = transformed_points[1:, positive_first]
        print(p[:, np.argmax(np.linalg.norm(p, axis=0))])
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
        ]
        # construct "cubification" map
        transformed_edge_vectors = [
            d - transformed_diag_4 for d in transformed_diagonals[:3]
        ]
        edge_cubifier = np.linalg.inv(np.column_stack(transformed_edge_vectors))
        fitted_transformation_mat = edge_cubifier @ estimated_transformation_mat

        cubifier = AffineMap(
            fitted_transformation_mat, -edge_cubifier @ transformed_diag_4
        )
        return CubeFit(diagonal_orthonormalizer, cubifier)
    return CubeFit, proj_cube_fit


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## An example on generated data""")
    return


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
        [[2, 1, 2.85], [0, 3, 0.25], [100, 0, 1.5]]
    )
    # _center = np.mean(_p, axis=1)
    _p = _p
    points = transformation_mat @ _p  # + 0.01 * _rng.normal(size=_p.shape)
    points_unit = _p
    print(transformation_mat)
    return n_dims, points, points_unit, rot_mat, transformation_mat


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
def _(algebraic_residual, fit, go, points):
    _fig = go.Figure()
    _fig.update_layout(dict(title="The cubified data"))

    _points = fit.cubifier @ points
    # we "hollow out" the cube a bit to make the plot more performant
    _points = _points[:, algebraic_residual > 0.3]

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
def _(algebraic_residual, fit, go, points):
    _fig = go.Figure()
    _fig.update_layout(dict(title="Data with orthonormalized diagonals"))

    _points = fit.diagonal_orthonormalizer @ points
    # we "hollow out" the cube a bit to make the plot more performant
    _points = _points[:, algebraic_residual > 0.3]

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
def _(algebraic_residual, estimated_map, go, points):
    _fig = go.Figure()
    _fig.update_layout(dict(title="The cubified data"))

    _points = estimated_map @ points
    # we "hollow out" the cube a bit to make the plot more performant
    _points = _points[:, algebraic_residual > 0.3]

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
def _(fit, points, proj_cube_fit):
    estimated_map = fit.diagonal_normalizer
    # we can improve our estimate by iteration:
    estimated_map2 = proj_cube_fit(
        estimated_map.apply_to(points), points_centered=False
    ).cubifier.after(estimated_map)
    return estimated_map, estimated_map2


if __name__ == "__main__":
    app.run()
