"""
This is a somewhat "clean" version of the code from the [proj_parallelotope_fit.py] notebook with an additional note on
extending the algorithm to not necessarily regular parallelotopes in R^n.
"""

import numpy as np
from typing import NamedTuple


class AffineMap(NamedTuple):
    """Represents an affine transformation x \mapsto Ax + b on R^(n)
    via its associated linear transformation A and offset b.
    """

    linear_transf: np.ndarray
    offset: np.ndarray

    def apply_to(self, points):
        _dim_codom = self.linear_transf.shape[0]
        return self.linear_transf @ points + self.offset.reshape(_dim_codom, 1)

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
            self.projected_sq_norms -= np.einsum("i,ij", proj_unit, self.points) ** 2
        return new_proj


class CubeFit(NamedTuple):
    # remaps the parallelotope in R^d such that d of its diagonals are orthonormal post-transformation (and the center is at the origin).
    diagonal_orthonormalizer: AffineMap
    # maps the parallelotope into the unit cube [0,1]^d.
    cubifier: AffineMap


def proj_cube_fit(points: np.ndarray, points_centered: bool = False) -> AffineMap:
    """
    Args:
    * points: Array of shape (d,n) where n is the number of points and d the dimension of space
    * points_centered: whether the inputs points may be assumed to be centered around the origin.
    """
    dim_space = points.shape[0]
    if dim_space != 3:
        raise NotImplementedError(
            "Only three-dimensional case is implemented."
            "For 2D embed your points into 3D in some way, for R^n I don't have a solution yet."
        )
    # center points as needed
    if points_centered:
        center = np.zeros(dim_space)
        centered_points = points.copy()
    else:
        center = np.mean(points, axis=1)
        centered_points = points - center.reshape(dim_space, -1)
    centralizer = AffineMap(np.identity(dim_space), -center)

    # determine d diagonals of the parallelotope ordered by their lengths
    norms_1 = np.linalg.norm(centered_points, axis=0)
    diag_1 = centered_points[:, np.argmax(norms_1)]
    diags = [diag_1]  # could preallocate for dim_space vectors here
    proj = UpdatableOrthogonalProjectionWithNormUpdate(dim_space, centered_points)
    # we want to / are able to determine n diagonals in total, but we already know one
    for _ in range(dim_space - 1):
        _ortho_projector = proj.add_vector_to_generators(diags[-1])
        max_idx = np.argmax(proj.projected_sq_norms)
        # if we're looking for parallelotopes  P = A [0,1]^n + b with A singular, we want to check
        # whether the maximum of the projected norms is (close to) zero and if it is break out of
        # the loop: at that point we've determined all diagonals of the parallelotope and can construct the desired map
        if np.isclose(proj.projected_sq_norms[max_idx], 0.0):
            # in this case we're dealing with a singular parallelotope and can directly construct
            # the desired map
            if (
                len(diags) == 1
            ):  # we have a line segment, the linesegment is [-d,d] where d is the diagonal we found
                d = diags[0]
                norm_d = np.linalg.norm(d)
                d_unit = d / norm_d
                line_length = 2 * norm_d
                cubifier = AffineMap(
                    1 / line_length * d_unit.reshape((1, -1)), 1 / 2
                ).after(centralizer)
                return CubeFit(None, cubifier)
            elif len(diags) == 2:
                # we have a parallelogram
                mat = np.array([[1, 1], [1, -1]]) / 2
                cubifier = AffineMap(
                    mat.T @ np.linalg.pinv(np.column_stack(diags)), np.ones(2) / 2
                ).after(centralizer)
                return CubeFit(None, cubifier)
            else:
                # same as for len == 2, just more general
                # Note for dim_codom >= 3:
                # This is more of a "best effort", I haven't tested it or thought it through in that much detail,
                # it's well possible that this assumes an incorrect topology between the vertices (it's probably
                # possible to find the correct topology somewhat heuristically: replace a subset of the diagonals
                # 2 through k by their reflected version and run the computation below. There's 2^(n-1) possible
                # choices for such subsets and one of them corresponds to the correct topology. "Just" compute
                # all of these (good luck in high dimensions) and somehow check which one is "best" --- maybe by
                # comparing bounding boxes and checking which one is closest to the cube under some suitable metric)
                # As for what this thing below actually computes: letting the diagonals be (v, w_1, ..., w_k) it
                # determines an affine map T(x) = Ax + b such that:
                #     T(-v) = 0, T(v) = (1,...,1),
                # and moreover for all i
                #     T(w_i) = e_i and T(-w_i) = "unit cube vertex opposite of e_i" = (1,...,1) - e_i
                # so it picks the first diagonal as "main diagonal" and then maps all the other ones to diagonals of the
                # unit cube in an arbitrary way. In 2D this "arbitraryness" doesn't matter since there's actually only
                # one option (or rather: getting it wrong simply flips the square along the "primary" diagonal), but in
                # higher dimensions it causes the above-mentioned problem.
                # Note that the surrounding if-statement here could actually be collapsed: the last branch is equivalent
                # to the first two ones
                dim_codom = len(diags)
                diags_mat = np.column_stack(diags)
                diagonal_orthonormalizer = np.linalg.pinv(diags_mat)
                b = np.ones(dim_codom) / 2
                eye = np.identity(dim_codom)
                mat = np.column_stack([b, *(e_i - b for e_i in eye[1:, 0:])])
                cubifier = AffineMap(mat @ diagonal_orthonormalizer, b).after(
                    centralizer
                )
                return CubeFit(
                    AffineMap(diagonal_orthonormalizer, np.zeros(dim_codom)).after(
                        centralizer
                    ),
                    cubifier,
                )
        else:
            new_diag = centered_points[:, max_idx]
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
    ).after(centralizer)
    return CubeFit(diagonal_orthonormalizer, cubifier)
