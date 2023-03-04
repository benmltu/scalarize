#!/usr/bin/env python3

r"""
Scalarization functions for multi-objective optimization. The scalarized values are
designed such that larger values imply greater quality. The default settings are
designed for a maximization problem.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import pi

import torch

from scipy.special import gamma
from torch import Tensor
from torch.nn import Module


class ScalarizationFunction(Module, ABC):
    num_params: int

    def __init__(self) -> None:
        r"""Base constructor for scalarization functions."""
        super().__init__()

    @staticmethod
    @abstractmethod
    def evaluate(Y: Tensor, **kwargs) -> Tensor:
        r"""Evaluate the scalarization functions on a set of points.

        Args:
            Y: An `batch_shape x num_points x M`-dim Tensor containing the objective
                vectors.

        Returns:
            A `batch_shape x num_points x param_shape`-dim Tensor containing the
                scalarized objective vectors.
        """
        pass  # pragma: no cover

    @abstractmethod
    def forward(self, Y: Tensor) -> Tensor:
        r"""Evaluate the scalarization functions on a set of points.
        Args:
            Y: An `batch_shape x num_points x M`-dim Tensor containing the objective
                vectors.

        Returns:
            A `batch_shape x num_points x param_shape`-dim Tensor containing the
                scalarized objective vectors.
        """
        pass  # pragma: no cover


class LinearScalarization(ScalarizationFunction):
    r"""Linear scalarization function."""
    num_params = 1

    def __init__(self, weights: Tensor, negate: bool = False) -> None:
        r"""Linear scalarization function.

        Args:
            weights: A `batch_shape x num_weights x M`-dim Tensor of weights.
            negate: If True, we negate the scalarization function.
        """
        super().__init__()

        self.weights = weights
        self.negate = negate

    @staticmethod
    def evaluate(Y: Tensor, weights: Tensor, negate: bool = False) -> Tensor:
        r"""Computes the linear scalarization.

            If `negate=False`:
                s(Y) = sum(w * Y)
            Else:
                s(Y) = - sum(w * Y)

        Args:
            Y: An `batch_shape x num_points x M`-dim Tensor containing the objective
                vectors.
            weights: A `batch_shape x num_weights x M`-dim Tensor of weights.

        Returns:
            A `batch_shape x num_points x num_weights`-dim Tensor containing the
                scalarized objective vectors.
        """
        sign = -1.0 if negate else 1.0
        # `num_points x batch_shape x 1 x M`
        reshaped_Y = Y.movedim(-2, 0).unsqueeze(-2)
        # `batch_shape x num_points x num_weights`
        scalarized_Y = torch.sum(weights * reshaped_Y, dim=-1).movedim(0, -2)
        # `batch_shape x num_points x num_weights`
        return sign * scalarized_Y

    def forward(self, Y: Tensor) -> Tensor:
        r"""Computes the linear scalarization.

        Args:
            Y: An `batch_shape x num_points x M`-dim Tensor containing the objective
                vectors.

        Returns:
            A `batch_shape x num_points num_weights`-dim Tensor containing the
                scalarized objective vectors.
        """
        return self.evaluate(
            Y=Y,
            weights=self.weights,
            negate=self.negate,
        )


class LpScalarization(ScalarizationFunction):
    r"""Lp scalarization function."""
    num_params = 2

    def __init__(
        self,
        weights: Tensor,
        p: float,
        ref_points: Tensor,
        invert: bool = False,
        clip: bool = False,
        negate: bool = True,
    ) -> None:
        r"""Lp scalarization function.

        Args:
            weights: A `batch_shape x num_weights x M`-dim Tensor of weights.
            p: The power of the Lp norm.
            ref_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                reference points.
            invert: If True, the residual is defined as the difference from the
                reference point `y-r`, else the residual is defined as the difference
                towards the reference point `r-y`, where `y` is an objective vector
                and `r` is a reference point.
            clip: If True, we clip the residual at zero.
            negate: If True, we negate the distance.
        """
        super().__init__()

        self.weights = weights
        # TODO: check whether p >= 1
        self.p = p
        self.ref_points = ref_points
        self.invert = invert
        self.clip = clip
        self.negate = negate

    @staticmethod
    def evaluate(
        Y: Tensor,
        weights: Tensor,
        p: float,
        ref_points: Tensor,
        invert: bool = False,
        clip: bool = False,
        negate: bool = True,
    ) -> Tensor:
        r"""Computes the Lp scalarization or one of its variants.

            If `invert=False`:
                residual = r - Y
            Else:
                residual = Y - r

            If `clip = True`:
                residual = max(residual, 0)

            If `negate=False`:
                sign = 1
            Else:
                sign = -1

            s(Y) = sign * ||w * residual||_p

        Args:
            Y: An `batch_shape x num_points x M`-dim Tensor containing the objective
                vectors.
            weights: A `batch_shape x num_weights x M`-dim Tensor of weights.
            p: The power of the Lp norm.
            ref_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                reference points.
            invert: If True, the residual is defined as the difference from the
                reference point `y-r`, else the residual is defined as the difference
                towards the reference point `r-y`, where `y` is an objective vector
                and `r` is a reference point.
            clip: If True, we clip the residual at zero.
            negate: If True, we negate the distance.

        Returns:
            A `batch_shape x num_points x num_weights x num_ref`-dim Tensor
                containing the scalarized objective vectors.
        """
        sign = -1.0 if negate else 1.0
        diff_sign = -1.0 if invert else 1.0

        # `num_points x batch_shape x 1 x M`
        reshaped_Y = Y.movedim(-2, 0).unsqueeze(-2)

        # `num_points x batch_shape x 1 x num_ref x M`
        diff = diff_sign * (ref_points - reshaped_Y).unsqueeze(-3)

        if clip:
            diff = torch.clamp(diff, min=0)

        # `batch_shape x num_points x num_weights x num_ref x M`
        weighted_diff = (weights.unsqueeze(-2) * diff).movedim(0, -4)

        # `batch_shape x num_points x num_weights x num_ref`
        scalarized_Y = torch.norm(weighted_diff, p=p, dim=-1)

        # `batch_shape x num_points x num_weights x num_ref`
        return sign * scalarized_Y

    def forward(self, Y: Tensor) -> Tensor:
        r"""Computes the Lp scalarization or one of its variants.

        Args:
            Y: An `batch_shape x num_points x M`-dim Tensor containing the objective
                vectors.

        Returns:
            A `batch_shape x num_points x num_weights x num_ref`-dim Tensor
                containing the scalarized objective vectors.
        """
        return self.evaluate(
            Y=Y,
            weights=self.weights,
            p=self.p,
            ref_points=self.ref_points,
            invert=self.invert,
            clip=self.clip,
            negate=self.negate,
        )


class ChebyshevScalarization(ScalarizationFunction):
    r"""Chebyshev scalarization function."""
    num_params = 2

    def __init__(
        self,
        weights: Tensor,
        ref_points: Tensor,
        invert: bool = False,
        pseudo: bool = False,
        clip: bool = False,
        negate: bool = True,
    ) -> None:
        r"""Chebyshev scalarization function.

        Args:
            weights: A `batch_shape x num_weights x M`-dim Tensor of weights.
            ref_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                reference points.
            invert: If True, the residual is defined as the difference from the
                reference point `y-r`, else the residual is defined as the difference
                towards the reference point `r-y`, where `y` is an objective vector
                and `r` is a reference point.
            pseudo: If True, then we compute the pseudo Chebyshev scalarization
                function, which uses the minimum instead of the maximum.
            clip: If True, we clip the distance at zero.
            negate: If True, we negate the distance.
        """
        super().__init__()

        self.weights = weights
        self.ref_points = ref_points
        self.invert = invert
        self.pseudo = pseudo
        self.clip = clip
        self.negate = negate

    @staticmethod
    def evaluate(
        Y: Tensor,
        weights: Tensor,
        ref_points: Tensor,
        invert: bool = False,
        pseudo: bool = False,
        clip: bool = False,
        negate: bool = True,
    ) -> Tensor:
        r"""Computes the Chebyshev scalarization or one of its variants.

            If `invert=False`:
                residual = r - Y
            Else:
                residual = Y - r

            If `clip = True`:
                residual = max(residual, 0)

            If `negate=False`:
                sign = 1
            Else:
                sign = -1

            If `pseudo=False`:
                s(Y) = sign * max(w * residual)
            Else:
                s(Y) = sign * min(w * residual)

        Args:
            Y: An `batch_shape x num_points M`-dim Tensor containing the objective
                vectors.
            weights: A `batch_shape x num_weights x M`-dim Tensor of weights.
            ref_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                reference points.
            invert: If True, the residual is defined as the difference from the
                reference point `y-r`, else the residual is defined as the difference
                towards the reference point `r-y`, where `y` is an objective vector
                and `r` is a reference point.
            pseudo: If True, then we compute the pseudo Chebyshev scalarization
                function, which uses the minimum instead of the maximum.
            clip: If True, we clip the distance at zero.
            negate: If True, we negate the distance.

        Returns:
            A `batch_shape x num_points x num_weights x num_ref`-dim Tensor
                containing the scalarized objective vectors.
        """
        sign = -1.0 if negate else 1.0
        diff_sign = -1.0 if invert else 1.0

        # `num_points x batch_shape x 1 x M`
        reshaped_Y = Y.movedim(-2, 0).unsqueeze(-2)
        # `num_points x batch_shape x 1 x num_ref x M`
        diff = diff_sign * (ref_points - reshaped_Y).unsqueeze(-3)

        if clip:
            diff = torch.clamp(diff, min=0)

        # `batch_shape x num_points x num_weights x num_ref x M`
        weighted_diff = (weights.unsqueeze(-2) * diff).movedim(0, -4)

        # `batch_shape x num_points x num_weights x num_ref`
        if pseudo:
            scalarized_Y = torch.min(weighted_diff, dim=-1).values
        else:
            scalarized_Y = torch.max(weighted_diff, dim=-1).values

        # `batch_shape x num_points x num_weights x num_ref`
        return sign * scalarized_Y

    def forward(self, Y: Tensor) -> Tensor:
        r"""Computes the Chebyshev scalarization or one of its variants.

        Args:
            Y: An `batch_shape x num_points x M`-dim Tensor containing the objective
                vectors.

        Returns:
            A `batch_shape x num_points x num_weights x num_ref`-dim Tensor
                containing the scalarized objective vectors.
        """
        return self.evaluate(
            Y=Y,
            weights=self.weights,
            ref_points=self.ref_points,
            invert=self.invert,
            pseudo=self.pseudo,
            clip=self.clip,
            negate=self.negate,
        )


class LengthScalarization(ScalarizationFunction):
    r"""Length scalarization function."""
    num_params = 2

    def __init__(
        self,
        weights: Tensor,
        ref_points: Tensor,
        invert: bool = True,
        clip: bool = True,
    ) -> None:
        r"""Length scalarization function.

        Args:
            weights: A `batch_shape x num_weights x M`-dim Tensor of weights.
            ref_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                reference points.
            invert: If True, the residual is defined as the difference from the
                reference point `y-r`, else the residual is defined as the difference
                towards the reference point `r-y`, where `y` is an objective vector
                and `r` is a reference point.
            clip: If True, we clamp the length at zero.
        """
        super().__init__()

        self.weights = weights
        self.ref_points = ref_points
        self.invert = invert
        self.clip = clip

    @staticmethod
    def evaluate(
        Y: Tensor,
        weights: Tensor,
        ref_points: Tensor,
        invert: bool = True,
        clip: bool = True,
    ) -> Tensor:
        r"""Computes the length scalarization.

            If `invert=False`:
                residual = r - Y
            Else:
                residual = Y - r

            If `clip = True`:
                residual = max(residual, 0)

            s(Y) = min(residual / w)

        Args:
            Y: An `batch_shape x num_points x M`-dim Tensor containing the objective
                vectors.
            weights: A `batch_shape x num_weights x M`-dim Tensor of weights.
            ref_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                reference points.
            invert: If True, the residual is defined as the difference from the
                reference point `y-r`, else the residual is defined as the difference
                towards the reference point `r-y`, where `y` is an objective vector
                and `r` is a reference point.
            clip: If True, we clamp the length at zero.

        Returns:
            A `batch_shape x num_points x num_weights x num_ref`-dim Tensor
                containing the scalarized objective vectors.
        """
        # TODO: need to be careful about dividing by zero?
        scalarized_Y = ChebyshevScalarization.evaluate(
            Y=Y,
            weights=1 / weights,
            ref_points=ref_points,
            invert=invert,
            pseudo=True,
            clip=clip,
            negate=False,
        )

        # `batch_shape x num_points x num_weights x num_ref`
        return scalarized_Y

    def forward(self, Y: Tensor) -> Tensor:
        r"""Computes the length scalarization.

        Args:
            Y: An `batch_shape x num_points x M`-dim Tensor containing the objective
                vectors.

        Returns:
            A `batch_shape x num_points x num_weights x num_ref`-dim Tensor
                containing the scalarized objective vectors.
        """
        return self.evaluate(
            Y=Y,
            weights=self.weights,
            ref_points=self.ref_points,
            invert=self.invert,
            clip=self.clip,
        )


class HypervolumeScalarization(ScalarizationFunction):
    r"""Hypervolume scalarization function."""
    num_params = 2

    def __init__(
        self,
        weights: Tensor,
        ref_points: Tensor,
        invert: bool = True,
        clip: bool = True,
    ) -> None:
        r"""Hypervolume scalarization function.

        Args:
            weights: A `batch_shape x num_weights x M`-dim Tensor of weights.
            ref_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                reference points.
            invert: If True, the residual is defined as `y-r`, else the residual is
                defined as `r-y`, where `y` is an objective vector and `r` is
                a reference point.
            clip: If True, we clamp the values at zero.
        """
        super().__init__()

        self.weights = weights
        self.ref_points = ref_points
        self.invert = invert
        self.clip = clip

    @staticmethod
    def evaluate(
        Y: Tensor,
        weights: Tensor,
        ref_points: Tensor,
        invert: bool = True,
        clip: bool = True,
    ) -> Tensor:
        r"""Computes the hypervolume scalarization.

            c = π^(M/2) / (2^M Γ(M/2 + 1))

            If `invert=False`:
                residual = r - Y
            Else:
                residual = Y - r

            If `clip = True`:
                residual = max(residual, 0)

            s(Y) = c g(min(residual / w))

            where g(z, M) = sign(z) * abs(z)^M.

        Args:
            Y: An `batch_shape x num_points x M`-dim Tensor containing the objective
                vectors.
            weights: A `batch_shape x num_weights x M`-dim Tensor of weights.
            ref_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                reference points.
            invert: If True, the residual is defined as the difference from the
                reference point `y-r`, else the residual is defined as the difference
                towards the reference point `r-y`, where `y` is an objective vector
                and `r` is a reference point.
            clip: If True, we clamp the values at zero.

        Returns:
            A `batch_shape x num_points x num_weights x num_ref`-dim Tensor
                containing the scalarized objective vectors.
        """
        length = LengthScalarization.evaluate(
            Y=Y, weights=weights, ref_points=ref_points, invert=invert, clip=clip
        )
        M = weights.shape[-1]
        # We multiply the constant before taking the power for numerical stability.
        constant = pi ** (1 / 2) / (2 * gamma(M / 2 + 1) ** (1 / M))

        if clip:
            return torch.pow(constant * length, M)
        else:
            return torch.sign(length) * torch.pow(constant * abs(length), M)

    def forward(self, Y: Tensor) -> Tensor:
        r"""Computes the hypervolume scalarization.

        Args:
            Y: An `batch_shape x num_points x M`-dim Tensor containing the objective
                vectors.

        Returns:
            A `batch_shape x num_points x num_weights x num_ref`-dim Tensor
                containing the scalarized objective vectors.
        """
        return self.evaluate(
            Y=Y,
            weights=self.weights,
            ref_points=self.ref_points,
            invert=self.invert,
            clip=self.clip,
        )


class AugmentedChebyshevScalarization(ScalarizationFunction):
    r"""Augmented Chebyshev scalarization function."""
    num_params = 2

    def __init__(
        self,
        weights: Tensor,
        ref_points: Tensor,
        beta: float = 5e-2,
        invert: bool = False,
        pseudo: bool = False,
        negate: bool = True,
    ) -> None:
        r"""Augmented Chebyshev scalarization function.

        Args:
            weights: A `batch_shape x num_weights x M`-dim Tensor of weights.
            ref_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                reference points.
            beta: The trade-off parameter controlling the L1 penalty.
            invert: If True we compute with respect to the nadir, else we compute
                with respect to the utopia.
            pseudo: If True, we compute the pseudo Chebyshev scalarization
                function, which uses the minimum instead of the maximum.
            negate: If True, we negate the distance.
        """
        super().__init__()

        self.weights = weights
        self.ref_points = ref_points
        self.beta = beta
        self.invert = invert
        self.pseudo = pseudo
        self.negate = negate

    @staticmethod
    def evaluate(
        Y: Tensor,
        weights: Tensor,
        ref_points: Tensor,
        beta: float = 5e-2,
        invert: bool = False,
        pseudo: bool = False,
        negate: bool = True,
    ) -> Tensor:
        r"""Computes the augmented Chebyshev scalarization or one of its variants.

            If `invert=False`:
                residual = r - Y
            Else:
                residual = Y - r

            If `negate=False`:
                sign = 1
            Else:
                sign = -1

            If `pseudo=False`:
                s(Y) = sign * max(w * residual) + sign * beta * sum(w * residual)
            Else:
                s(Y) = sign * min(w * residual) + sign * beta * sum(w * residual)

        Args:
            Y: An `batch_shape x num_points x M`-dim Tensor containing the objective
                vectors.
            weights: A `batch_shape x num_weights x M`-dim Tensor of weights.
            ref_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                reference points.
            beta: The trade-off parameter controlling the L1 penalty.
            invert: If True we compute with respect to the nadir, else we compute
                with respect to the utopia.
            pseudo: If True, we compute the pseudo Chebyshev scalarization function,
                which uses the minimum instead of the maximum.
            negate: If True, we negate the distance.

        Returns:
            A `batch_shape x num_points x num_weights x num_ref`-dim Tensor
                containing the scalarized objective vectors.
        """
        sign = -1.0 if negate else 1.0
        diff_sign = -1.0 if invert else 1.0

        # `num_points x batch_shape x 1 x M`
        reshaped_Y = Y.movedim(-2, 0).unsqueeze(-2)
        # `num_points x batch_shape x 1 x num_ref x M`
        diff = diff_sign * (ref_points - reshaped_Y).unsqueeze(-3)
        # `batch_shape x num_points x num_weights x num_ref x M`
        weighted_diff = (weights.unsqueeze(-2) * diff).movedim(0, -4)
        # `num_points x batch_shape x num_weights x num_ref`
        penalty = torch.sum(weighted_diff, dim=-1)

        # `batch_shape x num_points x num_weights x num_ref`
        if pseudo:
            scalarized_Y = torch.min(weighted_diff, dim=-1).values + beta * penalty
        else:
            scalarized_Y = torch.max(weighted_diff, dim=-1).values + beta * penalty

        # `batch_shape x num_points x num_weights x num_ref`
        return sign * scalarized_Y

    def forward(self, Y: Tensor) -> Tensor:
        r"""Computes the augmented Chebyshev scalarization or one of its variants.

        Args:
            Y: An `batch_shape x num_points x M`-dim Tensor containing the objective
                vectors.

        Returns:
            A `batch_shape x num_points x num_weights x num_ref`-dim Tensor
                containing the scalarized objective vectors.
        """
        return self.evaluate(
            Y=Y,
            weights=self.weights,
            ref_points=self.ref_points,
            beta=self.beta,
            invert=self.invert,
            pseudo=self.pseudo,
            negate=self.negate,
        )


class ModifiedChebyshevScalarization(ScalarizationFunction):
    r"""Modified Chebyshev scalarization function."""
    num_params = 2

    def __init__(
        self,
        weights: Tensor,
        ref_points: Tensor,
        beta: float = 5e-2,
        invert: bool = False,
        pseudo: bool = False,
        negate: bool = True,
    ) -> None:
        r"""Modified Chebyshev scalarization function.

        Args:
            weights: A `batch_shape x num_weights x M`-dim Tensor of weights.
            ref_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                reference points.
            beta: The trade-off parameter controlling the L1 penalty.
            invert: If True we compute with respect to the nadir, else we compute
                with respect to the utopia.
            pseudo: If True, we compute the pseudo Chebyshev scalarization
                function, which uses the minimum instead of the maximum.
            negate: If True, we negate the distance.
        """
        super().__init__()

        self.weights = weights
        self.ref_points = ref_points
        self.beta = beta
        self.invert = invert
        self.pseudo = pseudo
        self.negate = negate

    @staticmethod
    def evaluate(
        Y: Tensor,
        weights: Tensor,
        ref_points: Tensor,
        beta: float = 5e-2,
        invert: bool = False,
        pseudo: bool = False,
        negate: bool = True,
    ) -> Tensor:
        r"""Computes the modified Chebyshev scalarization function or one of its
        variants.

            If `invert=False`:
                residual = r - Y
            Else:
                residual = Y - r

            If `negate=False`:
                sign = 1
            Else:
                sign = -1

            If `pseudo=False`:
                s(Y) = sign * max(w * residual + sign * beta * sum(w * residual))
            Else:
                s(Y) = sign * min(w * residual + sign * beta * sum(w * residual))

        Args:
            Y: An `batch_shape x num_points x M`-dim Tensor containing the objective
                vectors.
            weights: A `batch_shape x num_weights x M`-dim Tensor of weights.
            ref_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                reference points.
            beta: The trade-off parameter controlling the L1 penalty.
            invert: If True we compute with respect to the nadir, else we compute
                with respect to the utopia.
            pseudo: If True, we compute the pseudo Chebyshev scalarization
                function, which uses the minimum instead of the maximum.
            negate: If True, we negate the distance.

        Returns:
            A `batch_shape x num_points x num_weights x num_ref`-dim Tensor
                containing the scalarized objective vectors.
        """
        sign = -1.0 if negate else 1.0
        diff_sign = 1.0 if invert else -1.0
        # `num_points x batch_shape x 1 x M`
        reshaped_Y = Y.movedim(-2, 0).unsqueeze(-2)
        # `num_points x batch_shape x 1 x num_ref x M`
        diff = diff_sign * (reshaped_Y - ref_points).unsqueeze(-3)
        # `batch_shape x num_points x num_weights x num_ref x M`
        weighted_diff = (weights.unsqueeze(-2) * diff).movedim(0, -4)
        penalty = sign * torch.sum(weighted_diff, dim=-1, keepdims=True)

        # `batch_shape x num_points x num_weights x num_ref`
        if pseudo:
            scalarized_Y = torch.min(weighted_diff + beta * penalty, dim=-1).values
        else:
            scalarized_Y = torch.max(weighted_diff + beta * penalty, dim=-1).values

        return sign * scalarized_Y

    def forward(self, Y: Tensor) -> Tensor:
        r"""Computes the modified Chebyshev scalarization function.

        Args:
            Y: An `batch_shape x num_points x M`-dim Tensor containing the objective
                vectors.

        Returns:
            A `batch_shape x num_points x num_weights x num_ref`-dim Tensor
                containing the scalarized objective vectors.
        """
        return self.evaluate(
            Y=Y,
            weights=self.weights,
            ref_points=self.ref_points,
            beta=self.beta,
            invert=self.invert,
            pseudo=self.pseudo,
            negate=self.negate,
        )


class PBIScalarization(ScalarizationFunction):
    r"""Penalty boundary intersection scalarization function."""
    num_params = 2

    def __init__(
        self,
        weights: Tensor,
        ref_points: Tensor,
        beta: float = 5,
        invert: bool = False,
        negate: bool = True,
    ) -> None:
        r"""Penalty boundary intersection scalarization function.

        Args:
            weights: A `batch_shape x num_weights x M`-dim Tensor of weights.
            ref_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                reference points.
            beta: The trade-off parameter controlling the diversity penalty.
            invert: If True, the residual is defined as the difference from the
                reference point `y-r`, else the residual is defined as the difference
                towards the reference point `r-y`, where `y` is an objective vector
                and `r` is a reference point.
            negate: If True, we negate the convergence term.
        """
        super().__init__()

        self.weights = weights
        self.ref_points = ref_points
        self.beta = beta
        self.invert = invert
        self.negate = negate

    @staticmethod
    def evaluate(
        Y: Tensor,
        weights: Tensor,
        ref_points: Tensor,
        beta: float = 5,
        invert: bool = False,
        negate: bool = True,
    ) -> Tensor:
        r"""Computes the penalty boundary intersection scalarization or inverted
        penalty boundary intersection scalarization.

            If `invert=False`:
                residual = r - Y
            Else:
                residual = Y - r

            If `negate=False`:
                sign = 1
            Else:
                sign = -1

            convergence(Y) = |sum(w * residual)|
            diversity(Y) = ||residual + convergence(Y)w||_2

            s(Y) = sign * convergence(Y) - beta * diversity(Y)

            The sign of the convergence term depends on the optimization problem
            and the choice of reference point. For a maximization problem, the
            standard PBI uses the utopia reference point and sets `invert=False` and
            `negate=False`. Whereas, the standard inverted PBI uses the nadir
            reference point and sets `invert=True` and `negate=True`.

        Args:
            Y: An `batch_shape x num_points x M`-dim Tensor containing the objective
                vectors.
            weights: A `batch_shape x num_weights x M`-dim Tensor of weights.
            ref_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                reference points.
            beta: The trade-off parameter controlling the diversity penalty.
            invert: If True we compute with respect to the nadir, else we compute
                with respect to the utopia.
            negate: If True, we negate the convergence term.

        Returns:
            A `batch_shape x num_points x num_weights x num_ref`-dim Tensor
                containing the scalarized objectives.
        """
        sign = -1.0 if negate else 1.0
        diff_sign = -1.0 if invert else 1.0
        # `num_points x batch_shape x 1 x M`
        reshaped_Y = Y.movedim(-2, 0).unsqueeze(-2)
        # `num_points x batch_shape x 1 x num_ref x M`
        diff = diff_sign * (ref_points - reshaped_Y).unsqueeze(-3)
        # `num_points x batch_shape x num_weights x num_ref`
        convergence = torch.abs(torch.sum(weights.unsqueeze(-2) * diff, dim=-1))
        # `num_points x batch_shape x num_weights x num_ref x M`
        point = ref_points.unsqueeze(-3) + sign * convergence.unsqueeze(
            -1
        ) * weights.unsqueeze(-2)

        # `num_points x batch_shape x num_weights x num_ref x M`
        projected_diff = reshaped_Y.unsqueeze(-2) - point

        # `num_points x batch_shape x num_weights x num_ref`
        diversity = torch.norm(projected_diff, p=2, dim=-1)

        # `batch_shape x num_points x num_weights x num_ref`
        scalarized_Y = (sign * convergence - beta * diversity).movedim(0, -3)

        # `batch_shape x num_points x num_weights x num_ref`
        return scalarized_Y

    def forward(self, Y: Tensor) -> Tensor:
        r"""Computes the penalty boundary intersection scalarization or inverted
        penalty boundary intersection scalarization.

        Args:
            Y: An `batch_shape x num_points x M`-dim Tensor containing the objective
                vectors.

        Returns:
            A `batch_shape x num_points x num_weights x num_ref`-dim Tensor
                containing the scalarized objectives.
        """
        return self.evaluate(
            Y=Y,
            weights=self.weights,
            ref_points=self.ref_points,
            beta=self.beta,
            invert=self.invert,
            negate=self.negate,
        )


class KSScalarization(ScalarizationFunction):
    r"""Kalai-Smorodinsky scalarization."""
    num_params = 1

    def __init__(
        self,
        utopia_points: Tensor,
        nadir_points: Tensor,
        maximize: bool = True,
    ) -> None:
        r"""Kalai-Smorodinsky scalarization.

        Args:
            utopia_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                utopia points, which are the best possible points.
            nadir_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                nadir points, which are the worst possible points.
            maximize: If True, we consider the maximization problem.
        """
        super().__init__()

        self.utopia_points = utopia_points
        self.nadir_points = nadir_points
        self.maximize = maximize

    @staticmethod
    def evaluate(
        Y: Tensor,
        utopia_points: Tensor,
        nadir_points: Tensor,
        maximize: bool = True,
    ) -> Tensor:
        r"""Computes the Kalai-Smorodinsky scalarization.

            If `maximize=True`:
                s(Y) = min((y - nadir) / (utopia - nadir))
            Else:
                s(Y) = min((nadir - y) / (nadir - utopia))

        Args:
            Y: An `batch_shape x num_points x M`-dim Tensor containing the objective
                vectors, which are the best possible points.
            utopia_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                utopia points.
            nadir_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                nadir points, which are the worst possible points.
            maximize: If True, we consider the maximization problem.

        Returns:
            A `batch_shape x num_ref`-dim Tensor containing the scalarized objectives.
        """
        sign = 1.0 if maximize else -1.0

        # `num_points x batch_shape x num_ref x M`
        obj_range = sign * (utopia_points - nadir_points)

        # `num_points x batch_shape x num_ref x num_ref`
        sY = ChebyshevScalarization.evaluate(
            Y=Y,
            weights=1 / obj_range,
            ref_points=nadir_points,
            invert=maximize,
            pseudo=True,
            negate=False,
        )

        return sY[..., 0]

    def forward(self, Y: Tensor) -> Tensor:
        r"""Computes the Kalai-Smorodinsky scalarization.

        Args:
            Y: An `batch_shape x num_points x M`-dim Tensor containing the objective
                vectors.

        Returns:
            A `batch_shape x num_points x num_ref`-dim Tensor containing the
                scalarized objectives.
        """
        return self.evaluate(
            Y=Y,
            utopia_points=self.utopia_points,
            nadir_points=self.nadir_points,
            maximize=self.maximize,
        )
