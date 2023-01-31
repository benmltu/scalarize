#!/usr/bin/env python3

r"""
Scalarization functions for a maximization problem.
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
    num_params = 2

    def __init__(
        self,
        weights: Tensor,
        ref_points: Tensor,
        invert: bool = False,
        clip: bool = False,
    ) -> None:
        r"""Linear scalarization function.

        Args:
            weights: A `batch_shape x num_weights x M`-dim Tensor of weights.
            ref_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                reference points.
            invert: If True, the reference point is assumed to be the nadir, else we
                assume it is the utopia.
            clip: If True, we clip the distance at zero.
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
        invert: bool = False,
        clip: bool = False,
    ) -> Tensor:
        r"""Computes the linear scalarization. For the linear scalarization, the
        reference points can be set arbitrarily.

            If `invert=False`:
                If `clip=False`:
                    s(Y) = - sum(w * (r - Y))
                Else:
                    s(Y) = - sum(w * max(r - Y, 0))

            If `invert=True`:
                If `clip=False`:
                    s(Y) = sum(w * (Y - r))
                Else:
                    s(Y) = sum(w * max(Y - r, 0))

        Args:
            Y: An `batch_shape x num_points x M`-dim Tensor containing the objective
                vectors.
            weights: A `batch_shape x num_weights x M`-dim Tensor of weights.
            ref_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                reference points.
            invert: If True, the reference point is assumed to be the nadir, else we
                assume it is the utopia.
            clip: If True, we clip the distance at zero.

        Returns:
            A `batch_shape x num_points x num_weights x num_ref`-dim Tensor
                containing the scalarized objective vectors.
        """
        sign = 1.0 if invert else -1.0
        # `num_points x batch_shape x 1 x M`
        reshaped_Y = Y.movedim(-2, 0).unsqueeze(-2)
        # `num_points x batch_shape x 1 x num_ref x M`
        diff = sign * (reshaped_Y - ref_points).unsqueeze(-3)

        if clip:
            diff = torch.clamp(diff, min=0)

        # `num_points x batch_shape x num_weights x num_ref`
        scalarized_Y = sign * torch.sum(weights.unsqueeze(-2) * diff, dim=-1)
        # `batch_shape x num_points x num_weights x num_ref`
        return scalarized_Y.movedim(0, -3)

    def forward(self, Y: Tensor) -> Tensor:
        r"""Computes the linear scalarization. For the linear scalarization, the
        reference points can be set arbitrarily.

        Args:
            Y: An `batch_shape x num_points x M`-dim Tensor containing the objective
                vectors.

        Returns:
            A `batch_shape x num_points num_weights x num_ref`-dim Tensor containing
                the scalarized objective vectors.
        """
        return self.evaluate(
            Y=Y,
            weights=self.weights,
            ref_points=self.ref_points,
            invert=self.invert,
            clip=self.clip,
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
    ) -> None:
        r"""Lp scalarization function.

        Args:
            weights: A `batch_shape x num_weights x M`-dim Tensor of weights.
            p: The power of the Lp norm.
            ref_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                reference points.
            invert: If True, the reference point is assumed to be the nadir, else we
                assume it is the utopia.
            clip: If True, we clip the distance at zero.
        """
        super().__init__()

        self.weights = weights
        # TODO: check whether p >= 1
        self.p = p
        self.ref_points = ref_points
        self.invert = invert
        self.clip = clip

    @staticmethod
    def evaluate(
        Y: Tensor,
        weights: Tensor,
        p: float,
        ref_points: Tensor,
        invert: bool = False,
        clip: bool = False,
    ) -> Tensor:
        r"""Computes the Lp scalarization.

            If `invert=False`:
                If `clip=False`:
                    s(Y) = - ||w * (r - Y)||_p
                Else:
                    s(Y) = - ||w * max(r - Y, 0)||_p

            If `invert=True`:
                If `clip=False`:
                    s(Y) = ||w * (Y - r)||_p
                Else:
                    s(Y) = ||w * max(Y - r, 0)||_p

        Args:
            Y: An `batch_shape x num_points x M`-dim Tensor containing the objective
                vectors.
            weights: A `batch_shape x num_weights x M`-dim Tensor of weights.
            p: The power of the Lp norm.
            ref_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                reference points.
            invert: If True, the reference point is assumed to be the nadir, else we
                assume it is the utopia.
            clip: If True, we clip the distance at zero.

        Returns:
            A `batch_shape x num_points x num_weights x num_ref`-dim Tensor
                containing the scalarized objective vectors.
        """
        sign = 1.0 if invert else -1.0
        # `num_points x batch_shape x 1 x M`
        reshaped_Y = Y.movedim(-2, 0).unsqueeze(-2)

        # `num_points x batch_shape x 1 x num_ref x M`
        diff = sign * (reshaped_Y - ref_points).unsqueeze(-3)

        if clip:
            diff = torch.clamp(diff, min=0)

        # `num_points x batch_shape x num_weights x num_ref`
        scalarized_Y = sign * torch.norm(weights.unsqueeze(-2) * diff, p=p, dim=-1)

        # `batch_shape x num_points x num_weights x num_ref`
        return scalarized_Y.movedim(0, -3)

    def forward(self, Y: Tensor) -> Tensor:
        r"""Computes the Lp scalarization.

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
    ) -> None:
        r"""Chebyshev scalarization function.

        Args:
            weights: A `batch_shape x num_weights x M`-dim Tensor of weights.
            ref_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                reference points.
            invert: If True, the reference point is assumed to be the nadir, else we
                assume it is the utopia.
            pseudo: If True, then we compute the pseudo Chebyshev scalarization
                function, which uses the minimum instead of the maximum.
            clip: If True, we clip the distance at zero.
        """
        super().__init__()

        self.weights = weights
        self.ref_points = ref_points
        self.invert = invert
        self.pseudo = pseudo
        self.clip = clip

    @staticmethod
    def evaluate(
        Y: Tensor,
        weights: Tensor,
        ref_points: Tensor,
        invert: bool = False,
        pseudo: bool = False,
        clip: bool = False,
    ) -> Tensor:
        r"""Computes the Chebyshev scalarization.

            If `invert=False` and `pseudo=True`:
                If `clip=False`:
                    s(Y) = - min(w * (r - Y))
                Else:
                    s(Y) = - min(w * max(r - Y, 0))

            If `invert=False` and `pseudo=False`:
                If `clip=False`:
                    s(Y) = - max(w * (r - Y))
                Else:
                    s(Y) = - max(w * max(r - Y, 0))

            If `invert=True` and `pseudo=True`:
                If `clip=False`:
                    s(Y) = min(w * (Y - r))
                Else:
                    s(Y) = min(w * max(Y - r, 0))

            If `invert=True` and `pseudo=False`:
                If `clip=False`:
                    s(Y) = max(w * (Y - r))
                Else:
                    s(Y) = max(w * max(Y - r, 0))
        Args:
            Y: An `batch_shape x num_points M`-dim Tensor containing the objective
                vectors.
            weights: A `batch_shape x num_weights x M`-dim Tensor of weights.
            ref_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                reference points.
            invert: If True, the reference point is assumed to be the nadir, else we
                assume it is the utopia.
            pseudo: If True, then we compute the pseudo Chebyshev scalarization
                function, which uses the minimum instead of the maximum.
            clip: If True, we clip the distance at zero.

        Returns:
            A `batch_shape x num_points x num_weights x num_ref`-dim Tensor
                containing the scalarized objective vectors.
        """
        sign = 1.0 if invert else -1.0
        # `num_points x batch_shape x 1 x M`
        reshaped_Y = Y.movedim(-2, 0).unsqueeze(-2)
        # `num_points x batch_shape x 1 x num_ref x M`
        diff = sign * (reshaped_Y - ref_points).unsqueeze(-3)

        if clip:
            diff = torch.clamp(diff, min=0)

        # `num_points x batch_shape x num_weights x num_ref`
        if invert:
            if pseudo:
                scalarized_Y = torch.min(weights.unsqueeze(-2) * diff, dim=-1).values
            else:
                scalarized_Y = torch.max(weights.unsqueeze(-2) * diff, dim=-1).values
        else:
            if pseudo:
                scalarized_Y = -torch.min(weights.unsqueeze(-2) * diff, dim=-1).values
            else:
                scalarized_Y = -torch.max(weights.unsqueeze(-2) * diff, dim=-1).values

        # `batch_shape x num_points x num_weights x num_ref`
        return scalarized_Y.movedim(0, -3)

    def forward(self, Y: Tensor) -> Tensor:
        r"""Computes the Chebyshev scalarization.

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
        )


class LengthScalarization(ScalarizationFunction):
    r"""Length scalarization function."""
    num_params = 2

    def __init__(
        self,
        weights: Tensor,
        ref_points: Tensor,
    ) -> None:
        r"""Length scalarization function.

        Args:
            weights: A `batch_shape x num_weights x M`-dim Tensor of weights.
            ref_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                reference points.
        """
        super().__init__()

        self.weights = weights
        self.ref_points = ref_points

    @staticmethod
    def evaluate(
        Y: Tensor,
        weights: Tensor,
        ref_points: Tensor,
    ) -> Tensor:
        r"""Computes the length scalarization.

            s(Y) = min(max(0, (Y - r) / w))

        Args:
            Y: An `batch_shape x num_points x M`-dim Tensor containing the objective
                vectors.
            weights: A `batch_shape x num_weights x M`-dim Tensor of weights.
            ref_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                reference points.

        Returns:
            A `batch_shape x num_points x num_weights x num_ref`-dim Tensor
                containing the scalarized objective vectors.
        """
        scalarized_Y = ChebyshevScalarization.evaluate(
            Y=Y,
            weights=1 / weights,
            ref_points=ref_points,
            invert=True,
            pseudo=True,
            clip=True,
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
        )


class HypervolumeScalarization(ScalarizationFunction):
    r"""Hypervolume scalarization function."""
    num_params = 2

    def __init__(self, weights: Tensor, ref_points: Tensor) -> None:
        r"""Hypervolume scalarization function.

        Args:
            weights: A `batch_shape x num_weights x M`-dim Tensor of weights.
            ref_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                reference points.
        """
        super().__init__()

        self.weights = weights
        self.ref_points = ref_points

    @staticmethod
    def evaluate(Y: Tensor, weights: Tensor, ref_points: Tensor) -> Tensor:
        r"""Computes the hypervolume scalarization.

            c = π^(M/2) / (2^M Γ(M/2 + 1))
            s(Y) = c min(max(0, (Y - r) / w)^M)

        Args:
            Y: An `batch_shape x num_points x M`-dim Tensor containing the objective
                vectors.
            weights: A `batch_shape x num_weights x M`-dim Tensor of weights.
            ref_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                reference points.

        Returns:
            A `batch_shape x num_points x num_weights x num_ref`-dim Tensor
                containing the scalarized objective vectors.
        """
        length = LengthScalarization.evaluate(
            Y=Y, weights=weights, ref_points=ref_points
        )
        M = weights.shape[-1]
        # TODO: might want to cache this value
        constant = pi ** (M / 2) / (2**M * gamma(M / 2 + 1))

        return constant * torch.pow(length, M)

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
        """
        super().__init__()

        self.weights = weights
        self.ref_points = ref_points
        self.beta = beta
        self.invert = invert
        self.pseudo = pseudo

    @staticmethod
    def evaluate(
        Y: Tensor,
        weights: Tensor,
        ref_points: Tensor,
        beta: float = 5e-2,
        invert: bool = False,
        pseudo: bool = False,
    ) -> Tensor:
        r"""Computes the augmented Chebyshev scalarization.

            If `invert=False` and `pseudo=True`:
                s(Y) = - min(w * (r - Y)) - beta * sum(w * (r - Y))

            If `invert=False` and `pseudo=False`:
                s(Y) = - max(w * (r - Y)) - beta * sum(w * (r - Y))

            If `invert=True` and `pseudo=True`:
                s(Y) = min(w * (Y - r)) + beta * sum(w * (Y - r))

            If `invert=True` and `pseudo=False`:
                s(Y) = max(w * (Y - r)) + beta * sum(w * (Y - r))

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

        Returns:
            A `batch_shape x num_points x num_weights x num_ref`-dim Tensor
                containing the scalarized objective vectors.
        """
        penalty = beta * LinearScalarization.evaluate(
            Y=Y, weights=weights, ref_points=ref_points, invert=invert, clip=False
        )
        objective = ChebyshevScalarization.evaluate(
            Y=Y,
            weights=weights,
            ref_points=ref_points,
            invert=invert,
            pseudo=pseudo,
            clip=False,
        )

        return objective + penalty

    def forward(self, Y: Tensor) -> Tensor:
        r"""Computes the augmented Chebyshev scalarization.

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
        """
        super().__init__()

        self.weights = weights
        self.ref_points = ref_points
        self.beta = beta
        self.invert = invert
        self.pseudo = pseudo

    @staticmethod
    def evaluate(
        Y: Tensor,
        weights: Tensor,
        ref_points: Tensor,
        beta: float = 5e-2,
        invert: bool = False,
        pseudo: bool = False,
    ) -> Tensor:
        r"""Computes the modified Chebyshev scalarization function.

            If `invert=False` and `pseudo=True`:
                s(Y) = - min(w * (r - Y) - beta * sum(w * (r - Y)))

            If `invert=False` and `pseudo=False`:
                s(Y) = - max(w * (r - Y) - beta * sum(w * (r - Y)))

            If `invert=True` and `pseudo=True`:
                s(Y) = min(w * (Y - r) + beta * sum(w * (Y - r)))

            If `invert=True` and `pseudo=False`:
                s(Y) = max(w * (Y - r) + beta * sum(w * (Y - r)))

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

        Returns:
            A `batch_shape x num_points x num_weights x num_ref`-dim Tensor
                containing the scalarized objective vectors.
        """

        # `batch_shape x num_points x num_weights x num_ref`
        penalty = LinearScalarization.evaluate(
            Y=Y, weights=weights, ref_points=ref_points, invert=invert, clip=False
        )

        sign = 1.0 if invert else -1.0
        # `num_points x batch_shape x 1 x M`
        reshaped_Y = Y.movedim(-2, 0).unsqueeze(-2)
        # `num_points x batch_shape x 1 x num_ref x M`
        diff = sign * (reshaped_Y - ref_points).unsqueeze(-3)
        # `batch_shape x num_points x num_weights x num_ref x M`
        weighted_obj = (weights.unsqueeze(-2) * diff).movedim(0, -4)
        beta_times_penalty = beta * penalty.unsqueeze(-1)

        # `batch_shape x num_points x num_weights x num_ref`
        if invert:
            if pseudo:
                return torch.min(weighted_obj + beta_times_penalty, dim=-1).values
            else:
                return torch.max(weighted_obj + beta_times_penalty, dim=-1).values
        else:
            if pseudo:
                return -torch.min(weighted_obj - beta_times_penalty, dim=-1).values
            else:
                return -torch.max(weighted_obj - beta_times_penalty, dim=-1).values

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
        )


class PBIScalarization(ScalarizationFunction):
    r"""Penalty Boundary Intersection scalarization function."""
    num_params = 2

    def __init__(
        self,
        weights: Tensor,
        ref_points: Tensor,
        beta: float = 5,
        invert: bool = False,
    ) -> None:
        r"""Penalty Boundary Intersection scalarization function.

        Args:
            weights: A `batch_shape x num_weights x M`-dim Tensor of weights.
            ref_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                reference points.
            beta: The trade-off parameter controlling the diversity penalty.
            invert: If True we compute with respect to the nadir, else we compute
                with respect to the utopia.
        """
        super().__init__()

        self.weights = weights
        self.ref_points = ref_points
        self.beta = beta
        self.invert = invert

    @staticmethod
    def evaluate(
        Y: Tensor,
        weights: Tensor,
        ref_points: Tensor,
        beta: float = 5,
        invert: bool = False,
    ) -> Tensor:
        r"""Computes the penalty boundary intersection scalarization or inverted
        penalty boundary intersection scalarization.

            If `invert=False`:
                d1(Y) = - |sum(w * (r - Y))|
                d2(Y) = sum(|Y - (r - d1(Y)w)|^2)^(1/2)
                s(Y) = - d1(Y) - beta * d2(Y)

            If `invert=True`:
                d1(Y) = - |sum(w * (Y - r))|
                d2(Y) = sum(|Y - (r + d1(Y)w)|^2)^(1/2)
                s(Y) = d1(Y) - beta * d2(Y)

        Args:
            Y: An `batch_shape x num_points x M`-dim Tensor containing the objective
                vectors.
            weights: A `batch_shape x num_weights x M`-dim Tensor of weights.
            ref_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                reference points.
            beta: The trade-off parameter controlling the diversity penalty.
            invert: If True we compute with respect to the nadir, else we compute
                with respect to the utopia.

        Returns:
            A `batch_shape x num_points x num_weights x num_ref`-dim Tensor
                containing the scalarized objectives.
        """
        sign = 1.0 if invert else -1.0
        # `num_points x batch_shape x 1 x M`
        reshaped_Y = Y.movedim(-2, 0).unsqueeze(-2)
        # `num_points x batch_shape x 1 x num_ref x M`
        diff = sign * (reshaped_Y - ref_points).unsqueeze(-3)
        # `num_points x batch_shape x num_weights x num_ref`
        d1 = torch.abs(torch.sum(weights.unsqueeze(-2) * diff, dim=-1))

        # `num_points x batch_shape x num_weights x num_ref x M`
        point = ref_points + sign * d1.unsqueeze(-1) * weights.unsqueeze(-2)

        # `num_points x batch_shape x num_weights x num_ref x M`
        diff_2 = reshaped_Y.unsqueeze(-2) - point

        # `num_points x batch_shape x num_weights x num_ref`
        d2 = torch.norm(diff_2, p=2, dim=-1)

        # `num_points x batch_shape x num_weights x num_ref`
        sY = sign * d1 - beta * d2

        # `batch_shape x num_points x num_weights x num_ref`
        return sY.movedim(0, -3)

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
        )


class KSScalarization(ScalarizationFunction):
    r"""Kalai-Smorodinsky scalarization."""
    num_params = 1

    def __init__(
        self,
        utopia_points: Tensor,
        nadir_points: Tensor,
    ) -> None:
        r"""Kalai-Smorodinsky scalarization.

        Args:
            utopia_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                utopia points.
            nadir_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                nadir points.
        """
        super().__init__()

        self.utopia_points = utopia_points
        self.nadir_points = nadir_points

    @staticmethod
    def evaluate(
        Y: Tensor,
        utopia_points: Tensor,
        nadir_points: Tensor,
    ) -> Tensor:
        r"""Computes the Kalai-Smorodinsky scalarization.

            If `invert=False`:
                s(Y) = min((y - nadir) / (utopia - nadir))

        Args:
            Y: An `batch_shape x num_points x M`-dim Tensor containing the objective
                vectors.
            utopia_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                utopia points.
            nadir_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                nadir points.

        Returns:
            A `batch_shape x num_ref`-dim Tensor containing the scalarized
                objectives.
        """

        # `num_points x batch_shape x num_ref x M`
        obj_range = utopia_points - nadir_points

        # `num_points x batch_shape x num_ref x num_ref`
        sY = ChebyshevScalarization.evaluate(
            Y=Y,
            weights=1 / obj_range,
            ref_points=nadir_points,
            invert=True,
            pseudo=True,
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
            Y=Y, utopia_points=self.utopia_points, nadir_points=self.nadir_points
        )
