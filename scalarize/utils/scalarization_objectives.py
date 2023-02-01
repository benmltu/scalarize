#!/usr/bin/env python3

r"""
Helper utilities for constructing the scalarization-based objectives.
"""

from __future__ import annotations

from typing import Optional

import torch
from botorch.acquisition.objective import GenericMCObjective
from botorch.models.transforms.outcome import OutcomeTransform
from torch import Tensor

from scalarize.models.transforms.outcome import Normalize
from scalarize.utils.scalarization_functions import ScalarizationFunction


def flatten_scalarized_objective(
    Y: Tensor,
    scalarization_fn: ScalarizationFunction,
) -> Tensor:
    r"""Compute the flattened scalarized objective.

    The flatten scalarized objective would take the form:
    `(s(Y_1, Theta), s(Y_2, Theta), ..., s(Y_{num_points}, Theta))`, where `Y_n` are
    the batches, whilst `Theta = (theta_1, ..., theta_{num_scalar})` is the
    collection of scalarization parameter configurations.

    Args:
        Y: A `batch_shape x num_points x m`-dim tensor of objective vectors.
        scalarization_fn: The scalarization function used to compute the Monte Carlo
            estimate of the utility.

    Returns:
        A `batch_shape x (num_points x num_scalar)`-dim tensor of scalarized
            objectives, where `num_scalar` is the number of scalarization parameter
            configuration.
    """
    # `batch_shape x (num_points x num_scalar)`
    return torch.flatten(
        scalarization_fn(Y), start_dim=-scalarization_fn.num_params - 1, end_dim=-1
    )


def compute_scalarized_objective(
    Y: Tensor,
    scalarization_fn: ScalarizationFunction,
    outcome_transform: OutcomeTransform = None,
    flatten: bool = False,
) -> Tensor:
    r"""Compute the scalarized objective.

    Args:
        Y: A `sample_shape x num_points x M`-dim Tensor of objective values. Only
            supports `sample_shape = (num_samples,)` or `sample_shape = ()`.
        scalarization_fn: A scalarization function defined for `M` dimensional
            outputs. The total number of scalarization parameter configurations is
            denoted by `num_scalar`.
        outcome_transform: An outcome transform, defaults to the identity.
        flatten: If True, we flatten the Monte Carlo dimension into the scalarization
            parameter dimension, else we leave the Monte Carlo dimension in the
            first dimension.

    Returns:
        If flatten is True, we return a `num_points x (num_scalar x num_samples)`-dim
        Tensor, else we return a `sample_shape x num_points x num_scalar`-dim Tensor.
    """
    # TODO: maybe we need to check the sample shape.

    if outcome_transform is None:
        # this is equivalent to the identity
        outcome_transform = Normalize()

    # `num_samples x num_points x M`
    transformed_samples, _ = outcome_transform(Y)

    if flatten:
        # `num_points x (num_samples x num_scalar)`
        scalarized_samples = flatten_scalarized_objective(
            Y=transformed_samples.movedim(0, -2), scalarization_fn=scalarization_fn
        )
    else:
        # `num_samples x num_points x num_scalar`
        scalarized_samples = flatten_scalarized_objective(
            Y=transformed_samples.unsqueeze(-2), scalarization_fn=scalarization_fn
        )

    return scalarized_samples


def get_utility_mcobjective(
    scalarization_fn: ScalarizationFunction,
    outcome_transform: OutcomeTransform = None,
) -> GenericMCObjective:
    r"""Computes the Monte Carlo set utility objective.

    Args:
        scalarization_fn: An initialized scalarization function containing the
            scalarization parameters.
        outcome_transform: An initialized transformation which will be applied to the
            objective before the scalarization function.

    Returns:
        The Monte Carlo objective.
    """

    def scalarized_objectives(Y: Tensor, X: Optional[Tensor] = None) -> Tensor:
        # `Y` has shape `num_mc_samples x batch_shape x m`
        # `s_obj` has shape `batch_shape x (num_mc_samples x num_scalar)`
        s_obj = compute_scalarized_objective(
            Y=Y,
            scalarization_fn=scalarization_fn,
            outcome_transform=outcome_transform,
            flatten=True,
        )
        # `s_obj` has shape `(num_mc_samples x num_scalar) x batch_shape`
        return s_obj.movedim(-1, 0)

    return GenericMCObjective(scalarized_objectives)
