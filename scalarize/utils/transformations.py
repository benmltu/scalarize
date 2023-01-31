#!/usr/bin/env python3

r"""
Helper utilities for constructing the transformations.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

import torch
from botorch.models.model import Model
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize
from torch import Tensor

from scalarize.models.transforms.outcome import GaussianQuantile, Normalize
from scalarize.utils.triangle_candidates import triangle_candidates


def estimate_bounds(
    Y_baseline: Optional[Tensor] = None,
    model: Optional[Model] = None,
    X_baseline: Optional[Tensor] = None,
    eta: Optional[float] = 0.5,
    kappa: Optional[float] = None,
    sampler: Optional[MCSampler] = None,
    num_samples: Optional[int] = None,
) -> Tensor:
    r"""Estimate the bounds.

    Model estimate:
    `upper_bound = max(mean(Y_i) + kappa * std(Y_i))`
    `lower_bound = min(mean(Y_i) - kappa * std(Y_i))`,
    where the mean and std are computed over samples `Y_i ~ f(X_baseline)` for
    `i=1, ..., num_samples`. For Gaussian processes, the mean and standard deviation
    can be computed analytically.

    Data estimate:
    `upper_bound = quantile(Y_baseline, 0.5+eta) + kappa * mad(Y_baseline)`
    `lower_bound = quantile(Y_baseline, 0.5-eta) + kappa * mad(Y_baseline)`,
    where `mad` is the median absolute deviation:
    `mad(Y_baseline) = median(abs(Y_baseline - median(Y_baseline))`

    Args:
        Y_baseline: A `num_points x M`-dim tensor of objective vectors.
        model: A fitted model.
        X_baseline: A `num_points x d`-dim tensor of input vectors.
        eta: A parameter that is used to control the quantile.
        kappa: A parameter that is used to control the amount of overestimation.
        sampler: If this is given along with the `model` and `X_baseline` then we
            compute a Monte Carlo estimate of the mean and standard deviation.
        num_samples: If this is given along with the `model` and `X_baseline` and
            `sampler` is None, then we compute a Monte Carlo estimate of the mean
            and standard deviation using SobolQMCNormalSampler with `num_samples` of
            Monte Carlo samples.

    Returns:
        A `2 x m`-dim Tensor containing the bounds.
    """

    if model is not None and X_baseline is not None:
        # compute model estimate

        if sampler is not None:
            sample_from_model = True
        elif num_samples is not None:
            sampler = SobolQMCNormalSampler(
                num_samples=num_samples, collapse_batch_dims=True
            )
            sample_from_model = True
        else:
            sample_from_model = False

        with torch.no_grad():
            posterior = model.posterior(X_baseline)
            if sample_from_model:
                samples = sampler(posterior)
                mean = samples.mean(dim=-3)
                std = torch.sqrt(samples.var(dim=-3))
            else:
                mean = posterior.mean
                std = torch.sqrt(posterior.variance)

        if kappa is None:
            kappa = 0.1

        upper = (mean + kappa * std).max(dim=-2).values
        lower = (mean - kappa * std).min(dim=-2).values

        bounds = torch.row_stack([lower, upper])
        return bounds

    elif Y_baseline is not None:
        # compute data estimate
        upper_quantile = Y_baseline.quantile(q=0.5 + eta, dim=-2)
        lower_quantile = Y_baseline.quantile(q=0.5 - eta, dim=-2)
        median = Y_baseline.quantile(q=0.5, dim=-2)
        mad = abs(Y_baseline - median).quantile(q=0.5, dim=-2)

        if kappa is None:
            kappa = 2 - 4 * eta

        upper = upper_quantile + kappa * mad
        lower = lower_quantile - kappa * mad

        bounds = torch.row_stack([lower, upper])
        return bounds

    else:
        raise ValueError(
            "Need to provide either `Y_baseline` or the `model` and `X_baseline` in "
            "order to estimate the bounds!"
        )


def get_normalize(
    model: Optional[Model] = None,
    X_baseline: Optional[Tensor] = None,
    Y_baseline: Optional[Tensor] = None,
    eta: Optional[float] = 0.5,
    kappa: Optional[float] = None,
    sampler: Optional[MCSampler] = None,
    num_samples: Optional[int] = None,
    bounds: Optional[Tensor] = None,
) -> OutcomeTransform:
    r"""Estimate the normalize outcome transform.

    Args:
        model: A fitted model.
        X_baseline: A `num_points x d`-dim tensor of input vectors.
        Y_baseline: A `num_points x M`-dim tensor of objective vectors.
        eta: A parameter that is used to control the quantile.
        kappa: A parameter that is used to control the amount of overestimation.
        sampler: If this is given along with the `model` and `X_baseline` then we
            compute a Monte Carlo estimate of the mean and standard deviation.
        num_samples: If this is given along with the `model` and `X_baseline` and
            `sampler` is None, then we compute a Monte Carlo estimate of the mean
            and standard deviation using SobolQMCNormalSampler with `num_samples` of
            Monte Carlo samples.
        bounds: A `2 x M`-dim Tensor of bounds for the objective.

    Returns:
        The normalize outcome transform.
    """
    if bounds is None:
        try:
            bounds = estimate_bounds(
                model=model,
                X_baseline=X_baseline,
                Y_baseline=Y_baseline,
                eta=eta,
                kappa=kappa,
                sampler=sampler,
                num_samples=num_samples,
            )
        except ValueError:
            bounds = None

    return Normalize(bounds)


def get_gaussian_quantile(
    model: Model,
    bounds: Tensor,
    num_samples: Optional[int] = 128,
    max_num_tricands: Optional[int] = 0,
    X: Optional[Tensor] = None,
) -> OutcomeTransform:
    r"""Estimate the Gaussian quantile outcome transform.

    Estimate of the central moments:
    `means = model.mean(X_baseline \cup X_tricands \cup X_uniform)`,
    `variances = model.variance(X_baseline \cup X_tricands \cup X_uniform)`,
    where `X_triangle` are the interior candidates obtained using triangulation and
    `X_uniform ~ Uniform(bounds)` are the uniformly sampled points.

    Args:
        model: A fitted model.
        bounds: A `2 x d`-dim Tensor containing the bounds of the inputs.
        num_samples: The number of uniform samples.
        max_num_tricands: The maximum number of candidates obtained by triangulation.
        X: A `num_points x d`-dim tensor of input vectors to append to the
            triangulation candidates and random samples.

    Returns:
        The Gaussian quantile outcome transform.
    """
    candidates = get_baseline_candidates(
        bounds=bounds, X=X, num_samples=num_samples, max_num_tricands=max_num_tricands
    )

    posterior = model.posterior(candidates)
    means = posterior.mean
    variances = posterior.variance

    return GaussianQuantile(means=means, variances=variances)


def get_triangle_candidates(
    X: Tensor,
    bounds: Tensor,
    fringe: Optional[bool] = True,
    max_num_candidates: Optional[int] = None,
    best_indices: Optional[Tensor] = None,
) -> Tensor:
    r"""Compute the triangle candidates.

    Args:
        X: A `N x d`-dim Tensor containing the baseline inputs. For triangulation
            when `d > 1`, we must have `N >= d+1`.
        bounds: A `2 x d`-dim Tensor containing the bounds of `X` for normalization.
        fringe: If true we compute the fringe candidates.
        max_num_candidates: The maximum number of candidates, defaults to `100 * N`.
        best_indices: A list containing the indices of the points in `X` that are
            given priority when the number of generated candidates surpasses
            `max_num_candidates`.

    Returns
        candidates: A `num_candidates x d`-dim Tensor containing the candidates.
    """
    d = X.shape[-1]
    N = X.shape[0]
    if max_num_candidates is None:
        max_num_candidates = 100 * N

    if d > 1:
        normalized_X = normalize(X=X, bounds=bounds).detach().numpy()
        tri_candidates = triangle_candidates(
            X=normalized_X,
            fringe=fringe,
            max_num_candidates=max_num_candidates,
            best_indices=best_indices,
        )

        candidates = unnormalize(X=torch.tensor(tri_candidates).to(X), bounds=bounds)

    else:
        # Handle the one-dimensional case separately.
        # The fringe candidates will essentially be the points half-way from the
        # boundary to the closest point.
        if fringe:
            aug_X = torch.cat([X, bounds])
        else:
            aug_X = X

        sorted_X, indices = torch.sort(aug_X, dim=-2)
        all_candidates = (sorted_X[1:] + sorted_X[:-1]) / 2
        num_generated_candidates = len(all_candidates)
        all_indices = torch.tensor(list(range(num_generated_candidates)))

        # Truncate the set of `all_indices` if necessary.
        adjacent_indices = torch.tensor([], dtype=int)
        if best_indices is not None:
            # First populate the `adjacent_indices` with the indices of the points
            # closest to the points highlighted by `best_indices`.
            for i in range(len(best_indices)):
                loc = torch.where(indices == best_indices[i])[0]

                if loc > 0:
                    adjacent_indices = torch.cat([adjacent_indices, loc - 1])

                if loc < num_generated_candidates:
                    adjacent_indices = torch.cat([adjacent_indices, loc])

                adjacent_indices = torch.unique(adjacent_indices)

        # If the set of `adjacent_indices` is still too large, truncate it randomly.
        # If the set of `adjacent_indices` is still too small, add points randomly.
        if len(adjacent_indices) >= max_num_candidates:
            selected_indices = adjacent_indices[
                torch.randperm(len(adjacent_indices))[:max_num_candidates]
            ]
        else:
            if len(adjacent_indices) > 0:
                # TODO: there must be a better way to get the np.delete
                #  functionality in torch.
                remaining_indices = np.delete(
                    all_indices.detach().numpy(), adjacent_indices.detach().numpy(), 0
                )
                remaining_indices = torch.tensor(remaining_indices, dtype=int)
            else:
                remaining_indices = all_indices

            other_indices = remaining_indices[
                torch.randperm(len(remaining_indices))[
                    : max_num_candidates - len(adjacent_indices)
                ]
            ]

            selected_indices = torch.cat([adjacent_indices, other_indices])

        candidates = all_candidates[selected_indices]

    return candidates


def get_baseline_candidates(
    bounds: Tensor,
    X: Optional[Tensor] = None,
    num_samples: Optional[int] = 0,
    max_num_tricands: Optional[int] = 0,
) -> Tensor:
    r"""Compute the baseline points using triangulation and sampling.

    This function returns:
    `candidates = X \cup X_triangle \cup X_uniform`,
    where `X` are the optional training inputs, `X_triangle` are the candidates
    obtained using triangulation, whilst `X_uniform` are the samples obtained from
    uniform sampling.

    Args:
        bounds: A `2 x d`-dim Tensor containing the bounds of the inputs.
        X: A `N x d`-dim Tensor containing the training inputs.
        num_samples: The number of candidates generated using uniform sampling.
        max_num_tricands: The maximum number of candidates obtained by triangulation,
            including the fringe points.

    Returns
        candidates: A `num_candidates x d`-dim Tensor containing the augmented
            baseline candidates.
    """
    candidates = torch.tensor([])
    if X is not None:
        candidates = torch.cat([candidates, X])
        if max_num_tricands > 0:
            triangle_X = get_triangle_candidates(
                X=X, bounds=bounds, fringe=True, max_num_candidates=max_num_tricands
            )
            candidates = torch.cat([candidates, triangle_X])

    if num_samples > 0:
        uniform_X = draw_sobol_samples(bounds=bounds, n=num_samples, q=1).squeeze(-2)

        if candidates is None:
            candidates = uniform_X
        else:
            candidates = torch.cat([candidates, uniform_X])

    return candidates
