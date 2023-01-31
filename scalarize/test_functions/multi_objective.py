#! /usr/bin/env python3

r"""
Multi-objective optimization benchmark problems.

Adapted some problems from https://github.com/ryojitanabe/reproblems
"""

from __future__ import annotations

import torch
from botorch.test_functions.base import MultiObjectiveTestProblem
from torch import Tensor


class MarineDesign(MultiObjectiveTestProblem):
    r"""Conceptual marine design.

    Adapted problem `RE32` in https://github.com/ryojitanabe/reproblems
    """
    dim = 6
    num_objectives = 4
    _bounds = [
        (150.0, 274.32),
        (20.0, 32.31),
        (13.0, 25.0),
        (10.0, 11.71),
        (14.0, 18.0),
        (0.63, 0.75),
    ]
    _ref_point = [-250.0, 20000.0, 25000.0, 15.0]
    _num_original_constraints = 9

    def evaluate_true(self, X: Tensor) -> Tensor:
        """
        Args:
            X: A `batch_shape x d`-dim Tensor containing the designs.

        Returns:
            A `batch_shape x M`-dim Tensor containing the objectives.

        """
        batch_size = X[..., 0].size()
        constraintFuncs = torch.zeros(
            batch_size + torch.Size([self._num_original_constraints])
        )
        x_L = X[:, 0]
        x_B = X[:, 1]
        x_D = X[:, 2]
        x_T = X[:, 3]
        x_Vk = X[:, 4]
        x_CB = X[:, 5]

        displacement = 1.025 * x_L * x_B * x_T * x_CB
        V = 0.5144 * x_Vk
        g = 9.8065
        Fn = V / torch.pow(g * x_L, 0.5)
        a = (4977.06 * x_CB * x_CB) - (8105.61 * x_CB) + 4456.51
        b = (-10847.2 * x_CB * x_CB) + (12817.0 * x_CB) - 6960.32

        power = (torch.pow(displacement, 2.0 / 3.0) * torch.pow(x_Vk, 3.0)) / (
            a + (b * Fn)
        )
        outfit_weight = (
            1.0
            * torch.pow(x_L, 0.8)
            * torch.pow(x_B, 0.6)
            * torch.pow(x_D, 0.3)
            * torch.pow(x_CB, 0.1)
        )
        steel_weight = (
            0.034
            * torch.pow(x_L, 1.7)
            * torch.pow(x_B, 0.7)
            * torch.pow(x_D, 0.4)
            * torch.pow(x_CB, 0.5)
        )
        machinery_weight = 0.17 * torch.pow(power, 0.9)
        light_ship_weight = steel_weight + outfit_weight + machinery_weight

        ship_cost = 1.3 * (
            (2000.0 * torch.pow(steel_weight, 0.85))
            + (3500.0 * outfit_weight)
            + (2400.0 * torch.pow(power, 0.8))
        )
        capital_costs = 0.2 * ship_cost

        DWT = displacement - light_ship_weight

        running_costs = 40000.0 * torch.pow(DWT, 0.3)

        round_trip_miles = 5000.0
        sea_days = (round_trip_miles / 24.0) * x_Vk
        handling_rate = 8000.0

        daily_consumption = ((0.19 * power * 24.0) / 1000.0) + 0.2
        fuel_price = 100.0
        fuel_cost = 1.05 * daily_consumption * sea_days * fuel_price
        port_cost = 6.3 * torch.pow(DWT, 0.8)

        fuel_carried = daily_consumption * (sea_days + 5.0)
        miscellaneous_DWT = 2.0 * torch.pow(DWT, 0.5)

        cargo_DWT = DWT - fuel_carried - miscellaneous_DWT
        port_days = 2.0 * ((cargo_DWT / handling_rate) + 0.5)
        RTPA = 350.0 / (sea_days + port_days)

        voyage_costs = (fuel_cost + port_cost) * RTPA
        annual_costs = capital_costs + running_costs + voyage_costs
        annual_cargo = cargo_DWT * RTPA

        f1 = annual_costs / annual_cargo
        f2 = light_ship_weight
        # f_2 is dealt as a minimization problem
        f3 = -annual_cargo

        # Reformulated objective functions
        constraintFuncs[..., 0] = (x_L / x_B) - 6.0
        constraintFuncs[..., 1] = -(x_L / x_D) + 15.0
        constraintFuncs[..., 2] = -(x_L / x_T) + 19.0
        constraintFuncs[..., 3] = 0.45 * torch.pow(DWT, 0.31) - x_T
        constraintFuncs[..., 4] = 0.7 * x_D + 0.7 - x_T
        constraintFuncs[..., 5] = 500000.0 - DWT
        constraintFuncs[..., 6] = DWT - 3000.0
        constraintFuncs[..., 7] = 0.32 - Fn

        constraintFuncs = torch.where(
            constraintFuncs < 0, -constraintFuncs, torch.zeros(constraintFuncs.size())
        )
        f4 = torch.sum(constraintFuncs, dim=-1)

        return torch.stack([f1, f2, f3, f4], dim=-1)


class RocketInjector(MultiObjectiveTestProblem):
    r"""Rocket injector design.

    Adapted problem `RE37` in https://github.com/ryojitanabe/reproblems
    """
    dim = 4
    num_objectives = 3
    _bounds = [
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
    ]
    _ref_point = [1.05, 1.3, 1.2]
    _num_original_constraints = 0

    def evaluate_true(self, X: Tensor) -> Tensor:
        """
        Args:
            X: A `batch_shape x d`-dim Tensor containing the designs.

        Returns:
            A `batch_shape x M`-dim Tensor containing the objectives.

        """
        xAlpha = X[..., 0]
        xHA = X[..., 1]
        xOA = X[..., 2]
        xOPTT = X[..., 3]

        # f1 (TF_max)
        f1 = (
            0.692
            + (0.477 * xAlpha)
            - (0.687 * xHA)
            - (0.080 * xOA)
            - (0.0650 * xOPTT)
            - (0.167 * xAlpha * xAlpha)
            - (0.0129 * xHA * xAlpha)
            + (0.0796 * xHA * xHA)
            - (0.0634 * xOA * xAlpha)
            - (0.0257 * xOA * xHA)
            + (0.0877 * xOA * xOA)
            - (0.0521 * xOPTT * xAlpha)
            + (0.00156 * xOPTT * xHA)
            + (0.00198 * xOPTT * xOA)
            + (0.0184 * xOPTT * xOPTT)
        )
        # f2 (X_cc)
        f2 = (
            0.153
            - (0.322 * xAlpha)
            + (0.396 * xHA)
            + (0.424 * xOA)
            + (0.0226 * xOPTT)
            + (0.175 * xAlpha * xAlpha)
            + (0.0185 * xHA * xAlpha)
            - (0.0701 * xHA * xHA)
            - (0.251 * xOA * xAlpha)
            + (0.179 * xOA * xHA)
            + (0.0150 * xOA * xOA)
            + (0.0134 * xOPTT * xAlpha)
            + (0.0296 * xOPTT * xHA)
            + (0.0752 * xOPTT * xOA)
            + (0.0192 * xOPTT * xOPTT)
        )
        # f3 (TT_max)
        f3 = (
            0.370
            - (0.205 * xAlpha)
            + (0.0307 * xHA)
            + (0.108 * xOA)
            + (1.019 * xOPTT)
            - (0.135 * xAlpha * xAlpha)
            + (0.0141 * xHA * xAlpha)
            + (0.0998 * xHA * xHA)
            + (0.208 * xOA * xAlpha)
            - (0.0301 * xOA * xHA)
            - (0.226 * xOA * xOA)
            + (0.353 * xOPTT * xAlpha)
            - (0.0497 * xOPTT * xOA)
            - (0.423 * xOPTT * xOPTT)
            + (0.202 * xHA * xAlpha * xAlpha)
            - (0.281 * xOA * xAlpha * xAlpha)
            - (0.342 * xHA * xHA * xAlpha)
            - (0.245 * xHA * xHA * xOA)
            + (0.281 * xOA * xOA * xHA)
            - (0.184 * xOPTT * xOPTT * xAlpha)
            - (0.281 * xHA * xAlpha * xOA)
        )

        return torch.stack([f1, f2, f3], dim=-1)
