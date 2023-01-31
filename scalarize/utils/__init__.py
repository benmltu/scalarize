#!/usr/bin/env python3

from scalarize.utils.sampling import (
    sample_ordered_simplex,
    sample_ordered_uniform,
    sample_ordered_unit_vector,
    sample_permutations,
    sample_unit_vector,
)

from scalarize.utils.scalarization_functions import (
    AugmentedChebyshevScalarization,
    ChebyshevScalarization,
    HypervolumeScalarization,
    KSScalarization,
    LengthScalarization,
    LinearScalarization,
    LpScalarization,
    ModifiedChebyshevScalarization,
    PBIScalarization,
    ScalarizationFunction,
)

from scalarize.utils.scalarization_objectives import (
    compute_scalarized_objective,
    flatten_scalarized_objective,
    get_utility_mcobjective,
)

from scalarize.utils.scalarization_parameters import (
    OrderedUniformExpSpacing,
    OrderedUniformScale,
    ScalarizationParameterTransform,
    SimplexWeightExpNormalize,
    SimplexWeightNormalize,
    SimplexWeightScale,
    UnitVectorErfNormalize,
    UnitVectorNormalize,
    UnitVectorPolar,
    UnitVectorScale,
)

from scalarize.utils.transformations import (
    estimate_bounds,
    get_baseline_candidates,
    get_normalize,
    get_triangle_candidates,
)

from scalarize.utils.triangle_candidates import (
    triangle_candidates,
    triangle_candidates_fringe,
    triangle_candidates_interior,
)


__all__ = [
    "sample_unit_vector",
    "sample_ordered_uniform",
    "sample_ordered_simplex",
    "sample_ordered_unit_vector",
    "sample_permutations",
    "ScalarizationFunction",
    "LinearScalarization",
    "LpScalarization",
    "ChebyshevScalarization",
    "LengthScalarization",
    "HypervolumeScalarization",
    "AugmentedChebyshevScalarization",
    "ModifiedChebyshevScalarization",
    "PBIScalarization",
    "KSScalarization",
    "flatten_scalarized_objective",
    "compute_scalarized_objective",
    "get_utility_mcobjective",
    "ScalarizationParameterTransform",
    "SimplexWeightNormalize",
    "SimplexWeightScale",
    "UnitVectorNormalize",
    "UnitVectorPolar",
    "UnitVectorScale",
    "SimplexWeightExpNormalize",
    "UnitVectorErfNormalize",
    "OrderedUniformExpSpacing",
    "OrderedUniformScale",
    "estimate_bounds",
    "get_normalize",
    "get_triangle_candidates",
    "get_baseline_candidates",
    "triangle_candidates_interior",
    "triangle_candidates_fringe",
    "triangle_candidates",
]
