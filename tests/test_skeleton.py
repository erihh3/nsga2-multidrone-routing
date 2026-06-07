"""Phase 0 smoke tests: the package imports and the toolchain is wired.

Real unit tests (instance, decode, fitness hand-calc, toy-front HV) arrive with
their modules in later phases — one test file per core module.
"""

from __future__ import annotations


def test_package_imports():
    import uav
    from uav.solution import Solution

    assert uav.__version__
    s = Solution(routes=((0, 1, 0),), makespan=1.0, energy=2.0)
    assert s.objectives == (1.0, 2.0)


def test_deap_smoke():
    """DEAP is installed and its NSGA-II building blocks are importable."""
    from deap import base, creator, tools

    assert hasattr(tools, "selNSGA2")
    assert hasattr(tools, "selTournamentDCD")
    assert hasattr(tools, "cxOrdered")
    # creator.create mutates global state; just confirm the entry points exist.
    assert callable(creator.create)
    assert base.Fitness is not None


def test_scientific_stack_smoke():
    import numpy as np
    import scipy.stats as ss

    # Mann-Whitney U is the planned cross-seed test; confirm it is available.
    assert hasattr(ss, "mannwhitneyu")
    assert np.array([1, 2, 3]).sum() == 6
