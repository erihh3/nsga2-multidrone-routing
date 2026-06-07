"""uav-routing: co-equal NSGA-II vs MOPSO on bi-objective multi-UAV mTSP.

The package is deliberately split so that everything downstream of a candidate
solution (decode -> fitness -> metrics/stats/plots) is shared code, and only the
optimizer loop + its genotype differ between the two algorithms.
"""

__version__ = "0.1.0"
