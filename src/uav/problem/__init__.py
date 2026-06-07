"""Shared problem core: instance loading, genotype decoding, fitness.

Build order is gated here: instance -> decode -> fitness. No optimizer may be
written before fitness is proven correct against a hand calculation.
"""
