# -*- coding: utf-8 -*-
"""
EXPERIMENT 3 -- spectral radius vs Henrici departure
=====================================================
Two-parameter sweep: spectral radius in {1.5, 3.0, 4.7846556, 7.0} crossed
with Henrici departure a in {0, 1, 2, 4, 6}.  n=200 and alpha stay at their
class defaults.

Because the final reservoir is rescaled to the requested spectral radius,
the Henrici departure scales roughly linearly with the spectral radius for
a > 0, so this sweep also varies the Henrici departure itself.

Output: results/exp3_sr_vs_a.npz
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import load_lorenz_pool, run_grid_experiment

sr_vals = [1.5, 3.0, 4.7846556, 7.0]
a_vals = [0.0, 1.0, 2.0, 4.0, 6.0]
seeds = [9001, 9011, 9021, 9031, 9041, 9051]  # 6 trials

settings = [{"a": a, "spectral_radius": sr,
             "reservoir_type": "erdos_renyi",
             "non_normal_type": "sparse"}
            for sr in sr_vals for a in a_vals]

grid = {"spectral_radius": sr_vals, "a": a_vals,
        "n": 200, "alpha": 0.4054178,
        "res_density": 0.02, "reservoir_type": "erdos_renyi"}

if __name__ == "__main__":
    pool = load_lorenz_pool()
    run_grid_experiment(
        "exp3_sr_vs_a",
        ("Spectral radius vs Henrici departure on Lorenz-63: "
         "sr in %s crossed with a in %s, 6 trials per combination. "
         "n=200, alpha at default." % (sr_vals, a_vals)),
        settings, seeds, pool, grid,
        extra="Henrici departure of the final (rescaled) matrix is recorded per trial.")
