# -*- coding: utf-8 -*-
"""
EXPERIMENT 4 -- learning rate alpha vs Henrici departure
=========================================================
Two-parameter sweep: alpha in {0.01, 0.1, 0.4054178, 0.8, 1.0} crossed with
Henrici departure a in {0, 1, 2, 4, 6}.  n=200 and spectral radius stay at
their class defaults.

alpha does not change the Henrici departure (which depends only on A), so
this experiment shows how VPT responds to alpha for each Henrici level.

Output: results/exp4_alpha_vs_a.npz
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import load_lorenz_pool, run_grid_experiment

alpha_vals = [0.01, 0.1, 0.4054178, 0.8, 1.0]
a_vals = [0.0, 1.0, 2.0, 4.0, 6.0]
seeds = [13001, 13011, 13021, 13031, 13041, 13051]  # 6 trials

settings = [{"a": a, "alpha": alpha,
             "reservoir_type": "erdos_renyi",
             "non_normal_type": "sparse"}
            for alpha in alpha_vals for a in a_vals]

grid = {"alpha": alpha_vals, "a": a_vals,
        "n": 200, "spectral_radius": 4.7846556,
        "res_density": 0.02, "reservoir_type": "erdos_renyi"}

if __name__ == "__main__":
    pool = load_lorenz_pool()
    run_grid_experiment(
        "exp4_alpha_vs_a",
        ("alpha vs Henrici departure on Lorenz-63: "
         "alpha in %s crossed with a in %s, 6 trials per combination. "
         "n=200, spectral radius at default." % (alpha_vals, a_vals)),
        settings, seeds, pool, grid,
        extra="alpha does not affect the Henrici departure (it depends only on A).")
