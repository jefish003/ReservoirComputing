# -*- coding: utf-8 -*-
"""
EXPERIMENT 2 -- reservoir size n vs Henrici departure a
========================================================
Two-parameter sweep: reservoir size n in {100, 200, 400, 800} crossed with
Henrici departure a in {0, 1, 2, 4, 6}.  Spectral radius and alpha stay at
their class defaults.

Note: the Frobenius norm of the non-normal part grows with n, so we also
record henrici/n (per-node Henrici departure) to make the n levels
comparable on the pooled Henrici-vs-VPT plots.

Output: results/exp2_n_vs_a.npz
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import load_lorenz_pool, run_grid_experiment

n_vals = [100, 200, 400, 800]
a_vals = [0.0, 1.0, 2.0, 4.0, 6.0]
seeds = [5001, 5011, 5021, 5031, 5041, 5051]  # 6 trials

settings = [{"a": a, "n": n,
             "reservoir_type": "erdos_renyi",
             "non_normal_type": "sparse"}
            for n in n_vals for a in a_vals]

grid = {"n": n_vals, "a": a_vals,
        "spectral_radius": 4.7846556, "alpha": 0.4054178,
        "res_density": 0.02, "reservoir_type": "erdos_renyi"}

if __name__ == "__main__":
    pool = load_lorenz_pool()
    run_grid_experiment(
        "exp2_n_vs_a",
        ("Reservoir size vs Henrici departure on Lorenz-63: "
         "n in %s crossed with a in %s, 6 trials per combination. "
         "Spectral radius and alpha at defaults." % (n_vals, a_vals)),
        settings, seeds, pool, grid,
        extra="henrici_norm = henrici/n for size-independent comparisons.")
