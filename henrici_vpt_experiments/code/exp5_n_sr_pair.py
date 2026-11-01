# -*- coding: utf-8 -*-
"""
EXPERIMENT 5 -- joint n x spectral-radius sweep at several Henrici levels
==========================================================================
Two-parameter sweep of reservoir size n in {100, 400} and spectral radius in
{3.0, 6.0}, repeated at three Henrici departure levels a in {0, 2, 6}.
alpha stays at its class default.

This shows how the n-sr trade-off interacts with the non-normality: for each
(a, n, sr) cell we record the VPT and the Henrici departure actually
achieved by that reservoir.

Output: results/exp5_n_sr_pair.npz
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import load_lorenz_pool, run_grid_experiment

a_vals = [0.0, 2.0, 6.0]
n_vals = [100, 400]
sr_vals = [3.0, 6.0]
seeds = [17001, 17011, 17021, 17031, 17041, 17051]  # 6 trials

settings = [{"a": a, "n": n, "spectral_radius": sr,
             "reservoir_type": "erdos_renyi",
             "non_normal_type": "sparse"}
            for a in a_vals for n in n_vals for sr in sr_vals]

grid = {"a": a_vals, "n": n_vals, "spectral_radius": sr_vals,
        "alpha": 0.4054178, "res_density": 0.02,
        "reservoir_type": "erdos_renyi"}

if __name__ == "__main__":
    pool = load_lorenz_pool()
    run_grid_experiment(
        "exp5_n_sr_pair",
        ("Joint n x spectral-radius sweep at a in %s on Lorenz-63: "
         "n in %s, sr in %s, 6 trials per cell. alpha at default."
         % (a_vals, n_vals, sr_vals)),
        settings, seeds, pool, grid,
        extra="henrici_norm = henrici/n recorded per trial.")
