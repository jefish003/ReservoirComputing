# -*- coding: utf-8 -*-
"""
EXPERIMENT 1 -- Henrici departure sweep (the central demonstration)
====================================================================
Everything is held at the class defaults (n=200, spectral radius 4.7846556,
alpha 0.4054178, ER graph density 0.02) EXCEPT the non-normality parameter
`a` of the reservoir, which controls the Henrici departure:

    a = 0  ->  symmetric (normal) reservoir matrix  ->  Henrici departure 0
    a > 0  ->  sparse upper-triangular perturbation of size `a` added

For every trial we record BOTH the VPT and the Henrici departure of the
reservoir that produced it, so Henrici departure can be plotted directly
against VPT.

Output: results/exp1_a_sweep.npz
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import load_lorenz_pool, run_grid_experiment

a_vals = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]
seeds = [1001, 1011, 1021, 1031, 1041, 1051, 1061, 1071, 1081, 1091]  # 10 trials

settings = [{"a": a,
             "reservoir_type": "erdos_renyi",
             "non_normal_type": "sparse"}
            for a in a_vals]

grid = {"a": a_vals,
        "n": 200,
        "spectral_radius": 4.7846556,
        "alpha": 0.4054178,
        "res_density": 0.02,
        "reservoir_type": "erdos_renyi"}

if __name__ == "__main__":
    pool = load_lorenz_pool()
    run_grid_experiment(
        "exp1_a_sweep",
        ("Henrici departure sweep on Lorenz-63: a controls the sparse "
         "upper-triangular non-normal perturbation of the ER reservoir. "
         "All other parameters at class defaults. 10 trials per a."),
        settings, seeds, pool, grid,
        extra="Henrici departure and VPT recorded per trial; henrici_norm = henrici/n.")
