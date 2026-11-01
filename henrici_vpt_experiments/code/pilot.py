# -*- coding: utf-8 -*-
"""
Quick pilot: does a larger Henrici departure (via the perturbation size `a`)
give a smaller valid prediction time (VPT)?

Fixed n=200, default spectral radius and alpha; sweeps a = 0 .. 8 with 3
trials each.  Run this before the full grids to check the trend and timing.
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from common import (generate_lorenz_pool, load_lorenz_pool, run_trial,
                    DATA_DIR, POOL_N_STEPS)

if not os.path.exists(os.path.join(DATA_DIR, "lorenz_pool.npz")):
    print("Generating Lorenz pool...")
    generate_lorenz_pool(n_traj=4)
pool = load_lorenz_pool()

a_vals = [0.0, 2.0, 4.0, 8.0]
seeds = [11, 22, 33]

t0 = time.time()
for a in a_vals:
    for k, seed in enumerate(seeds):
        X = pool[k % len(pool)]
        v, hd, hdn = run_trial(X, seed, a=a)
        print("a=%4.1f  seed=%2d  VPT=%5d  Henrici=%8.4f  (henrici/n=%.4f)"
              % (a, seed, v, hd, hdn))
print("pilot wall time: %.1f s for %d runs" % (time.time() - t0, len(a_vals)*len(seeds)))
