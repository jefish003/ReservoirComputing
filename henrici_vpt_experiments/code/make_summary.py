# -*- coding: utf-8 -*-
"""
Print (and save to results/summary.txt) a compact summary of every
experiment: Spearman correlations and median VPT / Henrici per level of the
swept parameter.
"""
import os
import numpy as np
from scipy.stats import spearmanr

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")
names = ["exp1_a_sweep", "exp2_n_vs_a", "exp3_sr_vs_a",
         "exp4_alpha_vs_a", "exp5_n_sr_pair"]

lines = []
def out(s=""):
    print(s)
    lines.append(s)

for n in names:
    with np.load(os.path.join(RESULTS, n + ".npz")) as f:
        d = {k: f[k] for k in f.files}
    out()
    out("== %s ===========  N trials: %d" % (n, len(d["vpt"])))
    out("  Spearman rho(henrici, vpt)    = %.3f"
        % spearmanr(d["henrici"], d["vpt"]).correlation)
    out("  Spearman rho(henrici/n, vpt)  = %.3f"
        % spearmanr(d["henrici_norm"], d["vpt"]).correlation)
    if np.ptp(d["a"]) > 0:
        out("  median VPT / Henrici per a:")
        for av in np.unique(d["a"]):
            m = np.isclose(d["a"], av)
            out("    a=%5.1f  median VPT=%6.1f  median Henrici=%8.3f"
                % (av, np.median(d["vpt"][m]), np.median(d["henrici"][m])))
    if np.ptp(d["n"]) > 0:
        out("  median VPT at a=0 per n:")
        for nv in np.unique(d["n"]):
            m = np.isclose(d["n"], nv)
            out("    n=%5d  median VPT=%6.1f"
                % (nv, np.median(d["vpt"][m & (d["a"] == 0)])))
    if np.ptp(d["spectral_radius"]) > 0 and np.ptp(d["a"]) == 0:
        out("  (a fixed at %.1f; VPT per n, sr)" % d["a"][0])

with open(os.path.join(RESULTS, "summary.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("\nsummary saved to", os.path.join(RESULTS, "summary.txt"))
