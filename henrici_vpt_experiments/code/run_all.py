# -*- coding: utf-8 -*-
"""
Run ALL experiments in order and combine their .npz outputs into one file.

Usage:
    python run_all.py            # run exp1..exp5 then combine
    python run_all.py 1 3        # run only experiments 1 and 3
"""
import os
import sys
import subprocess

BASE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = ["exp1_a_sweep.py", "exp2_n_vs_a.py", "exp3_sr_vs_a.py",
           "exp4_alpha_vs_a.py", "exp5_n_sr_pair.py"]

which = [int(s) for s in sys.argv[1:]] or list(range(1, 6))
for i in which:
    if i < 1 or i > len(SCRIPTS):
        sys.exit("experiments must be in 1..%d" % len(SCRIPTS))

for i in which:
    script = SCRIPTS[i - 1]
    print("\n" + "=" * 70)
    print("RUNNING EXPERIMENT %d: %s" % (i, script))
    print("=" * 70)
    code = subprocess.call([sys.executable, os.path.join(BASE, script)],
                           cwd=BASE)
    if code != 0:
        sys.exit("experiment %d failed with code %d" % (i, code))

# combine everything
import common
paths = [os.path.join(BASE, "results", s[:-3] + ".npz") for s in SCRIPTS]
paths = [p for p in paths if os.path.exists(p)]
common.combine_npz_files(paths)
print("\nAll done.")
