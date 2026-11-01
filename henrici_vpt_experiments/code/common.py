# -*- coding: utf-8 -*-
"""
Shared utilities for the Henrici departure vs. Valid Prediction Time (VPT)
experiments on the Lorenz-63 system.

Everything in this file is used by the individual experiment scripts:

    generate_lorenz_pool()   -> builds a pool of Lorenz-63 trajectories
    load_lorenz_pool()       -> loads the pool
    run_trial()              -> runs one reservoir-computing trial
    save_experiment_npz()    -> saves a full experiment (data + metadata) to .npz
    combine_npz_files()      -> merges per-experiment .npz files into one

Design choices
--------------
* ONE pool of Lorenz trajectories is generated once and shared by every
  experiment.  This keeps the comparison fair (same data, different
  reservoirs) and makes the runs much faster.
* For every trial we record BOTH the valid prediction time (VPT) and the
  Henrici departure of the reservoir that produced it, so that Henrici
  departure can be plotted against VPT directly.
* `henrici_norm` = Henrici departure / n is also recorded: the Frobenius
  norm of the non-normal part grows with matrix size, so the per-node
  quantity is the fair one to pool across different reservoir sizes.
* All randomness is seeded (the seed is recorded for every trial), so every
  experiment is fully reproducible.
"""
import os
import json
import datetime
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.preprocessing import StandardScaler
from reservoir_computer import reservoir_computer

# ------------------------------------------------------------------ paths --
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
RESULTS_DIR = os.path.join(BASE, "results")
PLOT_DIR = os.path.join(BASE, "plots")

# ------------------------------------------------------------- data setup --
DT = 0.01                 # time step of the Lorenz integration
BURN_IN = 5000            # transient steps discarded before recording
POOL_N_STEPS = 70000      # steps recorded per trajectory
TRAIN_LENGTH = 50000      # training steps (matches the example scripts)
PREDICTION_LENGTH = 2000  # rollout length used for the VPT
THRESHOLD_FACTOR = 0.5    # VPT threshold = 0.5 * std(true)  (class default)

LORENZ_SIGMA = 10.0
LORENZ_BETA = 8.0 / 3.0
LORENZ_RHO = 28.0


def lorenz63(t, state, sigma=LORENZ_SIGMA, beta=LORENZ_BETA, rho=LORENZ_RHO):
    x, y, z = state
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


# ---------------------------------------------------------- lorenz pool ---
def generate_lorenz_pool(n_traj=10, seed_base=1000, out_path=None, verbose=True):
    """
    Integrate n_traj independent Lorenz-63 trajectories, standardise each one
    (StandardScaler, as in the example scripts) and save them to a .npz pool.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    if out_path is None:
        out_path = os.path.join(DATA_DIR, "lorenz_pool.npz")

    pool = {}
    for i in range(n_traj):
        # Random starting point on the attractor (burn-in first)
        sol0 = solve_ivp(lorenz63, (0.0, BURN_IN * DT), [1.0, 1.0, 1.0],
                         method="RK45", rtol=1e-8, atol=1e-10)
        x0 = sol0.y[:, -1]

        t_eval = np.arange(0.0, POOL_N_STEPS * DT, DT)
        sol = solve_ivp(lorenz63, (0.0, POOL_N_STEPS * DT), x0,
                        method="RK45", rtol=1e-8, atol=1e-10, t_eval=t_eval)
        X = sol.y.T

        X = StandardScaler().fit_transform(X)
        pool["traj_%d" % i] = X
        if verbose:
            print("  lorenz pool: trajectory %d/%d done" % (i + 1, n_traj))

    np.savez_compressed(out_path, **pool)
    if verbose:
        print("  lorenz pool saved to", out_path)
    return out_path


def load_lorenz_pool(path=None):
    """Load the Lorenz pool and return a list of (T,3) standardised arrays."""
    if path is None:
        path = os.path.join(DATA_DIR, "lorenz_pool.npz")
    with np.load(path) as f:
        keys = sorted(k for k in f.files if k.startswith("traj_"))
        return [f[k] for k in keys]


# ----------------------------------------------------------- single run ----
def run_trial(X, seed, prediction_length=PREDICTION_LENGTH, **res_params):
    """
    Run one full reservoir-computing trial on trajectory X.

    * seeds all reservoir randomness with `seed`
    * trains, rolls out `prediction_length` steps, computes the VPT
      against the true continuation, and records the Henrici departure.

    Returns (vpt, henrici, henrici_norm) as floats/ints.
    """
    train_length = TRAIN_LENGTH
    np.random.seed(seed)

    data_in = X[0:train_length - 1, :]
    data_out = X[1:train_length, :]
    test_start = train_length

    res = reservoir_computer(data_in, data_out, **res_params)
    res.train_reservoir()

    pred = res.predict(X[test_start, :], prediction_length)
    vpt = res.valid_prediction_time(
        X[test_start:test_start + prediction_length, :], pred,
        threshold_factor=THRESHOLD_FACTOR)

    henrici = res.henrici_departure()
    henrici_norm = henrici / res.n
    return int(vpt), float(henrici), float(henrici_norm)


# ------------------------------------------------------------ npz saving ---
def _default_metadata(experiment_name, description, grid, n_trials,
                      data_pool=None, extra=None):
    md = {
        "experiment": experiment_name,
        "description": description,
        "date": datetime.datetime.now().isoformat(),
        "system": "Lorenz-63 (sigma=10, beta=8/3, rho=28)",
        "integration": "RK45, rtol=1e-8, atol=1e-10, dt=%g, burn-in=%d" % (DT, BURN_IN),
        "data": {
            "train_length": TRAIN_LENGTH,
            "prediction_length": PREDICTION_LENGTH,
            "vpt_threshold_factor": THRESHOLD_FACTOR,
            "scaling": "StandardScaler per trajectory",
            "pool_file": os.path.basename(data_pool) if data_pool else None,
        },
        "reservoir_defaults": {
            "reservoir_type": "erdos_renyi",
            "res_density": 0.02,
            "res_distribution": "normal",
            "res_weights": 1,
            "non_normal_type": "sparse",
            "Win_dist": "uniform",
            "gamma": 0.2749319,
            "beta": 0.0004361,
            "scale_A": True,
            "input_noise": True,
            "spectral_radius_default": 4.7846556,
            "alpha_default": 0.4054178,
            "n_default": 200,
        },
        "grid": grid,
        "n_trials_per_setting": n_trials,
        "notes": extra or "",
    }
    return md


def save_experiment_npz(name, description, arrays, grid, n_trials,
                        out_dir=None, data_pool=None, extra=None):
    """
    Save one experiment as an .npz file: every array in `arrays` is stored
    plus a `metadata` key holding all experiment details as JSON.
    Returns the path of the saved file.
    """
    if out_dir is None:
        out_dir = RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name + ".npz")

    md = _default_metadata(name, description, grid, n_trials,
                           data_pool=data_pool, extra=extra)
    payload = dict(arrays)
    payload["metadata"] = json.dumps(md, indent=2)
    np.savez_compressed(path, **payload)
    print("  saved:", path)
    return path


def combine_npz_files(paths, out_path=None):
    """Merge per-experiment npz files into one combined npz (arrays get an
    'exp<idx>_' prefix)."""
    if out_path is None:
        out_path = os.path.join(RESULTS_DIR, "all_experiments.npz")
    combined = {}
    for idx, p in enumerate(paths):
        with np.load(p) as f:
            for k in f.files:
                combined["exp%d_%s" % (idx, k)] = f[k]
    combined["combined_date"] = datetime.datetime.now().isoformat()
    combined["combined_files"] = json.dumps([os.path.basename(p) for p in paths])
    np.savez_compressed(out_path, **combined)
    print("  combined saved:", out_path)
    return out_path


# --------------------------------------------------------- experiment core --
def run_grid_experiment(name, description, settings, seeds, pool,
                        grid, extra=None, verbose=True):
    """
    Generic runner: for every setting in `settings` (a list of kwargs dicts)
    and every seed in `seeds`, run one trial and collect the results.

    `pool` is a list of trajectories; trial k of a setting uses
    pool[k % len(pool)], so different settings do not all see the same data.

    Saves the npz via save_experiment_npz and returns the result arrays dict.
    """
    n_sets = len(settings)
    n_runs = n_sets * len(seeds)
    if verbose:
        print("== %s: %d settings x %d trials = %d runs ==" %
              (name, n_sets, len(seeds), n_runs))

    vpt = np.zeros(n_runs, dtype=int)
    henrici = np.zeros(n_runs)
    henrici_norm = np.zeros(n_runs)
    seed_arr = np.zeros(n_runs, dtype=int)
    traj_arr = np.zeros(n_runs, dtype=int)
    setting_arr = np.zeros(n_runs, dtype=int)

    # echo the swept parameters per run (defaults -> the default values)
    echo_keys = ["a", "n", "spectral_radius", "alpha", "res_density"]
    echo_defaults = {"a": 0, "n": 200, "spectral_radius": 4.7846556,
                     "alpha": 0.4054178, "res_density": 0.02}
    echo = {k: np.zeros(n_runs) for k in echo_keys}

    run = 0
    t_start = datetime.datetime.now()
    for s_idx, params in enumerate(settings):
        for k, seed in enumerate(seeds):
            X = pool[k % len(pool)]
            v, hd, hdn = run_trial(X, seed, **params)
            vpt[run] = v
            henrici[run] = hd
            henrici_norm[run] = hdn
            seed_arr[run] = seed
            traj_arr[run] = k % len(pool)
            setting_arr[run] = s_idx
            for kkey in echo_keys:
                echo[kkey][run] = params.get(kkey, echo_defaults[kkey])
            run += 1
            if verbose and (run % 25 == 0 or run == n_runs):
                elapsed = (datetime.datetime.now() - t_start).total_seconds()
                print("  [%4d/%d runs]  %.1f s elapsed  (last: a=%s, VPT=%d, Henrici=%.3f)"
                      % (run, n_runs, elapsed,
                         params.get("a", 0), v, hd))

    arrays = {
        "vpt": vpt,
        "henrici": henrici,
        "henrici_norm": henrici_norm,
        "seed": seed_arr,
        "traj_index": traj_arr,
        "setting_index": setting_arr,
        "settings_json": json.dumps(settings, indent=2),
        "a": echo["a"],
        "n": echo["n"],
        "spectral_radius": echo["spectral_radius"],
        "alpha": echo["alpha"],
        "res_density": echo["res_density"],
    }
    save_experiment_npz(name, description, arrays, grid, len(seeds),
                        data_pool=os.path.join(DATA_DIR, "lorenz_pool.npz"),
                        extra=extra)
    return arrays
