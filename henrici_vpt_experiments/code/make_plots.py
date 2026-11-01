# -*- coding: utf-8 -*-
"""
Make all the figures from the saved .npz experiment results.

Reads  : results/exp*.npz  (and results/all_experiments.npz if present)
Writes : plots/fig1_a_sweep.png
         plots/fig2_n_vs_a.png
         plots/fig3_sr_vs_a.png
         plots/fig4_alpha_vs_a.png
         plots/fig5_n_sr_pair.png
         plots/fig6_pooled_all.png
"""
import os
import json
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")
PLOTS = os.path.join(BASE, "plots")
os.makedirs(PLOTS, exist_ok=True)


# ------------------------------------------------------------------ helpers
def load(name):
    with np.load(os.path.join(RESULTS, name + ".npz")) as f:
        d = {k: f[k] for k in f.files}
    meta = d["metadata"]
    if isinstance(meta, np.ndarray):
        meta = meta.item()
    d["metadata"] = json.loads(str(meta))
    return d


def unique_levels(arr):
    return np.unique(arr)


def jitter(n, seed=0):
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.12, 0.12, size=n)


def median_iqr(y):
    return np.median(y), (np.percentile(y, 25), np.percentile(y, 75))


def rolling_median(x, y, n_bins=10):
    """Median of y in n_bins equal-width x-bins -> (xs, ys) for a trend line."""
    order = np.argsort(x)
    x, y = x[order], y[order]
    edges = np.linspace(x.min(), x.max(), n_bins + 1)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x <= hi)
        if m.sum() >= 3:
            xs.append(0.5 * (lo + hi))
            ys.append(np.median(y[m]))
    return np.array(xs), np.array(ys)


def spearman(x, y):
    return float(spearmanr(x, y).correlation)


def vpt_vs_param_ax(ax, data, param_key, color_by_key, title,
                    x_label, ylim=None):
    """Generic 'VPT vs. parameter, one line per level of another parameter'."""
    levels = unique_levels(data[param_key].astype(float))
    color_by = unique_levels(data[color_by_key].astype(float))
    cmap = plt.cm.get_cmap("tab10" if len(color_by) <= 10 else "viridis")

    for c_idx, cval in enumerate(color_by):
        cmask = np.isclose(data[color_by_key].astype(float), cval)
        for l_idx, lval in enumerate(levels):
            m = cmask & np.isclose(data[param_key].astype(float), lval)
            y = data["vpt"][m]
            xs = l_idx + jitter(m.sum())
            ax.scatter(xs, y, s=22, alpha=0.75,
                       color=cmap(c_idx % cmap.N), zorder=3, edgecolors="none")
            med, (lo, hi) = median_iqr(y)
            ax.plot([l_idx, l_idx], [lo, hi], color=cmap(c_idx % cmap.N),
                    lw=2.5, alpha=0.9, zorder=4)
        # connecting line through medians
    for c_idx, cval in enumerate(color_by):
        cmask = np.isclose(data[color_by_key].astype(float), cval)
        meds = []
        for lval in levels:
            m = cmask & np.isclose(data[param_key].astype(float), lval)
            meds.append(np.median(data["vpt"][m]))
        ax.plot(range(len(levels)), meds, "--", color=cmap(c_idx % cmap.N),
                lw=1.2, alpha=0.7, zorder=5)

    handles = [plt.Line2D([0], [0], color=cmap(i % cmap.N), lw=3,
                          label="%.3g" % v) for i, v in enumerate(color_by)]
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(["%.3g" % v for v in levels], rotation=45, ha="right")
    ax.set_xlabel(x_label)
    ax.set_ylabel("VPT (steps)")
    ax.set_title(title)
    ax.legend(handles=handles, fontsize=9, loc="best")
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(alpha=0.25)
    return ax


def henrici_vpt_ax(ax, data, x_key="henrici_norm", log_x=False,
                   title="Henrici departure  vs  VPT",
                   xlabel=None, fit=True, trend=True, ylim=None):
    x = data[x_key].astype(float)
    y = data["vpt"].astype(float)
    if log_x:
        x = x[x > 0]
        y = y[x > 0]
    ax.scatter(x, y, s=22, alpha=0.6, c="#1f77b4", edgecolors="none")
    rho = spearman(x, y)
    if fit and x.size > 2 and np.ptp(x) > 0:
        xx = np.linspace(x.min(), x.max(), 100)
        fit_y = np.clip(np.polyval(np.polyfit(x, y, 1), xx), 0, None)
        ax.plot(xx, fit_y, "r-", lw=1.5, label="linear fit")
    if trend:
        xs, ys = rolling_median(x, y, n_bins=8)
        ax.plot(xs, ys, "k-", lw=2.2, alpha=0.9, label="median trend")
    ax.set_xlabel(xlabel or ("Henrici departure / n" if x_key == "henrici_norm"
                             else "Henrici departure"))
    ax.set_ylabel("VPT (steps)")
    ax.set_title(title)
    ax.annotate("Spearman rho = %.2f" % rho, xy=(0.04, 0.94),
                xycoords="axes fraction", fontsize=10,
                bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))
    ax.legend(fontsize=9)
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(alpha=0.25)
    if log_x:
        ax.set_xscale("log")
    return ax


# ================================================================= FIG 1 ===
def fig1():
    d = load("exp1_a_sweep")
    a_levels = unique_levels(d["a"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    ax = axes[0]
    for l_idx, aval in enumerate(a_levels):
        m = np.isclose(d["a"], aval)
        y = d["vpt"][m]
        ax.scatter(l_idx + jitter(m.sum()), y, s=30, alpha=0.8,
                   c="#1f77b4" if aval == 0 else "#d62728", edgecolors="none")
        med, (lo, hi) = median_iqr(y)
        ax.plot([l_idx] * 2, [lo, hi], color="k", lw=2.5)
    meds = [np.median(d["vpt"][np.isclose(d["a"], v)]) for v in a_levels]
    ax.plot(range(len(a_levels)), meds, "--", color="k", lw=1.2)
    ax.set_xticks(range(len(a_levels)))
    ax.set_xticklabels(["%.3g" % v for v in a_levels])
    ax.set_xlabel("a  (size of non-normal perturbation)")
    ax.set_ylabel("VPT (steps)")
    ax.set_title("Exp 1: VPT vs Henrici-departure parameter a\n"
                 "(blue = normal reservoir, red = non-normal)")
    ax.grid(alpha=0.25)

    rho = spearman(d["henrici"], d["vpt"])
    ax = axes[1]
    x = d["henrici"]
    y = d["vpt"]
    ax.scatter(x, y, s=30, alpha=0.7, c="#1f77b4", edgecolors="none")
    xx = np.linspace(x.min(), x.max(), 100)
    fit_y = np.clip(np.polyval(np.polyfit(x, y, 1), xx), 0, None)
    ax.plot(xx, fit_y, "r-", lw=1.5, label="linear fit")
    xs, ys = rolling_median(x, y, n_bins=8)
    ax.plot(xs, ys, "k-", lw=2.2, label="median trend")
    ax.set_xlabel("Henrici departure of the reservoir")
    ax.set_ylabel("VPT (steps)")
    ax.set_title("Exp 1: VPT vs Henrici departure (per trial)\n"
                 "Spearman rho = %.2f" % rho)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    fig.suptitle("Lorenz-63: larger Henrici departure -> shorter valid "
                 "prediction time (n=200, sr=4.78, alpha=0.405)", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "fig1_a_sweep.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote fig1_a_sweep.png")


# ================================================================= FIG 2 ===
def fig2():
    d = load("exp2_n_vs_a")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    vpt_vs_param_ax(axes[0], d, "a", "n",
                    "Exp 2: VPT vs a for each reservoir size n",
                    "a (Henrici-departure parameter)")
    henrici_vpt_ax(axes[1], d, "henrici_norm",
                   title="Exp 2: per-node Henrici departure vs VPT (all n pooled)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "fig2_n_vs_a.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote fig2_n_vs_a.png")


# ================================================================= FIG 3 ===
def fig3():
    d = load("exp3_sr_vs_a")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    vpt_vs_param_ax(axes[0], d, "a", "spectral_radius",
                    "Exp 3: VPT vs a for each spectral radius",
                    "a (Henrici-departure parameter)")
    henrici_vpt_ax(axes[1], d, "henrici",
                   title="Exp 3: Henrici departure vs VPT (all sr pooled)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "fig3_sr_vs_a.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote fig3_sr_vs_a.png")


# ================================================================= FIG 4 ===
def fig4():
    d = load("exp4_alpha_vs_a")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    vpt_vs_param_ax(axes[0], d, "a", "alpha",
                    "Exp 4: VPT vs a for each alpha",
                    "a (Henrici-departure parameter)")
    henrici_vpt_ax(axes[1], d, "henrici",
                   title="Exp 4: Henrici departure vs VPT (all alpha pooled)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "fig4_alpha_vs_a.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote fig4_alpha_vs_a.png")


# ================================================================= FIG 5 ===
def fig5():
    d = load("exp5_n_sr_pair")
    a_levels = unique_levels(d["a"])
    n_levels = unique_levels(d["n"])
    sr_levels = unique_levels(d["spectral_radius"])
    fig, axes = plt.subplots(1, len(a_levels) + 1,
                             figsize=(3.2 * (len(a_levels) + 1), 4.2))
    for i, aval in enumerate(a_levels):
        ax = axes[i]
        M = np.full((len(n_levels), len(sr_levels)), np.nan)
        for i_n, nval in enumerate(n_levels):
            for i_s, sval in enumerate(sr_levels):
                m = (np.isclose(d["a"], aval) & np.isclose(d["n"], nval)
                     & np.isclose(d["spectral_radius"], sval))
                M[i_n, i_s] = np.median(d["vpt"][m])
        im = ax.imshow(M, origin="lower", cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(sr_levels)))
        ax.set_xticklabels(["%.3g" % v for v in sr_levels])
        ax.set_yticks(range(len(n_levels)))
        ax.set_yticklabels(["%.3g" % v for v in n_levels])
        for i_n in range(len(n_levels)):
            for i_s in range(len(sr_levels)):
                ax.text(i_s, i_n, "%.0f" % M[i_n, i_s],
                        ha="center", va="center", fontsize=10, color="w")
        ax.set_xlabel("spectral radius")
        ax.set_ylabel("n")
        ax.set_title("a = %.3g\n(median VPT)" % aval)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[-1] = henrici_vpt_ax(axes[-1], d, "henrici_norm",
                              title="Exp 5: per-node Henrici departure vs VPT")
    fig.suptitle("Exp 5: n x spectral-radius grid at three Henrici levels",
                 y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "fig5_n_sr_pair.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote fig5_n_sr_pair.png")


# ================================================================= FIG 6 ===
def fig6():
    names = ["exp1_a_sweep", "exp2_n_vs_a", "exp3_sr_vs_a",
             "exp4_alpha_vs_a", "exp5_n_sr_pair"]
    data = [load(n) for n in names if
            os.path.exists(os.path.join(RESULTS, n + ".npz"))]
    x = np.concatenate([d["henrici_norm"].astype(float) for d in data])
    y = np.concatenate([d["vpt"].astype(float) for d in data])
    exp_id = np.concatenate([[i] * d["vpt"].size for i, d in enumerate(data)])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    cmap = plt.cm.get_cmap("tab10")
    for i, d in enumerate(data):
        xx = d["henrici_norm"].astype(float)
        yy = d["vpt"].astype(float)
        ax.scatter(xx, yy, s=18, alpha=0.5, color=cmap(i), edgecolors="none",
                   label=d["metadata"]["experiment"])
    xx = np.linspace(x.min(), x.max(), 100)
    ax.plot(xx, np.polyval(np.polyfit(x, y, 1), xx), "r-", lw=1.5,
            label="linear fit")
    xs, ys = rolling_median(x, y, n_bins=10)
    ax.plot(xs, ys, "k-", lw=2.4, label="median trend")
    rho = spearman(x, y)
    ax.annotate("Spearman rho = %.2f  (N=%d trials)" % (rho, x.size),
                xy=(0.04, 0.94), xycoords="axes fraction", fontsize=10,
                bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))
    ax.set_xlabel("Henrici departure / n  (per node)")
    ax.set_ylabel("VPT (steps)")
    ax.set_title("ALL experiments pooled: Henrici departure vs VPT")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    x2 = np.concatenate([d["henrici"].astype(float) for d in data])
    ax = axes[1]
    m = x2 > 0
    ax.scatter(x2[m], y[m], s=18, alpha=0.4, c="#1f77b4", edgecolors="none")
    xs, ys = rolling_median(x2[m], y[m], n_bins=10)
    ax.plot(xs, ys, "k-", lw=2.4, label="median trend")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Henrici departure (raw, log scale)")
    ax.set_ylabel("VPT (steps, log scale)")
    rho = spearman(x2[m], y[m])
    ax.annotate("Spearman rho = %.2f" % rho, xy=(0.04, 0.94),
                xycoords="axes fraction", fontsize=10,
                bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))
    ax.set_title("Raw Henrici departure vs VPT (log-log)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, which="both")

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "fig6_pooled_all.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote fig6_pooled_all.png")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    print("all figures written to", PLOTS)
