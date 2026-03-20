"""Scan k=2..10 with hmmlearn Gaussian HMM on (y_t,y_{t-1}) vs MS-AR(1) truth (k_true=5). Fast, not full MS-AR MLE."""
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings("ignore", category=RuntimeWarning)

SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_DIR = SCRIPT_DIR.parent / "plots" / "models"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def sim(T, k, P, rho, sig, seed=0):
    rng = np.random.default_rng(seed)
    s = np.zeros(T, dtype=int)
    y = np.zeros(T)
    s[0] = rng.integers(k)
    y[0] = rng.normal(0, sig[s[0]])
    for t in range(1, T):
        s[t] = rng.choice(k, p=P[s[t - 1]])
        j = s[t]
        y[t] = rho[j] * y[t - 1] + rng.normal(scale=sig[j])
    return y, s


def pointwise_match(true, pred, k_true, k_pred):
    """Map each predicted label j to argmax_i count(true=i|pred=j); return mask (mapped_pred == true)."""
    C = np.zeros((k_true, k_pred), dtype=int)
    for t in range(len(true)):
        C[true[t], pred[t]] += 1
    pred_to_true = np.zeros(k_pred, dtype=int)
    for j in range(k_pred):
        col = C[:, j]
        pred_to_true[j] = int(np.argmax(col)) if col.sum() > 0 else 0
    mapped = pred_to_true[pred]
    return mapped == true


def overlay_correct(ax, ok):
    """Green/red background from merged runs over indices [0, n)."""
    n = len(ok)
    edges = np.arange(n + 1, dtype=float)
    t = 0
    while t < n:
        good = ok[t]
        u = t + 1
        while u < n and ok[u] == good:
            u += 1
        color = (0.2, 0.65, 0.35, 0.32) if good else (0.85, 0.25, 0.2, 0.32)
        ax.axvspan(edges[t], edges[u], facecolor=color[:3], alpha=color[3], lw=0, zorder=0)
        t = u
    ax.set_xlim(0, n)


def main(show: bool = True):
    k_true = 5
    T = 2600
    # Separated AR and noise across states so 2D Gaussian HMM has a fighting chance
    rho = np.array([-0.75, -0.15, 0.45, 0.78, 0.05])
    sig = np.array([0.32, 0.55, 0.42, 0.38, 0.85])
    u = 0.91
    P = u * np.eye(k_true) + (1 - u) / k_true * np.ones((k_true, k_true))
    P /= P.sum(axis=1, keepdims=True)

    y, st = sim(T, k_true, P, rho, sig, seed=42)
    X = np.column_stack([y[1:], y[:-1]])
    z = st[1:]

    ks = list(range(2, 11))
    aris, bics = [], []
    models = {}

    for k in ks:
        h = GaussianHMM(
            n_components=k,
            covariance_type="diag",
            n_iter=60,
            tol=5e-2,
            random_state=17 * k,
            init_params="stmc",
        )
        h.fit(X)
        models[k] = h
        pred = h.predict(X)
        aris.append(adjusted_rand_score(z, pred))
        bics.append(h.bic(X))

    k_bic = ks[int(np.argmin(bics))]
    k_ari = ks[int(np.argmax(aris))]

    print("k_true =", k_true)
    print("k (best BIC) =", k_bic, "  k (best ARI) =", k_ari)
    print("BIC pick == k_true?", k_bic == k_true, "  ARI-max == k_true?", k_ari == k_true)
    print("(Proxy: Gaussian HMM on (y_t,y_{t-1}), not MS-AR MLE. BIC/ARI are exploratory.)")
    for k, a, b in zip(ks, aris, bics):
        tag = " *" if k == k_true else ""
        print(f"  k={k:2d}  ARI={a:.3f}  BIC={b:10.1f}{tag}")

    # --- fig 1: metrics vs k ---
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(ks, aris, "o-", color="tab:blue", lw=2, ms=6, label="ARI (vs true states)")
    ax1.axvline(k_true, color="tab:green", ls="--", lw=1.5, label=f"true k={k_true}")
    ax1.set_xlabel("k (fitted states)")
    ax1.set_ylabel("adjusted Rand index")
    ax1.set_xticks(ks)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(ks, bics, "s-", color="tab:red", lw=2, ms=5, alpha=0.85, label="BIC")
    ax2.set_ylabel("BIC")
    ax2.scatter([k_bic], [bics[ks.index(k_bic)]], color="darkred", s=100, zorder=5, marker="*", label=f"min BIC → k={k_bic}")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center left", bbox_to_anchor=(0.02, 0.35), fontsize=8, framealpha=0.9)

    fig1.suptitle("MS-AR(1) truth vs Gaussian HMM(k) on (y_t, y_{t-1}) — proxy fit, see BIC/ARI")
    fig1.subplots_adjust(top=0.88)
    p1 = PLOT_DIR / "kscan_metrics.pdf"
    fig1.savefig(p1, bbox_inches="tight")
    print("saved", p1)

    # --- fig 2: state paths (subset): under / match / over ---
    t0, t1 = 250, 450
    fig2, axes = plt.subplots(5, 1, figsize=(10, 8.5), sharex=True, layout="constrained")
    tt = np.arange(t0, t1)

    axes[0].plot(tt, y[t0:t1], "k-", lw=0.8)
    axes[0].set_ylabel("y")
    axes[0].set_title(f"y  (DGP: MS-AR(1), k_true={k_true})")

    axes[1].step(tt, z[t0:t1], where="mid", color="tab:green", lw=1.2)
    axes[1].set_ylabel("true")
    axes[1].set_yticks(range(k_true))

    for ax, k, c, lab in [
        (axes[2], 2, "tab:orange", "under"),
        (axes[3], k_true, "tab:blue", "match"),
        (axes[4], 10, "tab:purple", "over"),
    ]:
        pred = models[k].predict(X)[t0:t1]
        ari_w = adjusted_rand_score(z[t0:t1], pred)
        ax.step(tt, pred, where="mid", color=c, lw=1.0)
        ax.set_ylabel(f"k={k}")
        ax.set_title(f"Viterbi {lab}  (ARI={ari_w:.2f})")

    axes[-1].set_xlabel("t")
    fig2.suptitle("State paths (same window)", fontsize=11)
    p2 = PLOT_DIR / "kscan_paths.pdf"
    fig2.savefig(p2, bbox_inches="tight")
    print("saved", p2)

    # --- fig 3: y with green/red overlay for each k (single tall PDF) ---
    n = len(z)
    x = np.arange(n)
    n_ks = len(ks)
    fig3, axes3 = plt.subplots(
        n_ks,
        1,
        figsize=(11, 1.35 * n_ks + 0.8),
        sharex=True,
        layout="constrained",
    )
    if n_ks == 1:
        axes3 = [axes3]
    leg = [
        Patch(facecolor=(0.2, 0.65, 0.35), alpha=0.45, edgecolor="none", label="match true state (after label map)"),
        Patch(facecolor=(0.85, 0.25, 0.2), alpha=0.45, edgecolor="none", label="mismatch"),
    ]
    for ax, k in zip(axes3, ks):
        pred = models[k].predict(X)
        ok = pointwise_match(z, pred, k_true, k)
        overlay_correct(ax, ok)
        ax.plot(x, y[1:], color="black", lw=0.45, zorder=2)
        ax.set_ylabel("y", fontsize=8)
        ax.set_title(
            f"fitted k={k}   ARI={adjusted_rand_score(z, pred):.3f}   "
            f"pointwise acc={ok.mean():.2f}",
            fontsize=9,
            loc="left",
        )
        ax.legend(handles=leg, loc="lower left", ncol=2, fontsize=6, framealpha=0.9)
        ax.grid(True, alpha=0.15, zorder=1)
    axes3[-1].set_xlabel("t (aligned with y_t, t≥1)")
    fig3.suptitle(
        "Simulated y: background = Viterbi vs truth (per-column majority label map)",
        fontsize=11,
    )
    p3 = PLOT_DIR / "kscan_y_overlay.pdf"
    fig3.savefig(p3, bbox_inches="tight")
    print("saved", p3)

    if show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main(show="--no-show" not in sys.argv)
