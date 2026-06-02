"""
Stage B/C — Direction 1 pilot: lead-time, leakage control, generalization.

Builds on Stage A (which located a bistable band at r~1.0). Here we:
  (1) Render a finer phase map over (r, beta) for the figure.
  (2) For each of 3 bistable games, run S seeds x T steps and record, per run:
        - value-function asymmetry (mean leading Q-gap across agents) at a set
          of early horizons k,
        - commitment time (first t where both agents' Boltzmann policy has
          max-prob > 0.9),
        - eventual outcome (converged vs cycling).
  (3) Lead-time curve: AUC(k) of asymmetry@k predicting convergence, vs k,
      with the commitment-time distribution overlaid. The Catch-22 objection
      ("you detect commitment, not anticipation") fails iff AUC is already high
      at horizons well below the commitment times.
  (4) Undecided-subset control: recompute AUC(k) restricting to runs NOT yet
      committed by time k. If this stays high => genuine anticipation.
  (5) Generalization: (a) within-game AUC of the raw asymmetry feature in every
      bistable game; (b) cross-game transfer - train logistic regression on one
      game, test AUC on the others (feature z-scored per game).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

ALPHA, GAMMA = 0.1, 0.95
TAU0, DECAY, TAU_MIN = 1.0, 0.9998, 0.05
COMMIT_P = 0.9          # policy max-prob threshold defining "committed"
ENT_CYC = 0.4           # final-window entropy (bits) above which => cycling

def cyc_matrix():
    C = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            d = (j - i) % 3
            C[i, j] = 1.0 if d == 1 else (-1.0 if d == 2 else 0.0)
    return C
CYC = cyc_matrix()

def payoffs(r, beta):
    base = r * (1 - np.eye(3))
    return base + beta * CYC, base + beta * CYC.T

def safe_entropy(p):
    out = np.zeros(p.shape[0])
    mask = p > 0
    contrib = np.zeros_like(p)
    contrib[mask] = p[mask] * np.log2(p[mask])
    return -contrib.sum(1)

def softmax_rows(Z):
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)

def run_game(r, beta, S, T, seed, horizons=None, window=None, record=False):
    if window is None:
        window = T // 4
    rng = np.random.default_rng(seed)
    A1, A2 = payoffs(r, beta)
    Q1 = rng.normal(0, 0.01, (S, 3)); Q2 = rng.normal(0, 0.01, (S, 3))
    idx = np.arange(S)
    counts1 = np.zeros((S, 3)); counts2 = np.zeros((S, 3))
    commit_t = np.full(S, T, dtype=float)      # default: never committed
    committed = np.zeros(S, dtype=bool)
    feats = {} if record else None
    hset = set(horizons) if horizons else set()
    for t in range(T):
        tau = max(TAU_MIN, TAU0 * DECAY ** t)
        a1 = np.argmax(Q1 / tau + rng.gumbel(size=(S, 3)), axis=1)
        a2 = np.argmax(Q2 / tau + rng.gumbel(size=(S, 3)), axis=1)
        Q1[idx, a1] += ALPHA * (A1[a1, a2] + GAMMA * Q1.max(1) - Q1[idx, a1])
        Q2[idx, a2] += ALPHA * (A2[a1, a2] + GAMMA * Q2.max(1) - Q2[idx, a2])
        # commitment check (on current Boltzmann policy)
        p1 = softmax_rows(Q1 / tau); p2 = softmax_rows(Q2 / tau)
        now = (p1.max(1) > COMMIT_P) & (p2.max(1) > COMMIT_P)
        newly = now & (~committed)
        commit_t[newly] = t; committed |= now
        if record and (t + 1) in hset:
            s1 = np.sort(Q1, 1); s2 = np.sort(Q2, 1)
            gap = 0.5 * ((s1[:, -1] - s1[:, -2]) + (s2[:, -1] - s2[:, -2]))
            feats[t + 1] = gap.copy()
        if t >= T - window:
            counts1[idx, a1] += 1; counts2[idx, a2] += 1
    p1f = counts1 / counts1.sum(1, keepdims=True)
    p2f = counts2 / counts2.sum(1, keepdims=True)
    ent = np.maximum(safe_entropy(p1f), safe_entropy(p2f))
    converged = (ent <= ENT_CYC).astype(int)    # 1 = converged, 0 = cycling
    return dict(converged=converged, commit_t=commit_t, feats=feats,
                cyc_frac=float((1 - converged).mean()))

def auc_safe(y, x):
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, x)

# ---------------------------------------------------------------- (1) phase map
def phase_map():
    rs = np.linspace(0.5, 1.5, 9)
    betas = np.linspace(0.0, 0.8, 9)
    M = np.zeros((len(betas), len(rs)))
    for i, b in enumerate(betas):
        for j, r in enumerate(rs):
            M[i, j] = run_game(r, b, S=50, T=2500, seed=999)["cyc_frac"]
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    im = ax.imshow(M, origin="lower", aspect="auto", cmap="viridis",
                   extent=[rs[0], rs[-1], betas[0], betas[-1]], vmin=0, vmax=1)
    ax.set_xlabel("r  (anti-coordination reward)")
    ax.set_ylabel(r"$\beta$  (cyclic asymmetry)")
    ax.set_title("Cycling fraction across payoff space\n(intermediate band = bistable)")
    cb = fig.colorbar(im, ax=ax); cb.set_label("fraction of runs cycling")
    fig.tight_layout(); fig.savefig("fig_phasemap.png", dpi=130)
    plt.close(fig)
    return rs, betas, M

# --------------------------------------------------- (2-5) lead-time + transfer
def main():
    rs, betas, M = phase_map()
    print("Phase map rendered (fig_phasemap.png).")
    print("Cycling-fraction grid (rows=beta 0..0.8, cols=r 0.5..1.5):")
    np.set_printoptions(precision=2, suppress=True)
    print(M)

    games = {"G1 (r=1.0,b=0.10)": (1.0, 0.10),
             "G2 (r=1.0,b=0.20)": (1.0, 0.20),
             "G3 (r=1.0,b=0.30)": (1.0, 0.30)}
    horizons = [25, 50, 75, 100, 150, 200, 250, 300, 400, 600]
    S, T = 400, 10000
    data = {}
    for name, (r, b) in games.items():
        out = run_game(r, b, S=S, T=T, seed=2024, horizons=horizons, record=True)
        data[name] = out
        cf = out["cyc_frac"]
        # commitment time among convergent runs
        ct = out["commit_t"][out["converged"] == 1]
        med_ct = np.median(ct) if len(ct) else np.nan
        print(f"\n=== {name} ===  cycling={cf:.3f}  "
              f"n_conv={int(out['converged'].sum())}/{S}  "
              f"median commit (conv runs)={med_ct:.0f}")
        # full-sample AUC across horizons
        print(f"{'k':>5} {'AUC_full':>9} {'AUC_undecided':>14} {'n_undecided':>12}")
        for k in horizons:
            y = out["converged"]; x = out["feats"][k]
            auc_full = auc_safe(y, x)
            undec = out["commit_t"] > k
            auc_und = auc_safe(y[undec], x[undec])
            print(f"{k:>5} {auc_full:9.3f} {auc_und:14.3f} {int(undec.sum()):12d}")

    # lead-time figure (one panel per game)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, (name, out) in zip(axes, data.items()):
        y = out["converged"]
        auc_full = [auc_safe(y, out["feats"][k]) for k in horizons]
        auc_und = []
        for k in horizons:
            undec = out["commit_t"] > k
            auc_und.append(auc_safe(y[undec], out["feats"][k][undec]))
        ax.plot(horizons, auc_full, "o-", label="AUC (all runs)")
        ax.plot(horizons, auc_und, "s--", label="AUC (undecided subset)")
        ax.axhline(0.5, color="grey", lw=0.8, ls=":")
        ct = out["commit_t"][out["converged"] == 1]
        if len(ct):
            ax.axvline(np.median(ct), color="crimson", lw=1.2,
                       label=f"median commit={np.median(ct):.0f}")
        ax.set_title(name); ax.set_xlabel("measurement horizon k")
        ax.set_xscale("log")
    axes[0].set_ylabel("AUC (predict convergence)")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Lead-time: early value-asymmetry vs commitment", y=1.02)
    fig.tight_layout(); fig.savefig("fig_leadtime.png", dpi=130,
                                    bbox_inches="tight")
    plt.close(fig)
    print("\nLead-time figure rendered (fig_leadtime.png).")

    # ---- generalization: cross-game transfer at k=100, feature z-scored / game
    print("\n--- Cross-game transfer (train rows -> test cols), AUC at k=100 ---")
    names = list(games)
    def z(v): 
        s = v.std(); 
        return (v - v.mean()) / (s if s > 0 else 1.0)
    feat100 = {n: z(data[n]["feats"][100]) for n in names}
    ycg = {n: data[n]["converged"] for n in names}
    header = "train\\test " + " ".join(f"{n[:2]:>8}" for n in names)
    print(header)
    for tr in names:
        row = [f"{tr[:9]:>9}"]
        clf = LogisticRegression().fit(feat100[tr].reshape(-1, 1), ycg[tr])
        for te in names:
            prob = clf.predict_proba(feat100[te].reshape(-1, 1))[:, 1]
            row.append(f"{auc_safe(ycg[te], prob):8.3f}")
        print(" ".join(row))

if __name__ == "__main__":
    main()
