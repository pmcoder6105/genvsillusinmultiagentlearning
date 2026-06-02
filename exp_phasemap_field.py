"""
Experiment #6 — the headline map.

Two fields over payoff space (r, beta):
  (A) cycling fraction  -> the phase diagram (converge / bistable / cycle).
  (B) GENUINE anticipation = undecided-subset AUC at k=100 of the early Q-gap,
      computed only where the cell is bistable. This is the map of *where early
      prediction of equilibrium selection is real vs illusory* - the core claim.

Per cell: S seeds, T steps; record early Q-gap@100, commitment time, outcome.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.metrics import roc_auc_score

ALPHA, GAMMA = 0.1, 0.95
TAU0, DECAY, TAU_MIN = 1.0, 0.9998, 0.05
COMMIT_P, ENT_CYC, COMMIT_EVERY = 0.9, 0.4, 10

def cyc_matrix():
    C = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            d = (j - i) % 3
            C[i, j] = 1.0 if d == 1 else (-1.0 if d == 2 else 0.0)
    return C
CYC = cyc_matrix()

def payoffs(r, beta):
    base = r * (1 - np.eye(3)); return base + beta * CYC, base + beta * CYC.T

def softmax_rows(Z):
    Z = Z - Z.max(1, keepdims=True); E = np.exp(Z); return E / E.sum(1, keepdims=True)
def safe_entropy(p):
    c = np.zeros_like(p); m = p > 0; c[m] = p[m]*np.log2(p[m]); return -c.sum(1)
def lead_gap(Q):
    s = np.sort(Q, 1); return s[:, -1] - s[:, -2]

def cell(r, beta, S, T, seed):
    rng = np.random.default_rng(seed)
    A1, A2 = payoffs(r, beta)
    Q1 = rng.normal(0, 0.01, (S, 3)); Q2 = rng.normal(0, 0.01, (S, 3))
    idx = np.arange(S); window = T // 4
    c1 = np.zeros((S, 3)); c2 = np.zeros((S, 3))
    commit_t = np.full(S, T, float); committed = np.zeros(S, bool); gap100 = None
    for t in range(T):
        tau = max(TAU_MIN, TAU0 * DECAY ** t)
        a1 = np.argmax(Q1 / tau + rng.gumbel(size=(S, 3)), 1)
        a2 = np.argmax(Q2 / tau + rng.gumbel(size=(S, 3)), 1)
        Q1[idx, a1] += ALPHA * (A1[a1, a2] + GAMMA * Q1.max(1) - Q1[idx, a1])
        Q2[idx, a2] += ALPHA * (A2[a1, a2] + GAMMA * Q2.max(1) - Q2[idx, a2])
        if t + 1 == 100:
            gap100 = 0.5 * (lead_gap(Q1) + lead_gap(Q2))
        if t % COMMIT_EVERY == 0:
            now = (softmax_rows(Q1/tau).max(1) > COMMIT_P) & \
                  (softmax_rows(Q2/tau).max(1) > COMMIT_P)
            newly = now & (~committed); commit_t[newly] = t; committed |= now
        if t >= T - window:
            c1[idx, a1] += 1; c2[idx, a2] += 1
    p1 = c1 / c1.sum(1, keepdims=True); p2 = c2 / c2.sum(1, keepdims=True)
    ent = np.maximum(safe_entropy(p1), safe_entropy(p2))
    conv = (ent <= ENT_CYC).astype(int)
    und = commit_t > 100
    cyc = float((1 - conv).mean())
    # genuine anticipation: undecided-subset AUC@100, only if bistable + both classes
    gA = np.nan
    if 0.08 <= cyc <= 0.92:
        yU, xU = conv[und], gap100[und]
        if len(np.unique(yU)) > 1 and und.sum() >= 20:
            gA = roc_auc_score(yU, xU)
    return cyc, gA

def main():
    rs = np.round(np.linspace(0.70, 1.30, 13), 3)
    bs = np.round(np.linspace(0.00, 0.70, 8), 3)
    S, T = 220, 5000
    CYC_F = np.zeros((len(bs), len(rs))); GEN = np.full((len(bs), len(rs)), np.nan)
    for i, b in enumerate(bs):
        for j, r in enumerate(rs):
            cyc, gA = cell(r, b, S, T, seed=321)
            CYC_F[i, j] = cyc; GEN[i, j] = gA
        print(f"beta={b:.3f} done")

    np.set_printoptions(precision=2, suppress=True)
    print("\nCycling-fraction field (rows beta, cols r):\n", CYC_F)
    print("\nGenuine-anticipation AUC field (NaN = not bistable):\n", GEN)

    ext = [rs[0], rs[-1], bs[0], bs[-1]]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))

    # (A) phase diagram, 3 regimes
    reg = np.where(CYC_F < 0.08, 0, np.where(CYC_F > 0.92, 2, 1))
    cmap = ListedColormap(["#2c7fb8", "#fde08a", "#cb4b16"])
    axL.imshow(reg, origin="lower", aspect="auto", extent=ext, cmap=cmap,
               norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N))
    axL.set_title("(A) Phase diagram in payoff space")
    axL.set_xlabel("r (anti-coordination reward)"); axL.set_ylabel(r"$\beta$ (cyclic asymmetry)")
    from matplotlib.patches import Patch
    axL.legend(handles=[Patch(color="#2c7fb8", label="always converge"),
                        Patch(color="#fde08a", label="bistable"),
                        Patch(color="#cb4b16", label="always cycle")],
               loc="upper right", fontsize=8, framealpha=0.9)

    # (B) genuine-anticipation field over the bistable band
    im = axR.imshow(GEN, origin="lower", aspect="auto", extent=ext,
                    cmap="magma", vmin=0.5, vmax=1.0)
    axR.set_title("(B) Genuine anticipation: undecided AUC@100\n(only on bistable cells)")
    axR.set_xlabel("r (anti-coordination reward)"); axR.set_ylabel(r"$\beta$ (cyclic asymmetry)")
    cb = fig.colorbar(im, ax=axR); cb.set_label("undecided-subset AUC at k=100")
    fig.tight_layout(); fig.savefig("fig_anticipation_map.png", dpi=140)
    plt.close(fig)
    print("\nSaved fig_anticipation_map.png")

if __name__ == "__main__":
    main()
