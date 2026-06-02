"""
Experiments #4 (generalization) and #5 (beat the balanced-regime ceiling).

Shared harness with the pilot. For a set of bistable games spanning the band we
record, per run, at early horizons:
  - gap100  : mean leading Q-gap across agents at t=100  (pilot baseline feature)
  - gap50   : same at t=50
  - slope   : (gap100 - gap50)/50  -> early *rate* of symmetry-breaking  (#5)
  - pent100 : mean policy entropy at t=100  (low = already tilting)
plus commitment time and eventual outcome.

#5 ceiling test: in the MOST balanced game, compare undecided-subset AUC@100 of
  gap100 alone (baseline ~0.66) vs slope alone vs the 3-feature multivariate model.
  "Undecided" = commit_t > 100, so this is genuine anticipation, not leakage.

#4 generalization: leave-one-game-out. Standardize on train, fit logistic
  regression, choose the decision threshold on TRAIN (max balanced accuracy),
  apply to the held-out game's UNDECIDED runs. Report transferred balanced
  accuracy + AUC. Compare gap-only vs multivariate. This actually tests whether
  a single decision surface transports across the family (the pilot's 1-D AUC
  transfer was degenerate and did not).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

ALPHA, GAMMA = 0.1, 0.95
TAU0, DECAY, TAU_MIN = 1.0, 0.9998, 0.05
COMMIT_P, ENT_CYC = 0.9, 0.4
COMMIT_EVERY = 10            # check commitment on a coarse grid (commit ~240)

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

def softmax_rows(Z):
    Z = Z - Z.max(1, keepdims=True); E = np.exp(Z); return E / E.sum(1, keepdims=True)

def safe_entropy(p):
    c = np.zeros_like(p); m = p > 0; c[m] = p[m] * np.log2(p[m]); return -c.sum(1)

def lead_gap(Q):
    s = np.sort(Q, 1); return s[:, -1] - s[:, -2]

def run_game(r, beta, S, T, seed):
    rng = np.random.default_rng(seed)
    A1, A2 = payoffs(r, beta)
    Q1 = rng.normal(0, 0.01, (S, 3)); Q2 = rng.normal(0, 0.01, (S, 3))
    idx = np.arange(S); window = T // 4
    counts1 = np.zeros((S, 3)); counts2 = np.zeros((S, 3))
    commit_t = np.full(S, T, float); committed = np.zeros(S, bool)
    snap = {}
    for t in range(T):
        tau = max(TAU_MIN, TAU0 * DECAY ** t)
        a1 = np.argmax(Q1 / tau + rng.gumbel(size=(S, 3)), 1)
        a2 = np.argmax(Q2 / tau + rng.gumbel(size=(S, 3)), 1)
        Q1[idx, a1] += ALPHA * (A1[a1, a2] + GAMMA * Q1.max(1) - Q1[idx, a1])
        Q2[idx, a2] += ALPHA * (A2[a1, a2] + GAMMA * Q2.max(1) - Q2[idx, a2])
        if (t + 1) in (50, 100):
            gap = 0.5 * (lead_gap(Q1) + lead_gap(Q2))
            pe = 0.5 * (safe_entropy(softmax_rows(Q1 / tau)) +
                        safe_entropy(softmax_rows(Q2 / tau)))
            snap[t + 1] = (gap.copy(), pe.copy())
        if t % COMMIT_EVERY == 0:
            p1 = softmax_rows(Q1 / tau); p2 = softmax_rows(Q2 / tau)
            now = (p1.max(1) > COMMIT_P) & (p2.max(1) > COMMIT_P)
            newly = now & (~committed); commit_t[newly] = t; committed |= now
        if t >= T - window:
            counts1[idx, a1] += 1; counts2[idx, a2] += 1
    p1f = counts1 / counts1.sum(1, keepdims=True)
    p2f = counts2 / counts2.sum(1, keepdims=True)
    ent = np.maximum(safe_entropy(p1f), safe_entropy(p2f))
    conv = (ent <= ENT_CYC).astype(int)
    gap100, pent100 = snap[100]; gap50, _ = snap[50]
    feats = np.column_stack([gap100, (gap100 - gap50) / 50.0, pent100])  # gap, slope, pent
    return dict(y=conv, commit_t=commit_t, feats=feats,
                cyc=float((1 - conv).mean()))

def auc(y, x):
    return roc_auc_score(y, x) if len(np.unique(y)) > 1 else np.nan

def main():
    # candidate bistable games spanning the band (verified intermediate cycling)
    cand = [(0.875, 0.00), (0.875, 0.10), (1.00, 0.00), (1.00, 0.05),
            (1.00, 0.10), (1.125, 0.10), (1.00, 0.20), (1.00, 0.30)]
    S, T = 300, 8000
    data = {}
    print("Running games (S=%d, T=%d)..." % (S, T))
    for (r, b) in cand:
        d = run_game(r, b, S=S, T=T, seed=7)
        if 0.08 <= d["cyc"] <= 0.92:        # keep genuinely bistable
            data[(r, b)] = d
        print(f"  r={r:.3f} b={b:.2f}  cycling={d['cyc']:.3f}  "
              f"{'KEEP' if (r,b) in data else 'skip (not bistable)'}")
    games = list(data)
    print(f"\nUsing {len(games)} bistable games.")

    # ---------- #5 ceiling test on the MOST balanced game ----------
    most_bal = min(games, key=lambda g: abs(data[g]["cyc"] - 0.5))
    d = data[most_bal]; undec = d["commit_t"] > 100
    yU = d["y"][undec]; FU = d["feats"][undec]
    print(f"\n[#5] Ceiling test on most-balanced game r={most_bal[0]}, "
          f"b={most_bal[1]} (cycling={d['cyc']:.3f}); "
          f"undecided n={undec.sum()} of {S}")
    print(f"  undecided-subset AUC  gap-only   : {auc(yU, FU[:,0]):.3f}")
    print(f"  undecided-subset AUC  slope-only : {auc(yU, FU[:,1]):.3f}")
    # multivariate CV AUC on undecided subset
    if len(np.unique(yU)) > 1:
        clf = LogisticRegression(max_iter=1000)
        proba = cross_val_predict(clf, StandardScaler().fit_transform(FU), yU,
                                  cv=5, method="predict_proba")[:, 1]
        print(f"  undecided-subset AUC  multivariate(gap+slope+pent), 5-fold CV: "
              f"{auc(yU, proba):.3f}")

    # ---------- #4 leave-one-game-out transfer ----------
    print("\n[#4] Leave-one-game-out transfer "
          "(threshold tuned on train, evaluated on held-out UNDECIDED runs):")
    print(f"{'held-out game':>18} {'cyc':>5} | {'gap-only':>17} | {'multivariate':>17}")
    print(f"{'':>18} {'':>5} | {'balAcc':>8} {'AUC':>8} | {'balAcc':>8} {'AUC':>8}")
    def transfer(use_cols, test_g):
        train_g = [g for g in games if g != test_g]
        Xtr = np.vstack([data[g]["feats"][:, use_cols] for g in train_g])
        ytr = np.concatenate([data[g]["y"] for g in train_g])
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr), ytr)
        # threshold maximizing balanced accuracy on train
        ptr = clf.predict_proba(sc.transform(Xtr))[:, 1]
        ths = np.linspace(0.1, 0.9, 81)
        thr = ths[np.argmax([balanced_accuracy_score(ytr, ptr > th) for th in ths])]
        und = data[test_g]["commit_t"] > 100
        Xte = data[test_g]["feats"][und][:, use_cols]; yte = data[test_g]["y"][und]
        pte = clf.predict_proba(sc.transform(Xte))[:, 1]
        ba = balanced_accuracy_score(yte, pte > thr) if len(np.unique(yte)) > 1 else np.nan
        return ba, auc(yte, pte)
    for g in games:
        ba1, au1 = transfer([0], g)            # gap only
        ba3, au3 = transfer([0, 1, 2], g)      # multivariate
        print(f"{str(g):>18} {data[g]['cyc']:5.2f} | "
              f"{ba1:8.3f} {au1:8.3f} | {ba3:8.3f} {au3:8.3f}")

if __name__ == "__main__":
    main()
