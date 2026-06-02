"""
#2 scale-free feature + #3 bootstrap CIs (Q-learning setting, k=100).

The reward-axis transport gap (#4) came from the raw Q-gap having different
magnitudes at different r. We test a scale-free alternative: the POLICY MARGIN
(p_max - p_2nd) from the Boltzmann policy, which is bounded in [0,1] and should
transport across r without per-game standardization. We compare leave-one-game-out
transfer (transferred threshold, undecided subset) for raw-gap vs policy-margin,
with bootstrap 95% CIs on held-out balanced accuracy.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

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
def lead_gap(M):
    s = np.sort(M, 1); return s[:, -1] - s[:, -2]

def run_game(r, beta, S, T, seed):
    rng = np.random.default_rng(seed)
    A1, A2 = payoffs(r, beta)
    Q1 = rng.normal(0, 0.01, (S, 3)); Q2 = rng.normal(0, 0.01, (S, 3))
    idx = np.arange(S); window = T // 4
    c1 = np.zeros((S, 3)); c2 = np.zeros((S, 3))
    commit_t = np.full(S, T, float); committed = np.zeros(S, bool); feat = None
    for t in range(T):
        tau = max(TAU_MIN, TAU0 * DECAY ** t)
        pol1 = softmax_rows(Q1 / tau); pol2 = softmax_rows(Q2 / tau)
        a1 = np.argmax(Q1 / tau + rng.gumbel(size=(S, 3)), 1)
        a2 = np.argmax(Q2 / tau + rng.gumbel(size=(S, 3)), 1)
        Q1[idx, a1] += ALPHA * (A1[a1, a2] + GAMMA * Q1.max(1) - Q1[idx, a1])
        Q2[idx, a2] += ALPHA * (A2[a1, a2] + GAMMA * Q2.max(1) - Q2[idx, a2])
        if t + 1 == 100:
            gap = 0.5 * (lead_gap(Q1) + lead_gap(Q2))
            margin = 0.5 * (lead_gap(pol1) + lead_gap(pol2))   # p_max - p_2nd, in [0,1]
            feat = np.column_stack([gap, margin])
        if t % COMMIT_EVERY == 0:
            now = (pol1.max(1) > COMMIT_P) & (pol2.max(1) > COMMIT_P)
            newly = now & (~committed); commit_t[newly] = t; committed |= now
        if t >= T - window:
            c1[idx, a1] += 1; c2[idx, a2] += 1
    pe1 = c1 / c1.sum(1, keepdims=True); pe2 = c2 / c2.sum(1, keepdims=True)
    conv = (np.maximum(safe_entropy(pe1), safe_entropy(pe2)) <= ENT_CYC).astype(int)
    return dict(y=conv, commit_t=commit_t, feat=feat, cyc=float((1 - conv).mean()))

def auc(y, x):
    return roc_auc_score(y, x) if len(np.unique(y)) > 1 else np.nan

def boot_ci(fn, y, p, thr=None, n=1000, seed=0):
    rng = np.random.default_rng(seed); N = len(y); vals = []
    for _ in range(n):
        ii = rng.integers(0, N, N)
        ys, ps = y[ii], p[ii]
        if len(np.unique(ys)) < 2: continue
        vals.append(fn(ys, (ps > thr).astype(int)) if thr is not None else fn(ys, ps))
    return (np.percentile(vals, 2.5), np.percentile(vals, 97.5)) if vals else (np.nan, np.nan)

def main():
    cand = [(0.875, 0.00), (0.875, 0.10), (1.00, 0.00), (1.00, 0.05),
            (1.00, 0.10), (1.125, 0.10), (1.00, 0.20)]
    S, T = 300, 8000
    data = {}
    for (r, b) in cand:
        d = run_game(r, b, S, T, seed=7)
        if 0.08 <= d["cyc"] <= 0.92:
            data[(r, b)] = d
    games = list(data)
    print(f"Using {len(games)} bistable games.\n")

    def transfer(col, test_g):
        tr = [g for g in games if g != test_g]
        Xtr = np.vstack([data[g]["feat"][:, [col]] for g in tr])
        ytr = np.concatenate([data[g]["y"] for g in tr])
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr), ytr)
        ptr = clf.predict_proba(sc.transform(Xtr))[:, 1]
        ths = np.linspace(0.1, 0.9, 81)
        thr = ths[np.argmax([balanced_accuracy_score(ytr, ptr > th) for th in ths])]
        und = data[test_g]["commit_t"] > 100
        Xte = data[test_g]["feat"][und][:, [col]]; yte = data[test_g]["y"][und]
        pte = clf.predict_proba(sc.transform(Xte))[:, 1]
        ba = balanced_accuracy_score(yte, pte > thr) if len(np.unique(yte)) > 1 else np.nan
        lo, hi = boot_ci(balanced_accuracy_score, yte, pte, thr=thr)
        return ba, (lo, hi)

    print("[#2/#3] Leave-one-game-out transfer with 95% bootstrap CI (held-out balanced accuracy):")
    print(f"{'held-out':>16} {'cyc':>5} | {'raw-gap (95% CI)':>26} | {'policy-margin (95% CI)':>28}")
    for g in games:
        ba_g, ci_g = transfer(0, g)
        ba_m, ci_m = transfer(1, g)
        print(f"{str(g):>16} {data[g]['cyc']:5.2f} | "
              f"{ba_g:6.3f} [{ci_g[0]:.3f},{ci_g[1]:.3f}]   | "
              f"{ba_m:6.3f} [{ci_m[0]:.3f},{ci_m[1]:.3f}]")

    # #3: bootstrap CI on representative map-cell genuine AUC (undecided subset)
    print("\n[#3] Genuine anticipation AUC@100 with 95% bootstrap CI (representative cells):")
    for g in [min(games, key=lambda x: abs(data[x]['cyc']-0.5)),  # balanced/noise-dom
              min(games, key=lambda x: abs(data[x]['cyc']-0.2))]: # structured-ish
        d = data[g]; und = d["commit_t"] > 100
        yU, xU = d["y"][und], d["feat"][und, 0]
        a = auc(yU, xU); lo, hi = boot_ci(roc_auc_score, yU, xU)
        print(f"   {str(g):>14} (cyc={d['cyc']:.2f}): AUC={a:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

if __name__ == "__main__":
    main()
