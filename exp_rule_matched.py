"""
#1 (rule-matched horizon) + #4 (third rule, FAQ) in one clean experiment.

Three learning rules, same anti-coordination family:
  - 'q'   : Boltzmann Q-learning (annealed tau)        [value-based]
  - 'faq' : Frequency-Adjusted Q-learning (lr/p_a)     [value-based, replicator-aligned]
  - 'pg'  : softmax policy gradient (REINFORCE+baseline)[policy-based, value-free]

Predictor = leading gap in the internal preference variable (Q for q/faq,
logit for pg), averaged across agents.

Key fix vs. earlier #7: instead of measuring at the SAME absolute step (t=100),
we measure at a horizon matched to each rule's own commitment timescale:
  k_rule(f) = round(f * median commit time of convergent runs under that rule).
This separates "measured too early for this rule" from "intrinsically weaker".

Output: for each rule, median commit time, and the genuine (undecided-subset)
AUC contrast between noise-dominated (low-beta) and structured (high-beta)
bistable cells, evaluated at matched fractions f.
"""
import numpy as np
from sklearn.metrics import roc_auc_score

GAMMA = 0.95
ALPHA_Q = 0.1
TAU0, DECAY, TAU_MIN = 1.0, 0.9998, 0.05
ALPHA_FAQ, FAQ_CAP = 0.1, 0.5
LR_PG, BW_PG = 0.05, 0.01
ENT_CYC, COMMIT_P, COMMIT_EVERY = 0.4, 0.9, 10
HORIZONS = [25, 50, 75, 100, 150, 200, 250, 300, 400]

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

def run(rule, r, beta, S, T, seed):
    rng = np.random.default_rng(seed)
    A1, A2 = payoffs(r, beta)
    P1 = rng.normal(0, 0.01, (S, 3)); P2 = rng.normal(0, 0.01, (S, 3))  # Q or logits
    b1 = np.zeros(S); b2 = np.zeros(S)                                   # pg baselines
    idx = np.arange(S); window = T // 4
    c1 = np.zeros((S, 3)); c2 = np.zeros((S, 3))
    commit_t = np.full(S, T, float); committed = np.zeros(S, bool)
    snaps = {}
    for t in range(T):
        if rule in ("q", "faq"):
            tau = max(TAU_MIN, TAU0 * DECAY ** t)
            pol1 = softmax_rows(P1 / tau); pol2 = softmax_rows(P2 / tau)
            a1 = np.argmax(P1 / tau + rng.gumbel(size=(S, 3)), 1)
            a2 = np.argmax(P2 / tau + rng.gumbel(size=(S, 3)), 1)
            r1 = A1[a1, a2]; r2 = A2[a1, a2]
            td1 = r1 + GAMMA * P1.max(1) - P1[idx, a1]
            td2 = r2 + GAMMA * P2.max(1) - P2[idx, a2]
            if rule == "q":
                P1[idx, a1] += ALPHA_Q * td1; P2[idx, a2] += ALPHA_Q * td2
            else:  # faq: scale lr by 1/p_a, capped
                e1 = np.minimum(ALPHA_FAQ / np.maximum(pol1[idx, a1], 1e-3), FAQ_CAP)
                e2 = np.minimum(ALPHA_FAQ / np.maximum(pol2[idx, a2], 1e-3), FAQ_CAP)
                P1[idx, a1] += e1 * td1; P2[idx, a2] += e2 * td2
        else:  # pg
            pol1 = softmax_rows(P1); pol2 = softmax_rows(P2)
            a1 = (np.cumsum(pol1, 1) > rng.random((S, 1))).argmax(1)
            a2 = (np.cumsum(pol2, 1) > rng.random((S, 1))).argmax(1)
            r1 = A1[a1, a2]; r2 = A2[a1, a2]
            oh1 = np.zeros((S, 3)); oh1[idx, a1] = 1; oh2 = np.zeros((S, 3)); oh2[idx, a2] = 1
            P1 += LR_PG * (r1 - b1)[:, None] * (oh1 - pol1)
            P2 += LR_PG * (r2 - b2)[:, None] * (oh2 - pol2)
            b1 += BW_PG * (r1 - b1); b2 += BW_PG * (r2 - b2)
        if (t + 1) in HORIZONS:
            snaps[t + 1] = 0.5 * (lead_gap(P1) + lead_gap(P2))
        if t % COMMIT_EVERY == 0:
            now = (pol1.max(1) > COMMIT_P) & (pol2.max(1) > COMMIT_P)
            newly = now & (~committed); commit_t[newly] = t; committed |= now
        if t >= T - window:
            c1[idx, a1] += 1; c2[idx, a2] += 1
    pe1 = c1 / c1.sum(1, keepdims=True); pe2 = c2 / c2.sum(1, keepdims=True)
    conv = (np.maximum(safe_entropy(pe1), safe_entropy(pe2)) <= ENT_CYC).astype(int)
    return dict(y=conv, commit_t=commit_t, snaps=snaps,
                cyc=float((1 - conv).mean()))

def auc(y, x):
    return roc_auc_score(y, x) if len(np.unique(y)) > 1 else np.nan

def nearest_h(k):
    return min(HORIZONS, key=lambda h: abs(h - k))

def main():
    low_cells  = [(0.85, 0.0), (1.0, 0.0), (1.15, 0.0), (1.30, 0.0)]   # noise-dominated
    high_cells = [(0.95, 0.3), (1.05, 0.3), (0.95, 0.4), (1.05, 0.4)]  # structured
    S, T = 250, 8000
    for rule in ("q", "faq", "pg"):
        cells = {}
        for (r, b) in low_cells + high_cells:
            cells[(r, b)] = run(rule, r, b, S, T, seed=88)
        # median commit over convergent runs, pooled across this rule's bistable cells
        cts = []
        for d in cells.values():
            if 0.08 <= d["cyc"] <= 0.92:
                cts += list(d["commit_t"][(d["y"] == 1)])
        med = np.median(cts) if cts else float("nan")
        print(f"\n===== rule = {rule.upper()} =====")
        print(f"median commit time (convergent runs, bistable cells): {med:.0f}")
        bist_low  = [(k, d) for k, d in cells.items() if k in low_cells  and 0.08 <= d['cyc'] <= 0.92]
        bist_high = [(k, d) for k, d in cells.items() if k in high_cells and 0.08 <= d['cyc'] <= 0.92]
        print(f"bistable cells: low-beta {len(bist_low)}/4, high-beta {len(bist_high)}/4")
        for f in (0.25, 0.40, 0.60):
            k = nearest_h(f * med) if med == med else None
            if k is None:
                continue
            def grp_auc(group):
                vals = []
                for (cell, d) in group:
                    und = d["commit_t"] > k
                    yU = d["y"][und]; xU = d["snaps"][k][und]
                    a = auc(yU, xU)
                    if a == a:
                        vals.append(a)
                return vals
            la = grp_auc(bist_low); ha = grp_auc(bist_high)
            ls = f"{np.mean(la):.3f}" if la else "  -  "
            hs = f"{np.mean(ha):.3f}" if ha else "  -  "
            print(f"  f={f:.2f}  horizon k={k:<3} "
                  f"| low-beta undecided AUC={ls}  | high-beta undecided AUC={hs}")

if __name__ == "__main__":
    main()
