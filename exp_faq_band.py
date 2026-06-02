"""
FAQ follow-up: locate FAQ's own bistable band, then test the structured (high-beta)
vs noise-dominated (low-beta) anticipation contrast WITHIN that band, at a horizon
matched to FAQ's (much later) commitment timescale.
"""
import numpy as np
from sklearn.metrics import roc_auc_score

GAMMA, ALPHA_FAQ, FAQ_CAP = 0.95, 0.1, 0.5
TAU0, DECAY, TAU_MIN = 1.0, 0.9998, 0.05
ENT_CYC, COMMIT_P, COMMIT_EVERY = 0.4, 0.9, 20
HORIZONS = [100, 200, 400, 700, 1000, 1500, 2000, 3000]

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

def run_faq(r, beta, S, T, seed, record=False):
    rng = np.random.default_rng(seed)
    A1, A2 = payoffs(r, beta)
    Q1 = rng.normal(0, 0.01, (S, 3)); Q2 = rng.normal(0, 0.01, (S, 3))
    idx = np.arange(S); window = T // 4
    c1 = np.zeros((S, 3)); c2 = np.zeros((S, 3))
    commit_t = np.full(S, T, float); committed = np.zeros(S, bool); snaps = {}
    for t in range(T):
        tau = max(TAU_MIN, TAU0 * DECAY ** t)
        pol1 = softmax_rows(Q1 / tau); pol2 = softmax_rows(Q2 / tau)
        a1 = np.argmax(Q1 / tau + rng.gumbel(size=(S, 3)), 1)
        a2 = np.argmax(Q2 / tau + rng.gumbel(size=(S, 3)), 1)
        td1 = A1[a1, a2] + GAMMA * Q1.max(1) - Q1[idx, a1]
        td2 = A2[a1, a2] + GAMMA * Q2.max(1) - Q2[idx, a2]
        e1 = np.minimum(ALPHA_FAQ / np.maximum(pol1[idx, a1], 1e-3), FAQ_CAP)
        e2 = np.minimum(ALPHA_FAQ / np.maximum(pol2[idx, a2], 1e-3), FAQ_CAP)
        Q1[idx, a1] += e1 * td1; Q2[idx, a2] += e2 * td2
        if record and (t + 1) in HORIZONS:
            snaps[t + 1] = 0.5 * (lead_gap(Q1) + lead_gap(Q2))
        if t % COMMIT_EVERY == 0:
            now = (pol1.max(1) > COMMIT_P) & (pol2.max(1) > COMMIT_P)
            newly = now & (~committed); commit_t[newly] = t; committed |= now
        if t >= T - window:
            c1[idx, a1] += 1; c2[idx, a2] += 1
    pe1 = c1 / c1.sum(1, keepdims=True); pe2 = c2 / c2.sum(1, keepdims=True)
    conv = (np.maximum(safe_entropy(pe1), safe_entropy(pe2)) <= ENT_CYC).astype(int)
    return dict(y=conv, commit_t=commit_t, snaps=snaps, cyc=float((1 - conv).mean()))

def auc(y, x):
    return roc_auc_score(y, x) if len(np.unique(y)) > 1 else np.nan

def main():
    print("FAQ phase scan (cycling fraction):")
    rs = [0.30, 0.45, 0.60, 0.75, 0.90, 1.05]; bs = [0.0, 0.15, 0.30, 0.45]
    grid = {}
    print(f"{'':>7}" + "".join(f" r={r:<5}" for r in rs))
    for b in bs:
        row = []
        for r in rs:
            d = run_faq(r, b, S=100, T=12000, seed=11)
            grid[(r, b)] = d["cyc"]; row.append(d["cyc"])
        print(f"b={b:<5}" + "".join(f" {v:6.3f}" for v in row))

    bist = [k for k, v in grid.items() if 0.08 <= v <= 0.92]
    print(f"\nBistable cells under FAQ: {sorted(bist)}")
    low  = [k for k in bist if k[1] == 0.0]
    high = [k for k in bist if k[1] >= 0.30]
    print(f"low-beta bistable: {sorted(low)}\nhigh-beta bistable: {sorted(high)}")

    # detailed runs with recording on a few low/high-beta bistable cells
    sel_low  = sorted(low)[:3]; sel_high = sorted(high)[:3]
    detail = {}
    for k in sel_low + sel_high:
        detail[k] = run_faq(k[0], k[1], S=250, T=16000, seed=29, record=True)
    cts = []
    for k in sel_low + sel_high:
        d = detail[k]; cts += list(d["commit_t"][d["y"] == 1])
    med = np.median(cts) if cts else float("nan")
    print(f"\nFAQ median commit (convergent, selected cells): {med:.0f}")
    for f in (0.25, 0.40, 0.60):
        k = min(HORIZONS, key=lambda h: abs(h - f * med))
        def grp(cells):
            out = []
            for c in cells:
                d = detail[c]; und = d["commit_t"] > k
                a = auc(d["y"][und], d["snaps"][k][und])
                if a == a: out.append(a)
            return out
        la, ha = grp(sel_low), grp(sel_high)
        ls = f"{np.mean(la):.3f}" if la else "  -  "
        hs = f"{np.mean(ha):.3f}" if ha else "  -  "
        print(f"  f={f:.2f} horizon k={k:<4} | low-beta AUC={ls} | high-beta AUC={hs}")

if __name__ == "__main__":
    main()
