"""Anticipation fields for Q-learning and policy gradient on a shared (r,beta) grid.
Each measured at a rule-matched early horizon (Q: k=100, PG: k=300).
Saves cycling-fraction and genuine-AUC arrays to .npy for assembly."""
import numpy as np
from sklearn.metrics import roc_auc_score

GAMMA, ALPHA_Q = 0.95, 0.1
TAU0, DECAY, TAU_MIN = 1.0, 0.9998, 0.05
LR_PG, BW_PG = 0.05, 0.01
ENT_CYC, COMMIT_P, COMMIT_EVERY = 0.4, 0.9, 20

def cyc_matrix():
    C = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            d = (j - i) % 3
            C[i, j] = 1.0 if d == 1 else (-1.0 if d == 2 else 0.0)
    return C
CYC = cyc_matrix()
def payoffs(r, b):
    base = r * (1 - np.eye(3)); return base + b * CYC, base + b * CYC.T
def sm(Z):
    Z = Z - Z.max(1, keepdims=True); E = np.exp(Z); return E / E.sum(1, keepdims=True)
def ent(p):
    c = np.zeros_like(p); m = p > 0; c[m] = p[m]*np.log2(p[m]); return -c.sum(1)
def gap(M):
    s = np.sort(M, 1); return s[:, -1] - s[:, -2]

def cell(rule, r, b, S, T, k, seed):
    rng = np.random.default_rng(seed); A1, A2 = payoffs(r, b)
    P1 = rng.normal(0, 0.01, (S, 3)); P2 = rng.normal(0, 0.01, (S, 3))
    bl1 = np.zeros(S); bl2 = np.zeros(S); idx = np.arange(S); win = T // 4
    c1 = np.zeros((S, 3)); c2 = np.zeros((S, 3))
    ct = np.full(S, T, float); done = np.zeros(S, bool); snap = None
    for t in range(T):
        if rule == "q":
            tau = max(TAU_MIN, TAU0 * DECAY ** t)
            p1 = sm(P1/tau); p2 = sm(P2/tau)
            a1 = np.argmax(P1/tau + rng.gumbel(size=(S,3)), 1)
            a2 = np.argmax(P2/tau + rng.gumbel(size=(S,3)), 1)
            P1[idx,a1] += ALPHA_Q*(A1[a1,a2]+GAMMA*P1.max(1)-P1[idx,a1])
            P2[idx,a2] += ALPHA_Q*(A2[a1,a2]+GAMMA*P2.max(1)-P2[idx,a2])
        else:
            p1 = sm(P1); p2 = sm(P2)
            a1 = (np.cumsum(p1,1) > rng.random((S,1))).argmax(1)
            a2 = (np.cumsum(p2,1) > rng.random((S,1))).argmax(1)
            r1 = A1[a1,a2]; r2 = A2[a1,a2]
            o1 = np.zeros((S,3)); o1[idx,a1]=1; o2 = np.zeros((S,3)); o2[idx,a2]=1
            P1 += LR_PG*(r1-bl1)[:,None]*(o1-p1); P2 += LR_PG*(r2-bl2)[:,None]*(o2-p2)
            bl1 += BW_PG*(r1-bl1); bl2 += BW_PG*(r2-bl2)
        if t+1 == k:
            snap = 0.5*(gap(P1)+gap(P2))
        if t % COMMIT_EVERY == 0:
            now = (p1.max(1) > COMMIT_P) & (p2.max(1) > COMMIT_P)
            nw = now & (~done); ct[nw] = t; done |= now
        if t >= T-win:
            c1[idx,a1]+=1; c2[idx,a2]+=1
    pe1 = c1/c1.sum(1,keepdims=True); pe2 = c2/c2.sum(1,keepdims=True)
    conv = (np.maximum(ent(pe1), ent(pe2)) <= ENT_CYC).astype(int)
    cyc = float((1-conv).mean()); gA = np.nan
    if 0.08 <= cyc <= 0.92:
        und = ct > k; yU = conv[und]
        if und.sum() >= 20 and len(np.unique(yU)) > 1:
            gA = roc_auc_score(yU, snap[und])
    return cyc, gA

def run(rule, k, T, S):
    rs = np.round(np.linspace(0.65, 1.40, 8), 3)
    bs = np.round(np.linspace(0.00, 0.60, 6), 3)
    CY = np.zeros((len(bs), len(rs))); GA = np.full((len(bs), len(rs)), np.nan)
    for i, b in enumerate(bs):
        for j, r in enumerate(rs):
            CY[i,j], GA[i,j] = cell(rule, r, b, S, T, k, seed=321)
        print(f"[{rule}] beta={b:.2f} done")
    np.savez(f"map_{rule}.npz", CY=CY, GA=GA, rs=rs, bs=bs, k=k)
    print(f"saved map_{rule}.npz")

if __name__ == "__main__":
    run("q", k=100, T=5000, S=160)
    run("pg", k=300, T=8000, S=160)
