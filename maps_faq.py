"""FAQ anticipation field on the shared (r,beta) grid, matched horizon k=1500."""
import numpy as np
from sklearn.metrics import roc_auc_score

GAMMA, ALPHA_FAQ, FAQ_CAP = 0.95, 0.1, 0.5
TAU0, DECAY, TAU_MIN = 1.0, 0.9998, 0.05
ENT_CYC, COMMIT_P, COMMIT_EVERY = 0.4, 0.9, 25

def cyc_matrix():
    C = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            d = (j - i) % 3
            C[i, j] = 1.0 if d == 1 else (-1.0 if d == 2 else 0.0)
    return C
CYC = cyc_matrix()
def payoffs(r, b):
    base = r*(1-np.eye(3)); return base + b*CYC, base + b*CYC.T
def sm(Z):
    Z = Z - Z.max(1, keepdims=True); E = np.exp(Z); return E/E.sum(1, keepdims=True)
def ent(p):
    c = np.zeros_like(p); m = p>0; c[m] = p[m]*np.log2(p[m]); return -c.sum(1)
def gap(M):
    s = np.sort(M,1); return s[:,-1]-s[:,-2]

def cell(r, b, S, T, k, seed):
    rng = np.random.default_rng(seed); A1, A2 = payoffs(r, b)
    Q1 = rng.normal(0,0.01,(S,3)); Q2 = rng.normal(0,0.01,(S,3))
    idx = np.arange(S); win = T//4
    c1 = np.zeros((S,3)); c2 = np.zeros((S,3))
    ct = np.full(S, T, float); done = np.zeros(S, bool); snap = None
    for t in range(T):
        tau = max(TAU_MIN, TAU0*DECAY**t)
        p1 = sm(Q1/tau); p2 = sm(Q2/tau)
        a1 = np.argmax(Q1/tau + rng.gumbel(size=(S,3)),1)
        a2 = np.argmax(Q2/tau + rng.gumbel(size=(S,3)),1)
        td1 = A1[a1,a2]+GAMMA*Q1.max(1)-Q1[idx,a1]
        td2 = A2[a1,a2]+GAMMA*Q2.max(1)-Q2[idx,a2]
        e1 = np.minimum(ALPHA_FAQ/np.maximum(p1[idx,a1],1e-3), FAQ_CAP)
        e2 = np.minimum(ALPHA_FAQ/np.maximum(p2[idx,a2],1e-3), FAQ_CAP)
        Q1[idx,a1]+=e1*td1; Q2[idx,a2]+=e2*td2
        if t+1 == k: snap = 0.5*(gap(Q1)+gap(Q2))
        if t % COMMIT_EVERY == 0:
            now = (p1.max(1)>COMMIT_P)&(p2.max(1)>COMMIT_P)
            nw = now & (~done); ct[nw]=t; done|=now
        if t >= T-win:
            c1[idx,a1]+=1; c2[idx,a2]+=1
    pe1 = c1/c1.sum(1,keepdims=True); pe2 = c2/c2.sum(1,keepdims=True)
    conv = (np.maximum(ent(pe1),ent(pe2))<=ENT_CYC).astype(int)
    cyc = float((1-conv).mean()); gA = np.nan
    if 0.08 <= cyc <= 0.92:
        und = ct>k; yU = conv[und]
        if und.sum()>=20 and len(np.unique(yU))>1:
            gA = roc_auc_score(yU, snap[und])
    return cyc, gA

if __name__ == "__main__":
    rs = np.round(np.linspace(0.65,1.40,8),3)
    bs = np.round(np.linspace(0.00,0.60,6),3)
    S, T, k = 120, 13000, 1500
    CY = np.zeros((len(bs),len(rs))); GA = np.full((len(bs),len(rs)), np.nan)
    for i,b in enumerate(bs):
        for j,r in enumerate(rs):
            CY[i,j], GA[i,j] = cell(r,b,S,T,k,seed=321)
        print(f"[faq] beta={b:.2f} done")
    np.savez("map_faq.npz", CY=CY, GA=GA, rs=rs, bs=bs, k=k)
    print("saved map_faq.npz")
