"""
Mechanism, corrected. The predictor is early gap MAGNITUDE -> convergence vs
cycling (not which action). So the right mechanistic quantity is how much the
early gap SEPARATES eventual convergers from eventual cyclers, and whether that
separation is produced by bootstrap amplification (accelerating gap growth).

Per config we report, over runs split by eventual outcome:
  - mean early gap for convergers vs cyclers, and Cohen's d (the driver of AUC)
  - gap growth ratio gap(k)/gap(k/2) for convergers vs cyclers
    (value bootstrapping should make convergers' gap ACCELERATE: ratio >> cyclers')
"""
import numpy as np

GAMMA, ALPHA_Q, ALPHA_FAQ, FAQ_CAP = 0.95, 0.1, 0.1, 0.5
TAU0, DECAY, TAU_MIN = 1.0, 0.9998, 0.05
LR_PG, BW_PG = 0.05, 0.01
ENT_CYC = 0.4

def cyc_matrix():
    C = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            d = (j-i) % 3
            C[i, j] = 1.0 if d == 1 else (-1.0 if d == 2 else 0.0)
    return C
CYC = cyc_matrix()
def payoffs(r, b):
    base = r*(1-np.eye(3)); return base + b*CYC, base + b*CYC.T
def sm(Z):
    Z = Z - Z.max(1, keepdims=True); E = np.exp(Z); return E/E.sum(1, keepdims=True)
def ent_rows(p):
    c = np.zeros_like(p); m = p>0; c[m]=p[m]*np.log2(p[m]); return -c.sum(1)
def gap(M):
    s = np.sort(M,1); return s[:,-1]-s[:,-2]

def run(rule, r, b, S, T, k, seed):
    rng = np.random.default_rng(seed); A1, A2 = payoffs(r, b)
    P1 = rng.normal(0,0.01,(S,3)); P2 = rng.normal(0,0.01,(S,3))
    bl1=np.zeros(S); bl2=np.zeros(S); idx=np.arange(S); win=T//4
    c1=np.zeros((S,3)); c2=np.zeros((S,3)); ghalf=None; gfull=None
    for t in range(T):
        if rule in ("q","faq"):
            tau = max(TAU_MIN, TAU0*DECAY**t)
            p1=sm(P1/tau); p2=sm(P2/tau)
            a1=np.argmax(P1/tau+rng.gumbel(size=(S,3)),1)
            a2=np.argmax(P2/tau+rng.gumbel(size=(S,3)),1)
            td1=A1[a1,a2]+GAMMA*P1.max(1)-P1[idx,a1]; td2=A2[a1,a2]+GAMMA*P2.max(1)-P2[idx,a2]
            if rule=="q":
                P1[idx,a1]+=ALPHA_Q*td1; P2[idx,a2]+=ALPHA_Q*td2
            else:
                e1=np.minimum(ALPHA_FAQ/np.maximum(p1[idx,a1],1e-3),FAQ_CAP)
                e2=np.minimum(ALPHA_FAQ/np.maximum(p2[idx,a2],1e-3),FAQ_CAP)
                P1[idx,a1]+=e1*td1; P2[idx,a2]+=e2*td2
        else:
            p1=sm(P1); p2=sm(P2)
            a1=(np.cumsum(p1,1)>rng.random((S,1))).argmax(1)
            a2=(np.cumsum(p2,1)>rng.random((S,1))).argmax(1)
            r1=A1[a1,a2]; r2=A2[a1,a2]
            o1=np.zeros((S,3)); o1[idx,a1]=1; o2=np.zeros((S,3)); o2[idx,a2]=1
            P1+=LR_PG*(r1-bl1)[:,None]*(o1-p1); P2+=LR_PG*(r2-bl2)[:,None]*(o2-p2)
            bl1+=BW_PG*(r1-bl1); bl2+=BW_PG*(r2-bl2)
        if t+1==k//2: ghalf=0.5*(gap(P1)+gap(P2))
        if t+1==k:   gfull=0.5*(gap(P1)+gap(P2))
        if t>=T-win: c1[idx,a1]+=1; c2[idx,a2]+=1
    pe1=c1/c1.sum(1,keepdims=True); pe2=c2/c2.sum(1,keepdims=True)
    conv = np.maximum(ent_rows(pe1),ent_rows(pe2))<=ENT_CYC
    cyc_runs = ~conv
    def cohend(x, y):
        nx,ny=len(x),len(y)
        if nx<2 or ny<2: return np.nan
        sp=np.sqrt(((nx-1)*x.var(ddof=1)+(ny-1)*y.var(ddof=1))/(nx+ny-2))
        return (x.mean()-y.mean())/sp if sp>0 else np.nan
    gconv, gcyc = gfull[conv], gfull[cyc_runs]
    d = cohend(gconv, gcyc)
    # growth ratio gap(k)/gap(k/2), guard small denom
    gr_conv = float(np.mean(gfull[conv]/np.maximum(ghalf[conv],1e-3))) if conv.any() else np.nan
    gr_cyc  = float(np.mean(gfull[cyc_runs]/np.maximum(ghalf[cyc_runs],1e-3))) if cyc_runs.any() else np.nan
    return dict(cyc=float(cyc_runs.mean()),
                gap_conv=float(gconv.mean()) if conv.any() else np.nan,
                gap_cyc=float(gcyc.mean()) if cyc_runs.any() else np.nan,
                cohend=d, growth_conv=gr_conv, growth_cyc=gr_cyc)

def main():
    S=400
    cfg = [("Q  structured (b=0.4)","q",1.0,0.4,6000,100),
           ("Q  noise-dom  (b=0.0)","q",1.0,0.0,6000,100),
           ("FAQ structured(b=0.4)","faq",1.0,0.4,14000,1500),
           ("PG structured (b=0.4)","pg",1.0,0.4,9000,300),
           ("PG noise-dom  (b=0.0)","pg",1.0,0.0,9000,300)]
    print(f"{'config':>22} {'cyc':>5} {'gap|conv':>9} {'gap|cyc':>8} "
          f"{'Cohen d':>8} {'grow|conv':>10} {'grow|cyc':>9}")
    for name,rule,r,b,T,k in cfg:
        d=run(rule,r,b,S,T,k,seed=303)
        print(f"{name:>22} {d['cyc']:5.2f} {d['gap_conv']:9.3f} {d['gap_cyc']:8.3f} "
              f"{d['cohend']:8.3f} {d['growth_conv']:10.2f} {d['growth_cyc']:9.2f}")
    print("\nCohen d = standardized early-gap separation between convergers and cyclers")
    print("          (this is what the AUC measures; larger => predictor works)")
    print("grow = mean gap(k)/gap(k/2): acceleration of separation between k/2 and k")

if __name__ == "__main__":
    main()
