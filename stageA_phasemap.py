"""
Stage A — Phase map of the 2-player, 3-action anti-coordination family.

Family (payoff space, two axes):
  r    = reward for differing (anti-coordination strength)
  beta = cyclic (rock-paper-scissors) asymmetry strength

Player 1 payoff:  A1[i,j] = r * (i != j) + beta * cyc(i,j)
Player 2 payoff:  A2[i,j] = r * (i != j) + beta * cyc(j,i)
  cyc(i,j) = +1 if (j-i) mod 3 == 1, -1 if (j-i) mod 3 == 2, 0 if i==j

Learning: independent stateless Q-learning, Boltzmann action selection.
  Q[a] <- Q[a] + alpha*(reward + gamma*max_b Q[b] - Q[a])   (chosen action only)
  temperature tau_t = max(tau_min, tau0 * decay**t)
Vectorised across S seeds per grid point (Gumbel-max sampling).

Outcome label per run: from marginal action-frequency entropy over the final
window. Low entropy (one action dominant) => CONVERGED; high entropy => CYCLING.
Goal of Stage A: find which (r,beta) cells are ~0% cycling, ~100% cycling, and
which are intermediate (BISTABLE).
"""
import numpy as np

ALPHA, GAMMA = 0.1, 0.95
TAU0, DECAY, TAU_MIN = 1.0, 0.9998, 0.05

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
    A1 = base + beta * CYC
    A2 = base + beta * CYC.T
    return A1, A2

def simulate(r, beta, S, T, seed, window=None):
    """Vectorised over S seeds. Returns cycling-fraction and per-seed entropy."""
    if window is None:
        window = T // 4
    rng = np.random.default_rng(seed)
    A1, A2 = payoffs(r, beta)
    Q1 = rng.normal(0, 0.01, (S, 3))
    Q2 = rng.normal(0, 0.01, (S, 3))
    idx = np.arange(S)
    counts1 = np.zeros((S, 3))
    counts2 = np.zeros((S, 3))
    for t in range(T):
        tau = max(TAU_MIN, TAU0 * DECAY ** t)
        g1 = rng.gumbel(size=(S, 3)); g2 = rng.gumbel(size=(S, 3))
        a1 = np.argmax(Q1 / tau + g1, axis=1)
        a2 = np.argmax(Q2 / tau + g2, axis=1)
        r1 = A1[a1, a2]; r2 = A2[a1, a2]
        Q1[idx, a1] += ALPHA * (r1 + GAMMA * Q1.max(1) - Q1[idx, a1])
        Q2[idx, a2] += ALPHA * (r2 + GAMMA * Q2.max(1) - Q2[idx, a2])
        if t >= T - window:
            counts1[idx, a1] += 1; counts2[idx, a2] += 1
    p1 = counts1 / counts1.sum(1, keepdims=True)
    p2 = counts2 / counts2.sum(1, keepdims=True)
    def ent(p):
        return -np.sum(np.where(p > 0, p * np.log2(p), 0.0), axis=1)
    e = np.maximum(ent(p1), ent(p2))          # max entropy across the two agents
    cycling = e > 0.4                          # >0.4 bits => not concentrated
    return cycling.mean(), e

if __name__ == "__main__":
    rs = [0.5, 1.0, 1.5]
    betas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]
    print(f"{'r':>5} {'beta':>5} {'cyc_frac':>9}   regime")
    for r in rs:
        for b in betas:
            frac, e = simulate(r, b, S=60, T=3000, seed=12345)
            regime = ("converge" if frac < 0.05 else
                      "cycle" if frac > 0.95 else "BISTABLE")
            print(f"{r:5.2f} {b:5.2f} {frac:9.3f}   {regime}")
