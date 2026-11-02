"""Network model of the case study: regimes, random networks and slots.

One *network* is a random Waxman graph with channels per link, memory qubits per node and a success probability per
channel. One *slot* is a set of requests plus a pre-drawn outcome for every channel of every link. Every method and
every reference in a slot sees the same network, requests and potential outcomes (common random numbers).
"""
import math
from dataclasses import dataclass

import networkx as nx
import numpy as np


# --------------------------------------------------------------------------------------------------------------------
# Regimes
# --------------------------------------------------------------------------------------------------------------------
@dataclass
class Regime:
    name: str
    p_mean: float = 0.8          # mean per-slot success probability of one channel
    q: float = 0.9               # success probability of one entanglement swap
    W: tuple = (3, 5)            # channels per link, inclusive integer range
    Q: tuple = (10, 14)          # memory qubits per node, inclusive integer range
    H_max: int = None            # bound on links per connection (fidelity proxy); None = no bound
    R: int = 6                   # requests per slot


REGIMES = {
    'generic':      Regime('generic'),                                  # Q-CAST-like parameters
    'few-resources': Regime('few-resources', W=(1, 2), Q=(2, 4)),       # few memories and channels
    'lossy-links':  Regime('lossy-links', p_mean=0.3),
    'lossy-swaps':  Regime('lossy-swaps', q=0.5),
    'hop-bound':    Regime('hop-bound', H_max=3),                       # fidelity threshold as a 3-link bound
    'combined':     Regime('combined', p_mean=0.3, q=0.5, W=(1, 2), Q=(2, 4), H_max=3),   # all four changes
}


# --------------------------------------------------------------------------------------------------------------------
# Networks
# --------------------------------------------------------------------------------------------------------------------
@dataclass
class Instance:
    n: int                       # number of nodes
    G: nx.Graph                  # topology; G[u][v]['length'] is the link length
    edges: list                  # links as (u, v) with u < v; a link is referred to by its index e in this list
    eidx: dict                   # (u, v) and (v, u) -> link index e
    length: np.ndarray           # length of each link
    p: np.ndarray                # per-slot success probability of one channel on each link
    W: np.ndarray                # channels per link
    Q: np.ndarray                # memory qubits per node
    regime: Regime
    diam: int                    # diameter in links
    seed: int


def waxman(n, rng, alpha=0.4, target_deg=4.0, max_tries=500):
    """Connected Waxman graph on n nodes in the unit square.

    Nodes u, v are linked with probability beta * exp(-d_uv / (alpha * L)), L = largest distance between two nodes.
    beta is chosen so that the expected mean degree is target_deg, capped at 1. Disconnected draws are rejected."""
    for _ in range(max_tries):
        pos = rng.random((n, 2))
        D = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
        L = D.max()
        P = np.exp(-D / (alpha * L))
        iu = np.triu_indices(n, 1)
        beta = min(1.0, target_deg * n / 2 / P[iu].sum())
        A = (rng.random(len(iu[0])) < beta * P[iu])
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for k in np.nonzero(A)[0]:
            u, v = int(iu[0][k]), int(iu[1][k])
            G.add_edge(u, v, length=float(D[u, v]))
        if nx.is_connected(G):
            return G, pos
    raise RuntimeError('no connected Waxman graph')


def make_instance(n, regime, seed):
    """Random network of the given regime. Success probability p_e = exp(-length_e / ell), where ell is found by
    bisection so that the mean of p_e over the links equals regime.p_mean. W and Q are uniform in their ranges."""
    rng = np.random.default_rng(seed)
    G, pos = waxman(n, rng)
    edges = sorted((min(u, v), max(u, v)) for u, v in G.edges())
    eidx = {}
    for k, (u, v) in enumerate(edges):
        eidx[(u, v)] = k
        eidx[(v, u)] = k
    length = np.array([G[u][v]['length'] for u, v in edges])
    lo, hi = 1e-6, 1e6                       # bisection (geometric) for ell
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if np.exp(-length / mid).mean() < regime.p_mean:
            lo = mid
        else:
            hi = mid
    p = np.exp(-length / hi)
    W = rng.integers(regime.W[0], regime.W[1] + 1, size=len(edges))
    Q = rng.integers(regime.Q[0], regime.Q[1] + 1, size=n)
    return Instance(n, G, edges, eidx, length, p, W, Q, regime, nx.diameter(G), seed)


def network_seed(n, k):
    """Seed of network k (k = 0, 1, ...) with n nodes."""
    return 1000 * k + n


def slot_seed(n, k, t):
    """Seed of slot t of network k with n nodes."""
    return 10000 * k + 100 * n + t


# --------------------------------------------------------------------------------------------------------------------
# Slots
# --------------------------------------------------------------------------------------------------------------------
@dataclass
class Slot:
    requests: list               # list of (source, destination), distinct pairs in random order
    omega: list                  # omega[e][c] = 1 if channel c of link e would succeed in this slot, else 0


def make_slot(inst, rng):
    """Draw the requests and the potential outcome of every channel (rng is a random.Random)."""
    pairs = set()
    while len(pairs) < inst.regime.R:
        s, d = rng.sample(range(inst.n), 2)
        pairs.add((s, d))
    req = sorted(pairs)
    rng.shuffle(req)
    omega = [np.array([1 if rng.random() < inst.p[e] else 0 for _ in range(inst.W[e])]) for e in range(len(inst.edges))]
    return Slot(req, omega)


def realized(inst, slot, attempts):
    """Links obtained when attempts[e] channels are attempted on link e: the first attempts[e] outcomes are used."""
    return np.array([int(slot.omega[e][:attempts[e]].sum()) for e in range(len(inst.edges))])


def hop_limit(inst):
    """Largest number of links per connection: the regime's bound, or n - 1 (every simple path)."""
    return inst.n - 1 if inst.regime.H_max is None else inst.regime.H_max
