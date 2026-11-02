"""Correctness checks of the model, the references and the methods.

    python verify.py            # all checks on small instances (about a minute)
    python verify.py --full     # also check every method on every slot of the paper's runs (about 15 minutes)

1. The hop-layered hindsight program equals an integer program over enumerated simple paths (48 small slots).
2. The expected-capacity LP is at least the mean hindsight optimum over many outcome draws (Jensen's inequality).
3. Every method and REPS variant respects channels and memory, uses only the links it obtained, routes simple paths
   within the hop bound, and is ordered: method <= its selection optimum <= hindsight optimum.
"""
import argparse
import random

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds

from qrcs import (REGIMES, METHODS, OTHER_POLICIES, make_instance, make_slot, realized, hop_limit, network_seed,
                  slot_seed, hindsight_optimum, lp_bound, post_generation_optimum, check_paths)

ALL_POLICIES = {**METHODS, **OTHER_POLICIES}


def hindsight_by_path_enumeration(inst, slot, H):
    """Hindsight optimum as an integer program with one variable per simple path (small instances only)."""
    cap = np.array([int(w.sum()) for w in slot.omega])
    q = inst.regime.q
    P = []
    for r, (s, d) in enumerate(slot.requests):
        for path in nx.all_simple_paths(inst.G, s, d, cutoff=H):
            es = [inst.eidx[(a, b)] for a, b in zip(path, path[1:])]
            if all(cap[e] > 0 for e in es):
                P.append((r, path, es))
    if not P:
        return 0.0
    c = -np.array([q ** (len(p) - 2) for _, p, _ in P])
    rows, cols, vals, ub = [], [], [], []
    row = 0
    for e in range(len(inst.edges)):                       # link capacity
        js = [j for j, (_, _, es) in enumerate(P) if e in es]
        if js:
            for j in js:
                rows.append(row); cols.append(j); vals.append(1.0)
            ub.append(cap[e]); row += 1
    for v in range(inst.n):                                # memory: one qubit per link end at v
        js = [(j, sum(1 for e in es if v in inst.edges[e])) for j, (_, _, es) in enumerate(P)]
        js = [(j, k) for j, k in js if k]
        if js:
            for j, k in js:
                rows.append(row); cols.append(j); vals.append(float(k))
            ub.append(inst.Q[v]); row += 1
    A = sp.csr_matrix((vals, (rows, cols)), shape=(row, len(P)))
    res = milp(c, constraints=LinearConstraint(A, -np.inf, ub), integrality=np.ones(len(P)), bounds=Bounds(0, np.inf))
    return -res.fun


def check_attempts(inst, a):
    assert (a >= 0).all() and (a <= inst.W).all(), 'channel limit violated'
    use = np.zeros(inst.n, dtype=int)
    for e, (u, v) in enumerate(inst.edges):
        use[u] += a[e]; use[v] += a[e]
    assert (use <= inst.Q).all(), 'memory limit violated'


def check_policies(inst, slot, t, ho=None):
    H = hop_limit(inst)
    for name, f in ALL_POLICIES.items():
        v, conns, links, a = f(inst, slot, random.Random(t))
        a = np.asarray(a)
        check_attempts(inst, a)
        assert (links == realized(inst, slot, a)).all(), 'links differ from the realized outcomes'
        check_paths(inst, slot, conns, links)
        if ho is not None:
            pg = post_generation_optimum(inst, slot, links, H)[0]
            assert v <= pg + 1e-6 <= ho + 2e-6, (name, v, pg, ho)
    return len(ALL_POLICIES)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--full', action='store_true', help='check every method on every slot of the paper runs')
    a = ap.parse_args()
    regimes = list(REGIMES.values())

    worst = 0.0
    for seed in range(12):
        inst = make_instance(7, regimes[seed % len(regimes)], seed)
        rng = random.Random(seed)
        for _ in range(4):
            slot = make_slot(inst, rng)
            H = hop_limit(inst)
            worst = max(worst, abs(hindsight_optimum(inst, slot, H)[0] - hindsight_by_path_enumeration(inst, slot, H)))
    assert worst < 1e-6
    print('1. layered program = path enumeration on 48 slots (largest difference %.1e)  OK' % worst)

    for name in ('generic', 'lossy-links', 'combined'):
        inst = make_instance(10, REGIMES[name], 5)
        rng = random.Random(3)
        slot0 = make_slot(inst, rng)
        H = hop_limit(inst)
        lp = lp_bound(inst, slot0, H)[0]
        hos = []
        for _ in range(150):
            s2 = make_slot(inst, rng)
            s2.requests = slot0.requests                   # same requests, new outcomes
            hos.append(hindsight_optimum(inst, s2, H)[0])
        m, se = np.mean(hos), np.std(hos) / np.sqrt(len(hos))
        assert lp >= m - 2 * se
        print('2. %-12s LP %.3f >= mean hindsight %.3f (s.e. %.3f)  OK' % (name, lp, m, se))

    n_checked = 0
    for seed in range(6):
        for reg in regimes:
            inst = make_instance(12, reg, 100 + seed)
            slot = make_slot(inst, random.Random(seed))
            ho = hindsight_optimum(inst, slot, hop_limit(inst))[0]
            n_checked += check_policies(inst, slot, seed, ho)
    print('3. %d (policy, slot) pairs on 12-node networks: feasible and method <= selection <= hindsight  OK'
          % n_checked)

    if a.full:
        from run_case_study import PAPER_CELLS
        n_checked = 0
        for k in range(10):
            for name, n in PAPER_CELLS:
                inst = make_instance(n, REGIMES[name], network_seed(n, k))
                for t in range(25):
                    slot = make_slot(inst, random.Random(slot_seed(n, k, t)))
                    n_checked += check_policies(inst, slot, t)
        print('4. %d (policy, slot) pairs of the paper runs: feasible  OK' % n_checked)


if __name__ == '__main__':
    main()
