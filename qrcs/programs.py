"""Optimization references of the case study, all built on one hop-layered integer/linear program.

Variables x[r, k, u->v] >= 0 count the connections of request r whose k-th link goes from u to v (k = 1..H).
Objective: sum over requests and k of q**(k-1) * (connections with k links), i.e. expected end-to-end pairs when
each of the k-1 swaps succeeds with probability q. Constraints: flow conservation between layers, link capacity
cap[e], and optionally node memory (each use of a link holds one qubit at each end).

  hindsight_optimum       cap = channels that WOULD succeed, memory constraint, integer   (knows outcomes in advance)
  lp_bound                cap = W_e * p_e (expected), memory constraint, continuous       (expected-capacity LP)
  post_generation_optimum cap = links a method obtained, no memory constraint, integer    (best selection after
                                                                                           that method's attempts)
Solved with HiGHS through scipy.optimize.milp / linprog.
"""
import time

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, linprog, LinearConstraint, Bounds

def routing_program(inst, requests, cap, H, mem=None, integral=True, time_limit=60.0):
    """max sum_r sum_k q**(k-1) (connections of request r with k links)
    s.t. link capacity cap[e] (sum over requests, layers and directions), memory mem[v] (each link used occupies
    one qubit at each end; None = no memory constraint). Variables x[r,k,u->v] >= 0, integral if integral=True.
    Returns (value, paths, info) with paths = list of (r, [nodes]) when integral (decomposed), else None."""
    q = inst.regime.q
    n = inst.n
    adj = {v: [] for v in range(n)}
    for (u, v) in inst.edges:
        e = inst.eidx[(u, v)]
        if cap[e] > 1e-12:
            adj[u].append(v)
            adj[v].append(u)
    var = []                     # (r, k, u, v)
    vid = {}
    for r, (s, d) in enumerate(requests):
        ds = nx.single_source_shortest_path_length(nx.Graph([(u, w) for u in adj for w in adj[u]]), s) if adj[s] else {s: 0}
        dd = nx.single_source_shortest_path_length(nx.Graph([(u, w) for u in adj for w in adj[u]]), d) if adj[d] else {d: 0}
        if d not in ds:
            continue
        for k in range(1, H + 1):
            for u in range(n):
                if u == d or ds.get(u, 10 ** 9) > k - 1:
                    continue
                if k == 1 and u != s:
                    continue
                if k > 1 and u == s:
                    continue
                for v in adj[u]:
                    if v == s or dd.get(v, 10 ** 9) > H - k:
                        continue
                    vid[(r, k, u, v)] = len(var)
                    var.append((r, k, u, v))
    nv = len(var)
    if nv == 0:
        return 0.0, [], {'nvar': 0, 'time': 0.0, 'status': 'empty', 'optimal': True, 'bound': 0.0, 'max_hops_used': 0}
    c = np.zeros(nv)
    for j, (r, k, u, v) in enumerate(var):
        if v == requests[r][1]:
            c[j] = -(q ** (k - 1))
    rows, cols, vals = [], [], []
    lb, ub = [], []
    row = 0
    # conservation: inflow to v at layer k = outflow from v at layer k+1 (v not s, not d)
    inflow = {}
    outflow = {}
    for j, (r, k, u, v) in enumerate(var):
        inflow.setdefault((r, k, v), []).append(j)
        outflow.setdefault((r, k, u), []).append(j)
    for (r, k, v), js in inflow.items():
        s, d = requests[r]
        if v == d:
            continue
        for j in js:
            rows.append(row); cols.append(j); vals.append(1.0)
        for j in outflow.get((r, k + 1, v), []):
            rows.append(row); cols.append(j); vals.append(-1.0)
        lb.append(0.0); ub.append(0.0); row += 1
    for (r, k, u), js in outflow.items():
        s, d = requests[r]
        if u == s or k == 1:
            continue
        if (r, k - 1, u) not in inflow:
            for j in js:
                rows.append(row); cols.append(j); vals.append(1.0)
            lb.append(0.0); ub.append(0.0); row += 1
    # link capacity
    by_edge = {}
    for j, (r, k, u, v) in enumerate(var):
        by_edge.setdefault(inst.eidx[(u, v)], []).append(j)
    for e, js in by_edge.items():
        for j in js:
            rows.append(row); cols.append(j); vals.append(1.0)
        lb.append(-np.inf); ub.append(float(cap[e])); row += 1
    if mem is not None:
        by_node = {}
        for e, js in by_edge.items():
            u, v = inst.edges[e]
            by_node.setdefault(u, []).extend(js)
            by_node.setdefault(v, []).extend(js)
        for v, js in by_node.items():
            for j in js:
                rows.append(row); cols.append(j); vals.append(1.0)
            lb.append(-np.inf); ub.append(float(mem[v])); row += 1
    A = sp.csr_matrix((vals, (rows, cols)), shape=(row, nv))
    upper = np.array([cap[inst.eidx[(u, v)]] for (r, k, u, v) in var], dtype=float)
    t0 = time.perf_counter()
    if integral:
        res = milp(c, constraints=LinearConstraint(A, lb, ub), integrality=np.ones(nv), bounds=Bounds(0, upper),
                   options={'time_limit': time_limit, 'mip_rel_gap': 1e-6})
    else:
        lb_a, ub_a = np.array(lb), np.array(ub)
        eq = np.nonzero(lb_a == ub_a)[0]
        ine = np.nonzero(lb_a != ub_a)[0]
        res = linprog(c, A_ub=A[ine] if len(ine) else None, b_ub=ub_a[ine] if len(ine) else None,
                      A_eq=A[eq] if len(eq) else None, b_eq=ub_a[eq] if len(eq) else None,
                      bounds=list(zip(np.zeros(nv), upper)), method='highs', options={'time_limit': time_limit})
    dt = time.perf_counter() - t0
    info = {'nvar': nv, 'ncon': row, 'time': dt, 'status': res.status, 'msg': res.message}
    if res.x is None:
        raise RuntimeError('routing program failed: %s' % res.message)
    val = -float(res.fun)
    info['optimal'] = (res.status == 0)
    info['bound'] = val
    if integral:
        info['mip_gap'] = getattr(res, 'mip_gap', None)
        db = getattr(res, 'mip_dual_bound', None)
        if res.status != 0 and db is not None:
            info['bound'] = -float(db)          # valid upper bound when the time limit is hit
    paths = None
    if integral:
        x = np.round(res.x).astype(int)
        paths = decompose_layered(var, x, requests)
        info['max_hops_used'] = max((len(p) - 1 for _, p in paths), default=0)
    return val, paths, info


def decompose_layered(var, x, requests):
    """Turn layered integral flows into connections (r, [nodes])."""
    flow = {}
    for j, (r, k, u, v) in enumerate(var):
        if x[j] > 0:
            flow[(r, k, u, v)] = flow.get((r, k, u, v), 0) + int(x[j])
    paths = []
    for r, (s, d) in enumerate(requests):
        while True:
            start = [key for key in flow if key[0] == r and key[1] == 1 and key[2] == s and flow[key] > 0]
            if not start:
                break
            key = start[0]
            path = [s, key[3]]
            flow[key] -= 1
            k = 1
            while path[-1] != d:
                k += 1
                nxt = [kk for kk in flow if kk[0] == r and kk[1] == k and kk[2] == path[-1] and flow[kk] > 0]
                if not nxt:
                    raise RuntimeError('decomposition failed')
                flow[nxt[0]] -= 1
                path.append(nxt[0][3])
            paths.append((r, path))
    return paths


def hindsight_optimum(inst, slot, H, time_limit=120.0):
    """Best integral routing when the outcomes are known before choosing the attempts: max over attempts a of
    S(a, omega). Every method's connections are feasible for it, so it is an upper reference in every slot."""
    cap = np.array([int(w.sum()) for w in slot.omega])
    return routing_program(inst, slot.requests, cap, H, mem=inst.Q, integral=True, time_limit=time_limit)


def lp_bound(inst, slot, H):
    """Expected-capacity linear program: continuous variables, capacity W_e * p_e. By Jensen's inequality its value
    is at least the expected hindsight optimum (not a per-slot bound)."""
    cap = inst.W * inst.p
    return routing_program(inst, slot.requests, cap, H, mem=inst.Q, integral=False)


def post_generation_optimum(inst, slot, links, H):
    """Selection program S(a, omega): best integral routing over the links a method obtained. No memory constraint:
    the memories were committed when the channels were attempted. method <= this <= hindsight optimum."""
    return routing_program(inst, slot.requests, links, H, mem=None, integral=True)


def hindsight_relaxation(inst, slot, H):
    """Linear relaxation of the hindsight program (realized capacities, continuous variables). Splits the gap
    LP / HO into an expectation part (LP / this) and an integrality part (this / HO)."""
    cap = np.array([int(w.sum()) for w in slot.omega])
    return routing_program(inst, slot.requests, cap, H, mem=inst.Q, integral=False)


def value_of(inst, paths):
    """Expected end-to-end pairs of a set of connections (r, [nodes]): sum of q**(links - 1)."""
    q = inst.regime.q
    return sum(q ** (len(p) - 2) for _, p in paths)


def check_paths(inst, slot, paths, links):
    """Every connection is a simple s-d path within the hop bound, and link use stays within links."""
    use = np.zeros(len(inst.edges), dtype=int)
    for r, p in paths:
        s, d = slot.requests[r]
        assert p[0] == s and p[-1] == d, 'endpoints'
        assert len(set(p)) == len(p), 'not simple'
        if inst.regime.H_max is not None:
            assert len(p) - 1 <= inst.regime.H_max, 'hop bound'
        for a, b in zip(p, p[1:]):
            use[inst.eidx[(a, b)]] += 1
    assert (use <= links).all(), 'link capacity'
    return True
