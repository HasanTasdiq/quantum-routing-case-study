"""Routing methods of the case study, reimplemented from the published descriptions (not checked against the
authors' code). Every method takes (inst, slot, rng) and returns (value, connections, links, attempts):

  value        expected end-to-end pairs of its connections (sum of q**(links-1))
  connections  list of (request index, [nodes])
  links        links obtained on each link e (from the first attempts[e] channel outcomes)
  attempts     channels attempted on each link e (the provisioning decision, made before outcomes are known)

  greedy_sp  fewest-hop greedy: width-1 fewest-hop paths reserved for the requests in turn until resources run out;
             a reserved path delivers only if all its links succeed. (A simple proactive baseline; not the greedy
             baseline used in the Q-CAST evaluation.)
  qcast      Q-CAST [Shi and Qian 2020], transcribed from the authors' simulator; see qrcs/qcast.py.
  reps       REPS [Zhao and Qiao 2021]: provisioning by LP relaxation and progressive rounding (PFT), then selection
             by LP and randomized rounding (EPS) and greedy link selection (ELS). Rounding of f/p is set by
             `reserve` and `attempt`: ceil/ceil (main), ceil/floor (as the paper writes it), floor/floor.
  reactive   reactive greedy, loosely based on the global-link-state protocol of Pant et al. 2019: memory is spread
             over the links (one more attempted channel per link per round, links in random order) while channels
             and memory allow; after generation, fewest-hop paths over the links obtained, requests in turn. Pant et
             al. attempt every channel with one memory per channel, and share several flows in time or space.
  reps_x     provisioning from one solution of the REPS relaxation (rounded down, then up by fractional part while
             resources allow) followed by REPS's selection. Its attempts are also those of the LP+IP policy.
"""
import math
from collections import deque

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from .model import realized
from .programs import value_of
from .qcast import qcast


def bfs_path(inst, s, d, edge_ok, node_ok, hmax, avoid=()):
    """Fewest-hop s-d path over edges e with edge_ok(e), intermediate nodes v with node_ok(v), at most hmax links."""
    if s == d:
        return None
    prev = {s: None}
    depth = {s: 0}
    dq = deque([s])
    while dq:
        v = dq.popleft()
        if hmax is not None and depth[v] >= hmax:
            continue
        for u in inst.G.neighbors(v):
            if u in prev or u in avoid:
                continue
            if not edge_ok(inst.eidx[(v, u)]):
                continue
            if u != d and not node_ok(u):
                continue
            prev[u] = v
            depth[u] = depth[v] + 1
            if u == d:
                path = [d]
                while prev[path[-1]] is not None:
                    path.append(prev[path[-1]])
                return path[::-1]
            dq.append(u)
    return None


def edges_of(inst, path):
    return [inst.eidx[(a, b)] for a, b in zip(path, path[1:])]


# ----------------------------------------------------------------------------------------------------------------
def greedy_sp(inst, slot, rng):
    chan = inst.W.copy()
    mem = inst.Q.copy()
    hmax = inst.regime.H_max
    reserved = []                           # (r, path, channel index per hop)
    active = list(range(len(slot.requests)))
    while active:
        progressed = False
        order = active[:]
        rng.shuffle(order)
        for r in order:
            s, d = slot.requests[r]
            if mem[s] < 1 or mem[d] < 1:
                active.remove(r)
                continue
            path = bfs_path(inst, s, d, lambda e: chan[e] >= 1, lambda v: mem[v] >= 2, hmax)
            if path is None:
                active.remove(r)
                continue
            idx = []
            for e in edges_of(inst, path):
                idx.append(inst.W[e] - chan[e])
                chan[e] -= 1
            mem[s] -= 1
            mem[d] -= 1
            for v in path[1:-1]:
                mem[v] -= 2
            reserved.append((r, path, idx))
            progressed = True
        if not progressed:
            break
    attempts = inst.W - chan
    links = realized(inst, slot, attempts)
    conns = []
    for r, path, idx in reserved:
        if all(slot.omega[e][c] == 1 for e, c in zip(edges_of(inst, path), idx)):
            conns.append((r, path))
    return value_of(inst, conns), conns, links, attempts


# ----------------------------------------------------------------------------------------------------------------
def flow_lp(inst, requests, link_cap=None, rem_c=None, rem_m=None, t_cap=None, p_eff=None):
    """REPS LPs. PFT (link_cap None): max sum t_r with sum_r f_r(u,v)+f_r(v,u) <= p_e x_e, x_e <= rem_c[e],
    sum_{e at v} x_e <= rem_m[v]. EPS (link_cap given): sum_r f_r(u,v)+f_r(v,u) <= link_cap[e], t_r <= t_cap[r].
    Returns per-request directed flows {(u,v): f} and t."""
    R = len(requests)
    E = len(inst.edges)
    n = inst.n
    arcs = [(u, v) for (u, v) in inst.edges] + [(v, u) for (u, v) in inst.edges]
    A = len(arcs)
    nf = R * A
    nx_ = E if link_cap is None else 0
    nvar = nf + R + nx_
    fi = lambda r, a: r * A + a
    ti = lambda r: nf + r
    xi = lambda e: nf + R + e
    c = np.zeros(nvar)
    for r in range(R):
        c[ti(r)] = -1.0
    rows, cols, vals, beq = [], [], [], []
    row = 0
    for r, (s, d) in enumerate(requests):
        for v in range(n):
            for a, (x, y) in enumerate(arcs):
                if x == v:
                    rows.append(row); cols.append(fi(r, a)); vals.append(1.0)
                if y == v:
                    rows.append(row); cols.append(fi(r, a)); vals.append(-1.0)
            if v == s:
                rows.append(row); cols.append(ti(r)); vals.append(-1.0)
            if v == d:
                rows.append(row); cols.append(ti(r)); vals.append(1.0)
            beq.append(0.0)
            row += 1
    Aeq = sp.csr_matrix((vals, (rows, cols)), shape=(row, nvar))
    rows, cols, vals, bub = [], [], [], []
    row = 0
    for e in range(E):
        for r in range(R):
            rows.append(row); cols.append(fi(r, e)); vals.append(1.0)
            rows.append(row); cols.append(fi(r, e + E)); vals.append(1.0)
        if link_cap is None:
            rows.append(row); cols.append(xi(e)); vals.append(-float(p_eff[e]))
            bub.append(0.0)
        else:
            bub.append(float(link_cap[e]))
        row += 1
    if link_cap is None:
        for v in range(n):
            for e, (x, y) in enumerate(inst.edges):
                if v in (x, y):
                    rows.append(row); cols.append(xi(e)); vals.append(1.0)
            bub.append(float(max(rem_m[v], 0)))
            row += 1
    Aub = sp.csr_matrix((vals, (rows, cols)), shape=(row, nvar))
    bounds = [(0, None)] * nf
    bounds += [(0, None if t_cap is None else float(t_cap[r])) for r in range(R)]
    if link_cap is None:
        bounds += [(0, float(max(rem_c[e], 0))) for e in range(E)]
    res = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=bounds, method='highs')
    if res.x is None:
        raise RuntimeError('REPS LP failed: %s' % res.message)
    flows = []
    for r in range(R):
        fr = {}
        for a, arc in enumerate(arcs):
            val = res.x[fi(r, a)]
            if val > 1e-9:
                fr[arc] = val
        flows.append(fr)
    t = np.array([res.x[ti(r)] for r in range(R)])
    x = np.array([res.x[xi(e)] for e in range(E)]) if link_cap is None else None
    return flows, t, x


def decompose_flow(flow, s, d, eps=1e-7):
    """Decompose a fractional s-d flow into simple paths with values."""
    f = dict(flow)
    out = []
    while True:
        # DFS for a path with positive flow
        stack = [(s, [s])]
        seen = {s}
        found = None
        while stack:
            v, path = stack.pop()
            if v == d:
                found = path
                break
            for (x, y), val in f.items():
                if x == v and val > eps and y not in seen:
                    seen.add(y)
                    stack.append((y, path + [y]))
        if found is None:
            break
        b = min(f[(x, y)] for x, y in zip(found, found[1:]))
        for x, y in zip(found, found[1:]):
            f[(x, y)] -= b
        out.append((found, b))
    return out


def reps(inst, slot, rng, max_iter=30, rounding='ceil', reserve=None, attempt=None):
    """REPS [Zhao and Qiao 2021]. PFT reserves channels for the expected flow f of a link as round(f/p) and REPS then
    attempts round(f/p) channels, where round is set by `reserve` and `attempt` ('ceil' or 'floor'; `rounding` sets
    both). Main reimplementation: ceil for both, consistent with the linear program. The paper reserves ceil(.)
    channels in PFT but attempts floor(sum f / p) links (Sec. III-A): reps_written. reps_floor rounds both down."""
    reserve = reserve or rounding
    attempt = attempt or rounding
    req = slot.requests
    R = len(req)
    E = len(inst.edges)
    hmax = inst.regime.H_max
    p = inst.p
    F = np.zeros(E)                         # expected successful links provisioned per link
    t_hat = np.zeros(R, dtype=int)

    def att(Fv, how=None):
        if (how or reserve) == 'floor':
            return np.floor(Fv / p + 1e-9).astype(int)
        return np.ceil(Fv / p - 1e-9).astype(int)

    def node_use(A):
        use = np.zeros(inst.n, dtype=int)
        for e, (u, v) in enumerate(inst.edges):
            use[u] += A[e]
            use[v] += A[e]
        return use

    def can_add(path):
        F2 = F.copy()
        for e in edges_of(inst, path):
            F2[e] += 1
        A2 = att(F2)
        return (A2 <= inst.W).all() and (node_use(A2) <= inst.Q).all()

    def add(r, path):
        for e in edges_of(inst, path):
            F[e] += 1
        t_hat[r] += 1

    for _ in range(max_iter):
        A = att(F)
        rem_c = inst.W - A
        rem_m = inst.Q - node_use(A)
        flows, t, _ = flow_lp(inst, req, rem_c=rem_c, rem_m=rem_m, p_eff=p)
        if t.sum() < 1e-6:
            break
        paths = []
        for r, (s, d) in enumerate(req):
            for path, val in decompose_flow(flows[r], s, d):
                paths.append((r, path, val))
        added = False
        for r, path, val in paths:
            for _k in range(int(math.floor(val + 1e-9))):
                if can_add(path):
                    add(r, path)
                    added = True
        fr = sorted([(val - math.floor(val + 1e-9), r, path) for r, path, val in paths], key=lambda z: -z[0])
        for frac, r, path in fr:
            if frac > 1e-6 and can_add(path):
                add(r, path)
                added = True
        if not added:
            break
    attempts = att(F, attempt)
    return reps_select(inst, slot, rng, attempts, t_hat)


def reps_select(inst, slot, rng, attempts, t_hat):
    """REPS after generation: EPS (LP over the links obtained, randomized rounding) and ELS."""
    req = slot.requests
    R = len(req)
    hmax = inst.regime.H_max
    links = realized(inst, slot, attempts)
    cand = {r: [] for r in range(R)}
    if links.sum() > 0 and (t_hat is None or t_hat.sum() > 0):
        flows, t, _ = flow_lp(inst, req, link_cap=links, t_cap=t_hat)
        for r, (s, d) in enumerate(req):
            if t[r] < 1e-9:
                continue
            dec = decompose_flow(flows[r], s, d)
            tot = sum(v for _, v in dec)
            if tot < 1e-9:
                continue
            nr = int(math.floor(t[r] + 1e-9)) + (1 if rng.random() < t[r] - math.floor(t[r] + 1e-9) else 0)
            for _k in range(nr):
                x = rng.random() * tot
                acc = 0.0
                for path, v in dec:
                    acc += v
                    if x <= acc:
                        cand[r].append(path)
                        break
    # ELS
    resid = links.copy()
    sel = {r: [] for r in range(R)}

    def feasible(path):
        return (hmax is None or len(path) - 1 <= hmax) and all(resid[e] >= 1 for e in edges_of(inst, path))

    T = set(range(R))
    while T:
        m = min(len(sel[r]) for r in T)
        r = rng.choice(sorted(x for x in T if len(sel[x]) == m))
        cs = [P for P in cand[r] if feasible(P)]
        if cs:
            P = min(cs, key=len)
            cand[r].remove(P)
            for e in edges_of(inst, P):
                resid[e] -= 1
            sel[r].append(P)
        else:
            T.discard(r)
    T = set(range(R))
    while T:
        m = min(len(sel[r]) for r in T)
        r = rng.choice(sorted(x for x in T if len(sel[x]) == m))
        s, d = req[r]
        P = bfs_path(inst, s, d, lambda e: resid[e] >= 1, lambda v: True, hmax)
        if P is None:
            T.discard(r)
            continue
        for e in edges_of(inst, P):
            resid[e] -= 1
        sel[r].append(P)
    conns = [(r, P) for r in range(R) for P in sel[r]]
    return value_of(inst, conns), conns, links, attempts


# ----------------------------------------------------------------------------------------------------------------
def reactive(inst, slot, rng):
    hmax = inst.regime.H_max
    E = len(inst.edges)
    a = np.zeros(E, dtype=int)
    mem = inst.Q.copy()
    order = list(range(E))
    progressed = True
    while progressed:
        progressed = False
        rng.shuffle(order)
        for e in order:
            u, v = inst.edges[e]
            if a[e] < inst.W[e] and mem[u] >= 1 and mem[v] >= 1:
                a[e] += 1
                mem[u] -= 1
                mem[v] -= 1
                progressed = True
    links = realized(inst, slot, a)
    resid = links.copy()
    conns = []
    active = list(range(len(slot.requests)))
    while active:
        progressed = False
        order = active[:]
        rng.shuffle(order)
        for r in order:
            s, d = slot.requests[r]
            P = bfs_path(inst, s, d, lambda e: resid[e] >= 1, lambda v: True, hmax)
            if P is None:
                active.remove(r)
                continue
            for e in edges_of(inst, P):
                resid[e] -= 1
            conns.append((r, P))
            progressed = True
        if not progressed:
            break
    return value_of(inst, conns), conns, links, a


def reps_x(inst, slot, rng):
    """Diagnostic variant of REPS (not a published method): attempts taken from the PFT relaxation (x rounded down,
    then up by fractional part while channels and memory allow) instead of rounding expected link counts; the
    selection step (EPS and ELS) is REPS's, without a cap on connections per request."""
    E = len(inst.edges)
    flows, t, x = flow_lp(inst, slot.requests, rem_c=inst.W, rem_m=inst.Q, p_eff=inst.p)
    a = np.minimum(np.floor(x + 1e-9).astype(int), inst.W)
    use = np.zeros(inst.n, dtype=int)
    for e, (u, v) in enumerate(inst.edges):
        use[u] += a[e]
        use[v] += a[e]
    for e in sorted(range(E), key=lambda e: -(x[e] - math.floor(x[e] + 1e-9))):
        u, v = inst.edges[e]
        if x[e] - a[e] > 1e-6 and a[e] < inst.W[e] and use[u] < inst.Q[u] and use[v] < inst.Q[v]:
            a[e] += 1
            use[u] += 1
            use[v] += 1
    return reps_select(inst, slot, rng, a, None)


def reps_floor(inst, slot, rng):
    """REPS with floor(f/p) channels reserved and attempted."""
    return reps(inst, slot, rng, rounding='floor')


def reps_written(inst, slot, rng):
    """REPS as its paper writes it: ceil(f/p) channels reserved, floor(f/p) attempted."""
    return reps(inst, slot, rng, reserve='ceil', attempt='floor')


# The four compared methods (REPS with ceil/ceil rounding) and the other policies that decide before outcomes.
METHODS = {'greedy': greedy_sp, 'qcast': qcast, 'reps': reps, 'reactive': reactive}
OTHER_POLICIES = {'reps_ceil_floor': reps_written, 'reps_floor_floor': reps_floor, 'reps_x': reps_x}
