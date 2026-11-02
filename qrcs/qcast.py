"""Q-CAST [Shi and Qian, SIGCOMM 2020], transcribed from the authors' simulator
(github.com/sshi27/QuantumRouting: algorithm/OnlineAlgorithm.kt, topo/Topo.kt, functions P2, P2Extra, calCandidates,
P4 and e) into the channel model of this case study.

P2   Major paths, one at a time: for every request, the extended Dijkstra search (EDA) is run for widths
     w = min(free qubits at source and destination) down to 1 and the first width that yields a path is kept; among
     the requests, the path with the largest expected throughput EXT is reserved (w channels on each of its links).
     Repeat while EXT > 0.
P2+  Recovery paths: for each major path, in the order chosen, for l = 1..k and every pair of nodes l hops apart on it,
     one EDA path in the residual network is reserved if its EXT > 0.
P4   After link generation, each width-w major path is repaired w times: broken links of the major path are covered
     by its recovery paths (non-overlapping, by position), the new route is the shortest path in the major path with
     the covered segments replaced, and swaps are attempted along it on entangled, unused channels.

Departures from the simulator, all forced by the case-study model:
  - Swaps are not sampled. Every completed chain of swaps between a source and its destination counts q**(swaps),
    its probability of success. This is the expected value of the simulator's count, because no decision in P4
    depends on whether a swap succeeded.
  - In the hop-bound regimes, EDA does not extend paths beyond the bound and P4 skips a route longer than it.
  - Where the simulator iterates over a hash set (the recovery paths picked for one repair), recovery paths are taken
    in order of their position on the major path.
Like the simulator (and unlike the paper's text), there is no maximum path length h_m and no cap of 200 paths, and P4
uses the link states of the whole network.
"""
import heapq
from collections import deque
from math import comb

import numpy as np

from .model import realized

K_RECOVERY = 3          # link-state range k; reference value of the paper


# --------------------------------------------------------------------------------------------------------------------
# Expected throughput of a width-w path (Topo.e): distribution of the end-to-end width, built link by link
# --------------------------------------------------------------------------------------------------------------------
def _binom(w, p):
    return [comb(w, i) * p ** i * (1 - p) ** (w - i) for i in range(w + 1)]


def _extend(oldP, p, w):
    """Distribution of min(width so far, successes on the new link) when the new link has w channels."""
    ex = _binom(w, p)
    at_least = [sum(ex[i:]) for i in range(w + 1)]
    if sum(oldP) == 0.0:                     # first link of the path
        return ex
    return [oldP[i] * at_least[i] + ex[i] * sum(oldP[i + 1:]) for i in range(w + 1)]


def _ext(P, hops, q):
    return sum(m * P[m] for m in range(1, len(P))) * q ** (hops - 1)


# --------------------------------------------------------------------------------------------------------------------
# P2: candidate paths (calCandidates with its extended Dijkstra search)
# --------------------------------------------------------------------------------------------------------------------
def _eda(inst, src, dst, w, chan, mem, hmax):
    """Best path from src to dst by EXT with every link reserving w channels; None if there is none."""
    q = inst.regime.q
    adj = {}
    for e, (u, v) in enumerate(inst.edges):
        if chan[e] < w:
            continue
        if (u not in (src, dst) and mem[u] < 2 * w) or (v not in (src, dst) and mem[v] < 2 * w):
            continue
        adj.setdefault(u, []).append((v, e))
        adj.setdefault(v, []).append((u, e))
    if src not in adj or dst not in adj:
        return None
    best = {src: (float('inf'), [0.0] * (w + 1))}
    prev = {}
    heap = [(-float('inf'), 0, src, -1)]
    cnt = 1
    while heap:
        _, _, u, pu = heapq.heappop(heap)
        if u in prev:
            continue
        prev[u] = pu
        path = [u]
        while prev[path[-1]] != -1:
            path.append(prev[path[-1]])
        path.reverse()
        if u == dst:
            return best[u][0], path
        hops = len(path) - 1
        if hmax is not None and hops >= hmax:
            continue
        for v, e in adj[u]:
            P = _extend(best[u][1], inst.p[e], w)
            E = _ext(P, hops + 1, q)
            if best.get(v, (-float('inf'),))[0] < E:
                best[v] = (E, P)
                heapq.heappush(heap, (-E, cnt, v, u))
                cnt += 1
    return None


def _candidates(inst, pairs, chan, mem, hmax):
    """For each pair, the path found at the largest width that admits one: list of (EXT, width, path, pair index)."""
    out = []
    for idx, (s, d) in enumerate(pairs):
        for w in range(int(min(mem[s], mem[d])), 0, -1):
            found = _eda(inst, s, d, w, chan, mem, hmax)
            if found is not None:
                out.append((found[0], w, found[1], idx))
                break
    return out


def _reserve(inst, path, w, chan, mem, owned):
    """Reserve the w lowest free channels on every link of the path."""
    for a, b in zip(path, path[1:]):
        e = inst.eidx[(a, b)]
        first = inst.W[e] - chan[e]
        owned[e].extend(range(first, first + w))
        chan[e] -= w
        mem[a] -= w
        mem[b] -= w


# --------------------------------------------------------------------------------------------------------------------
# P4 helpers: channel states
# --------------------------------------------------------------------------------------------------------------------
class _Channels:
    def __init__(self, inst, slot, attempts):
        self.inst = inst
        self.ent = [[c < attempts[e] and bool(slot.omega[e][c]) for c in range(inst.W[e])] for e in range(len(inst.edges))]
        self.assigned = [[c < attempts[e] for c in range(inst.W[e])] for e in range(len(inst.edges))]
        self.swapped = [[set() for _ in range(inst.W[e])] for e in range(len(inst.edges))]   # nodes that swapped it
        self.used = [[False] * inst.W[e] for e in range(len(inst.edges))]
        self.internal = {}                   # node -> list of ((e, c), (e, c)) swaps attempted there

    def broken(self, a, b):
        """A link of the pair (a, b) is assigned, not swapped and failed."""
        e = self.inst.eidx[(a, b)]
        return any(self.assigned[e][c] and not self.swapped[e][c] and not self.ent[e][c]
                   for c in range(self.inst.W[e]))

    def free(self, n, other):
        """Lowest-index channel between n and other that is entangled, not swapped at n and not used."""
        e = self.inst.eidx[(n, other)]
        for c in range(self.inst.W[e]):
            if self.ent[e][c] and n not in self.swapped[e][c] and not self.used[e][c]:
                return e, c
        return None

    def swap(self, n, l1, l2):
        self.swapped[l1[0]][l1[1]].add(n)
        self.swapped[l2[0]][l2[1]].add(n)
        self.internal.setdefault(n, []).append((l1, l2))

    def chains(self, src, dst):
        """Chains of attempted swaps joining src to dst (Topo.getEstablishedEntanglements with every swap counted):
        list of (frozenset of channels, [nodes], swaps)."""
        out = []
        for v, e in ((self.inst.edges[e][1] if self.inst.edges[e][0] == src else self.inst.edges[e][0], e)
                     for e in range(len(self.inst.edges)) if src in self.inst.edges[e]):
            for c in range(self.inst.W[e]):
                if not self.ent[e][c] or src in self.swapped[e][c]:
                    continue
                stack = [((e, c), v, [src, v], [(e, c)])]
                while stack:
                    link, cur, nodes, links = stack.pop()
                    if cur == dst:
                        out.append((frozenset(links), nodes, len(nodes) - 2))
                        continue
                    for l1, l2 in self.internal.get(cur, []):
                        if link in (l1, l2):
                            nxt = l2 if link == l1 else l1
                            a, b = self.inst.edges[nxt[0]]
                            other = b if a == cur else a
                            if len(links) <= 4 * self.inst.n:          # guard against malformed cycles
                                stack.append((nxt, other, nodes + [other], links + [nxt]))
        return out


def _shortest_path(edge_set, src, dst):
    """Fewest-hop path over an undirected set of node pairs (Topo.shortestPath with unit weights)."""
    adj = {}
    for a, b in edge_set:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    prev = {src: None}
    dq = deque([src])
    while dq:
        u = dq.popleft()
        if u == dst:
            path = [dst]
            while prev[path[-1]] is not None:
                path.append(prev[path[-1]])
            return path[::-1]
        for v in adj.get(u, []):
            if v not in prev:
                prev[v] = u
                dq.append(v)
    return []


def _edges(path):
    return list(zip(path, path[1:]))


def _segment(path, first, last):
    """acc.dropWhile { it != first }.dropLastWhile { it != last }"""
    if first not in path:
        return []
    seg = path[path.index(first):]
    if last not in seg:
        return []
    return seg[:len(seg) - seg[::-1].index(last)]


# --------------------------------------------------------------------------------------------------------------------
def qcast(inst, slot, rng, k=K_RECOVERY):
    hmax = inst.regime.H_max
    q = inst.regime.q
    chan = inst.W.copy()
    mem = inst.Q.copy().astype(int)
    owned = [[] for _ in inst.edges]
    pairs = slot.requests

    # P2: major paths
    majors = []                              # (request index, width, path)
    while True:
        cands = _candidates(inst, pairs, chan, mem, hmax)
        if not cands:
            break
        E, w, path, r = max(cands, key=lambda c: c[0])      # first maximum, as maxBy
        if not E > 0.0:
            break
        _reserve(inst, path, w, chan, mem, owned)
        majors.append((r, w, path))
    # P2 extra: recovery paths
    recov = {i: [] for i in range(len(majors))}
    for i, (r, w, p) in enumerate(majors):
        for l in range(1, k + 1):
            for j in range(0, len(p) - l):
                cands = _candidates(inst, [(p[j], p[j + l])], chan, mem, hmax)
                if cands and cands[0][0] > 0.0:
                    E, rw, rpath, _ = cands[0]
                    _reserve(inst, rpath, rw, chan, mem, owned)
                    recov[i].append((rw, rpath))

    attempts = inst.W - chan
    links = realized(inst, slot, attempts)
    st = _Channels(inst, slot, attempts)
    value, conns = 0.0, []

    # P4
    for i, (r, width, major) in enumerate(majors):
        src, dst = major[0], major[-1]
        before = {ch[0] for ch in st.chains(src, dst)} if len(major) > 2 else set()
        rps = sorted(recov[i], key=lambda x: len(x[1]) * 10000 + major.index(x[1][0]))
        rp_width = {tuple(p): w for w, p in rps}
        pos = {n: t for t, n in enumerate(major)}
        medges = [(t, t + 1) for t in range(len(major) - 1)]
        for _ in range(width):
            broken = [ed for ed in medges if st.broken(major[ed[0]], major[ed[1]])]
            edge_to_rps = {ed: [] for ed in broken}
            rp_to_edges = {tuple(p): [] for _, p in rps}
            for _, p in rps:
                s1, s2 = pos[p[0]], pos[p[-1]]
                for ed in zip(range(s1, s2), range(s1 + 1, s2 + 1)):
                    if ed in edge_to_rps:
                        rp_to_edges[tuple(p)].append(ed)
                        edge_to_rps[ed].append(tuple(p))
            picked, repaired = set(), set()
            for be in broken:
                if be in repaired:
                    continue
                ok, nxt = False, 0
                options = sorted((p for p in edge_to_rps[be] if rp_width[p] > 0 and p not in picked),
                                 key=lambda p: pos[p[0]] * 10000 + pos[p[-1]])
                for p in options:
                    if pos[p[0]] < nxt:
                        continue
                    nxt = pos[p[-1]]
                    new_picked, new_repaired = set(picked), set(repaired)
                    clash = False
                    for ed in set(rp_to_edges[p]) - {be}:
                        if [x for x in edge_to_rps[ed] if x in new_picked and x != p]:
                            clash = True
                            break
                        new_repaired.add(ed)
                    if clash:
                        continue
                    ok = True
                    new_repaired.add(be)
                    new_picked.add(p)
                    for x in picked - new_picked:
                        rp_width[x] += 1
                    for x in new_picked - picked:
                        rp_width[x] -= 1
                    picked, repaired = new_picked, new_repaired
                    break
                if not ok:
                    break
            route = list(major)
            for p in sorted(picked, key=lambda p: (pos[p[0]], pos[p[-1]])):
                if not route:
                    break
                es = (set(_edges(route)) - set(_edges(_segment(route, p[0], p[-1])))) | set(_edges(list(p)))
                route = _shortest_path(es, route[0], route[-1])
            if len(route) < 3 or (hmax is not None and len(route) - 1 > hmax):
                continue
            for a, n, b in zip(route, route[1:], route[2:]):
                l1, l2 = st.free(n, a), st.free(n, b)
                if l1 is not None and l2 is not None:
                    st.swap(n, l1, l2)
                    st.used[l1[0]][l1[1]] = True
                    if b == route[-1]:
                        st.used[l2[0]][l2[1]] = True
        if len(major) > 2:
            for key, nodes, swaps in st.chains(src, dst):
                if key not in before:
                    value += q ** swaps
                    conns.append((r, nodes))
        else:                                # one-hop major path: direct links, no swap
            e = inst.eidx[(src, dst)]
            direct = [c for c in range(inst.W[e]) if st.ent[e][c] and src not in st.swapped[e][c] and not st.used[e][c]]
            for c in direct[:width]:
                st.used[e][c] = True
                value += 1.0
                conns.append((r, [src, dst]))
    return value, conns, links, attempts
