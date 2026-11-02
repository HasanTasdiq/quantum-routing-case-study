"""Path search with a fidelity threshold (single request), the timing experiment of the paper.

Problem: find a source-destination path that maximizes the bottleneck rate subject to a fidelity threshold, with
Werner links and perfect swaps (the Werner parameters of the links multiply along a path). This is polynomial:
threshold enumeration tries each distinct link rate, keeps the links at least that fast and takes the path of highest
fidelity. The script also scores four heuristics against it: widest path, highest-fidelity path, fewest hops and
Dijkstra on an additive cost (inverse rate plus scaled -log Werner parameter). A path scores its bottleneck rate if it
meets the threshold and 0 otherwise.

Instances: 20 Waxman networks of 30 nodes (seeds 500..519), 50 random pairs each, integral link rates 1..20, link
fidelities uniform in [0.93, 0.995]; thresholds 0.87 and 0.835.

    python path_search.py            # writes results/path_search.json and prints it (under a minute)
"""
import heapq
import json
import math
import os
import random
import time

import numpy as np
import networkx as nx

from qrcs.model import waxman

OUT = 'results'


def werner(F):
    return (4 * F - 1) / 3


def best_fidelity_path(G, s, d, ok=lambda u, v: True):
    """Dijkstra on -log w (maximum product of Werner parameters)."""
    dist = {s: 0.0}
    prev = {s: None}
    heap = [(0.0, s)]
    done = set()
    while heap:
        c, v = heapq.heappop(heap)
        if v in done:
            continue
        done.add(v)
        if v == d:
            break
        for u in G.neighbors(v):
            if not ok(v, u):
                continue
            c2 = c - math.log(G[v][u]['w'])
            if c2 < dist.get(u, float('inf')) - 1e-15:
                dist[u] = c2
                prev[u] = v
                heapq.heappush(heap, (c2, u))
    if d not in prev:
        return None
    path = [d]
    while prev[path[-1]] is not None:
        path.append(prev[path[-1]])
    return path[::-1]


def widest_path(G, s, d):
    """Maximum bottleneck rate, ties broken by fewest hops."""
    best = {s: (float('inf'), 0)}
    prev = {s: None}
    heap = [(-float('inf'), 0, s)]
    done = set()
    while heap:
        nb, h, v = heapq.heappop(heap)
        if v in done:
            continue
        done.add(v)
        if v == d:
            break
        for u in G.neighbors(v):
            b = min(-nb, G[v][u]['rate'])
            key = (b, -(h + 1))
            cur = best.get(u)
            if cur is None or key > (cur[0], -cur[1]):
                best[u] = (b, h + 1)
                prev[u] = v
                heapq.heappush(heap, (-b, h + 1, u))
    path = [d]
    while prev[path[-1]] is not None:
        path.append(prev[path[-1]])
    return path[::-1]


def composite_path(G, s, d, a, b):
    for u, v in G.edges():
        G[u][v]['cc'] = a / G[u][v]['rate'] + b * (-math.log(G[u][v]['w']))
    return nx.dijkstra_path(G, s, d, weight='cc')


def value(G, path, Wstar):
    if path is None:
        return 0.0
    W = 1.0
    r = float('inf')
    for u, v in zip(path, path[1:]):
        W *= G[u][v]['w']
        r = min(r, G[u][v]['rate'])
    return r if W >= Wstar - 1e-12 else 0.0


def exact(G, s, d, Wstar):
    rates = sorted({G[u][v]['rate'] for u, v in G.edges()}, reverse=True)
    for r in rates:
        path = best_fidelity_path(G, s, d, ok=lambda u, v: G[u][v]['rate'] >= r)
        if path is not None and value(G, path, Wstar) > 0:
            return path, value(G, path, Wstar)
    return None, 0.0


def main(n=30, topologies=20, pairs=50, Fstar_list=(0.87, 0.835), Frange=(0.93, 0.995)):
    out = {}
    for Fstar in Fstar_list:
        Wstar = werner(Fstar)
        stats = {k: {'opt': 0, 'infeasible': 0, 'ratio': []} for k in ('widest', 'fidelity', 'hops', 'composite')}
        times = []
        solvable = 0
        total = 0
        for tseed in range(topologies):
            rng = np.random.default_rng(500 + tseed)
            G, pos = waxman(n, rng)
            for u, v in G.edges():
                G[u][v]['rate'] = float(rng.integers(1, 21))
                G[u][v]['w'] = werner(float(rng.uniform(*Frange)))
            mean_inv_rate = np.mean([1 / G[u][v]['rate'] for u, v in G.edges()])
            mean_logw = np.mean([-math.log(G[u][v]['w']) for u, v in G.edges()])
            prng = random.Random(tseed)
            for _ in range(pairs):
                s, d = prng.sample(range(n), 2)
                total += 1
                t0 = time.perf_counter()
                _, opt = exact(G, s, d, Wstar)
                times.append(time.perf_counter() - t0)
                if opt <= 0:
                    continue
                solvable += 1
                cand = {'widest': widest_path(G, s, d), 'fidelity': best_fidelity_path(G, s, d),
                        'hops': nx.shortest_path(G, s, d),
                        'composite': composite_path(G, s, d, 1 / mean_inv_rate, 1 / mean_logw)}
                for k, path in cand.items():
                    v = value(G, path, Wstar)
                    stats[k]['ratio'].append(v / opt)
                    stats[k]['opt'] += int(v >= opt - 1e-9)
                    stats[k]['infeasible'] += int(v == 0)
        out[str(Fstar)] = {
            'n': n, 'instances': total, 'solvable': solvable,
            'exact_time_ms_mean': 1000 * float(np.mean(times)), 'exact_time_ms_max': 1000 * float(np.max(times)),
            'heuristics': {k: {'optimal_share': v['opt'] / solvable, 'infeasible_share': v['infeasible'] / solvable,
                               'mean_ratio': float(np.mean(v['ratio']))} for k, v in stats.items()}}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'path_search.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    return out


if __name__ == '__main__':
    print(json.dumps(main(), indent=1))
