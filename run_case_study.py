"""Run the case study and write one JSON line per slot.

    python run_case_study.py                      # everything in the paper (about 1 hour on 2 cores)
    python run_case_study.py --quick              # 20-node regimes, 2 networks x 3 slots (seconds)
    python run_case_study.py --regimes generic combined --networks 3 --workers 4

A job is one network of one regime and size; it writes results/<regime>_n<nodes>_net<k>.jsonl. Finished jobs are
skipped, so an interrupted run can be resumed. In every slot, all methods and references see the same network,
requests and channel outcomes.
"""
import argparse
import json
import os
import random
import time
from multiprocessing import Pool

import numpy as np

from qrcs import (REGIMES, METHODS, OTHER_POLICIES, make_instance, make_slot, hop_limit, network_seed, slot_seed,
                  hindsight_optimum, lp_bound, hindsight_relaxation, post_generation_optimum)

# Cells of the paper: every regime on 20 nodes, and the generic regime also on 10 and 30 nodes.
PAPER_CELLS = [(name, 20) for name in REGIMES] + [('generic', 10), ('generic', 30)]


def solver_fields(prefix, value, info):
    """Value, solver time, optimality flag, best bound and relative gap of one program."""
    return {prefix: value, prefix + '_time': info['time'], prefix + '_optimal': bool(info.get('optimal', True)),
            prefix + '_bound': info.get('bound', value), prefix + '_gap': info.get('mip_gap')}


def run_slot(inst, slot, t):
    H = hop_limit(inst)
    rec = {}
    # References that do not depend on a method
    v, _, info = hindsight_optimum(inst, slot, H, time_limit=120.0)
    rec.update(solver_fields('hindsight', v, info))
    v, _, info = lp_bound(inst, slot, H)
    rec.update({'lp': v, 'lp_time': info['time']})
    rec['lp_relaxed_hindsight'] = hindsight_relaxation(inst, slot, H)[0]
    # Methods, each followed by the best selection over the links it obtained
    for name, method in list(METHODS.items()) + list(OTHER_POLICIES.items()):
        t0 = time.perf_counter()
        v, conns, links, attempts = method(inst, slot, random.Random(t))
        rec.update({name: v, name + '_time': time.perf_counter() - t0,
                    name + '_attempts': int(np.asarray(attempts).sum()), name + '_links': int(links.sum()),
                    name + '_connections': len(conns)})
        if name in METHODS or name == 'reps_x':
            sv, _, info = post_generation_optimum(inst, slot, links, H)
            # reps_x followed by the selection program is the LP+IP policy
            rec.update(solver_fields(name + '_selection' if name in METHODS else 'lp_ip', sv, info))
    return rec


def job(args):
    regime, n, k, slots, out = args
    path = os.path.join(out, '%s_n%d_net%d.jsonl' % (regime, n, k))
    if os.path.exists(path):
        return path, 'skipped'
    inst = make_instance(n, REGIMES[regime], network_seed(n, k))
    lines = []
    for t in range(slots):
        slot = make_slot(inst, random.Random(slot_seed(n, k, t)))
        rec = {'regime': regime, 'n': n, 'network': k, 'slot': t, 'links': len(inst.edges), 'diameter': inst.diam,
               'requests': slot.requests}
        rec.update(run_slot(inst, slot, t))
        lines.append(json.dumps(rec))
    with open(path + '.tmp', 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    os.replace(path + '.tmp', path)
    return path, 'done'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--regimes', nargs='+', choices=list(REGIMES), help='default: all cells of the paper')
    ap.add_argument('--nodes', type=int, nargs='+', help='network sizes (default: 20; generic also 10 and 30)')
    ap.add_argument('--networks', type=int, default=10, help='networks per regime and size (default 10)')
    ap.add_argument('--slots', type=int, default=25, help='slots per network (default 25)')
    ap.add_argument('--workers', type=int, default=2, help='parallel jobs (default 2)')
    ap.add_argument('--out', default='results', help='output folder (default results/)')
    ap.add_argument('--quick', action='store_true', help='smoke test: 20-node regimes, 2 networks x 3 slots, output in results_quick/')
    a = ap.parse_args()
    cells = PAPER_CELLS
    if a.quick:                              # 20-node regimes only, 2 networks x 3 slots
        a.networks, a.slots, a.out = 2, 3, 'results_quick' if a.out == 'results' else a.out
        cells = [(r, 20) for r in REGIMES]
    if a.regimes or a.nodes:
        cells = [(r, n) for r in (a.regimes or list(REGIMES)) for n in (a.nodes or [20])]
    os.makedirs(a.out, exist_ok=True)
    jobs = [(r, n, k, a.slots, a.out) for k in range(a.networks) for r, n in cells]
    print('%d jobs (%d cells x %d networks), %d slots each, %d workers -> %s/'
          % (len(jobs), len(cells), a.networks, a.slots, a.workers, a.out), flush=True)
    with Pool(a.workers) as pool:
        for i, (path, status) in enumerate(pool.imap_unordered(job, jobs), 1):
            print('[%3d/%d] %s %s %s' % (i, len(jobs), status, os.path.basename(path), time.strftime('%H:%M:%S')),
                  flush=True)


if __name__ == '__main__':
    main()
