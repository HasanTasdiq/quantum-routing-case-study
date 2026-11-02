"""Summarize the results of run_case_study.py: throughput relative to the hindsight optimum, with 95% intervals.

    python analyze.py                 # reads results/, prints tables, writes results/summary.json
    python analyze.py --results results_quick

Estimator: for a method with value Y[s,t] in slot t of network s and hindsight value Z[s,t], the reported ratio is
sum(Y) / sum(Z) (ratio of means, so slots with Z = 0 need no special handling). The 95% interval is a percentile
bootstrap with 2,000 resamples of the networks (all slots of a network stay together; numerator and denominator are
resampled together).
"""
import argparse
import glob
import json
import os

import numpy as np

METHODS = ['greedy', 'qcast', 'reps', 'reactive']
LABELS = {'greedy': 'fewest-hop greedy', 'qcast': 'Q-CAST', 'reps': 'REPS ceil/ceil', 'reactive': 'reactive greedy',
          'reps_ceil_floor': 'REPS ceil/floor', 'reps_floor_floor': 'REPS floor/floor', 'reps_x': 'LP provisioning',
          'lp_ip': 'LP+IP'}
B = 2000                                    # bootstrap resamples


def load(folder):
    cells = {}
    for f in sorted(glob.glob(os.path.join(folder, '*_n*_net*.jsonl'))):
        with open(f) as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    cells.setdefault((r['regime'], r['n']), []).append(r)
    return cells


def network_sums(values, networks):
    ids = np.unique(networks)
    return np.array([values[networks == i].sum() for i in ids])


def ratio_ci(num, den, networks, seed=0):
    """sum(num) / sum(den) with a 95% percentile bootstrap interval over networks."""
    num, den, networks = np.asarray(num, float), np.asarray(den, float), np.asarray(networks)
    sn, sd = network_sums(num, networks), network_sums(den, networks)
    idx = np.random.default_rng(seed).integers(0, len(sn), size=(B, len(sn)))
    boot = sn[idx].sum(1) / sd[idx].sum(1)
    return [float(num.sum() / den.sum()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def paired_difference(a, b, den, networks, seed=1):
    """(sum(a) - sum(b)) / sum(den) with a 95% interval from the same resampling of networks."""
    a, b, den = (np.asarray(x, float) for x in (a, b, den))
    networks = np.asarray(networks)
    sd, sdiff = network_sums(den, networks), network_sums(a - b, networks)
    idx = np.random.default_rng(seed).integers(0, len(sd), size=(B, len(sd)))
    boot = sdiff[idx].sum(1) / sd[idx].sum(1)
    return [float((a - b).sum() / den.sum()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def summarize(rs):
    net = [r['network'] for r in rs]
    ho = [r['hindsight_bound'] for r in rs]          # equals the hindsight optimum, which is solved to optimality
    col = lambda k: [r[k] for r in rs]
    out = {'slots': len(rs), 'networks': len(set(net)),
           'hindsight_mean': float(np.mean(col('hindsight'))),
           'lp_over_hindsight': ratio_ci(col('lp'), ho, net),
           # LP / HO = (LP / relaxed hindsight) x (relaxed hindsight / HO): expectation part x integrality part
           'lp_expectation_part': float(sum(col('lp')) / sum(col('lp_relaxed_hindsight'))),
           'lp_integrality_part': float(sum(col('lp_relaxed_hindsight')) / sum(ho))}
    for m in METHODS:
        out[m] = {'over_hindsight': ratio_ci(col(m), ho, net),
                  'selection_over_hindsight': ratio_ci(col(m + '_selection'), ho, net),
                  'attempts_per_slot': float(np.mean(col(m + '_attempts')))}
    for m in ('reps_ceil_floor', 'reps_floor_floor', 'reps_x'):
        out[m] = {'over_hindsight': ratio_ci(col(m), ho, net), 'attempts_per_slot': float(np.mean(col(m + '_attempts')))}
    out['lp_ip'] = {'over_hindsight': ratio_ci(col('lp_ip'), ho, net)}
    # Best value among policies that decide their attempts before seeing the outcomes
    cand = {m + '_selection': out[m]['selection_over_hindsight'][0] for m in METHODS}
    cand.update({m: out[m]['over_hindsight'][0] for m in ('reps_ceil_floor', 'reps_floor_floor', 'reps_x', 'lp_ip')})
    best = max(cand, key=cand.get)
    out['best_without_foresight'] = {'policy': best, 'over_hindsight': cand[best]}
    # Ranking of the four methods and paired differences of consecutive methods
    ranking = sorted(METHODS, key=lambda m: -np.mean(col(m)))
    out['ranking'] = ranking
    out['consecutive_differences'] = {'%s-%s' % (a, b): paired_difference(col(a), col(b), ho, net)
                                      for a, b in zip(ranking, ranking[1:])}
    # Solver statistics
    sel = [(r['%s_selection_optimal' % m], r['%s_selection_time' % m]) for r in rs for m in METHODS]
    out['solver'] = {
        'hindsight_all_optimal': all(col('hindsight_optimal')),
        'hindsight_time_median_max': [float(np.median(col('hindsight_time'))), float(max(col('hindsight_time')))],
        'selection_not_optimal': sum(not o for o, _ in sel), 'selection_programs': len(sel),
        'lp_ip_not_optimal': sum(not o for o in col('lp_ip_optimal')),
        'lp_ip_time_median_max': [float(np.median(col('lp_ip_time'))), float(max(col('lp_ip_time')))],
        'largest_gap_of_time_limited_runs': max([r[k + '_gap'] or 0.0 for r in rs for k in
                                                 [m + '_selection' for m in METHODS] + ['lp_ip']
                                                 if not r[k + '_optimal']], default=None)}
    return out


def pct(x):
    return '%5.1f%%' % (100 * x)


def show(summary):
    for cell, s in summary.items():
        print('\n== %s: %d networks x %d slots; hindsight optimum %.2f pairs per slot'
              % (cell, s['networks'], s['slots'] // max(s['networks'], 1), s['hindsight_mean']))
        lp = s['lp_over_hindsight']
        print('   LP / hindsight %.3f [%.3f, %.3f]  (expectation part %.3f, integrality part %.3f)'
              % (*lp, s['lp_expectation_part'], s['lp_integrality_part']))
        print('   %-20s %-24s %-24s %s' % ('', 'delivered / hindsight', 'selection / hindsight', 'attempts/slot'))
        for m in METHODS:
            d, p = s[m]['over_hindsight'], s[m]['selection_over_hindsight']
            print('   %-20s %s [%s, %s]   %s [%s, %s]   %6.1f' % (LABELS[m], pct(d[0]), pct(d[1]), pct(d[2]),
                                                                pct(p[0]), pct(p[1]), pct(p[2]),
                                                                s[m]['attempts_per_slot']))
        for m in ('reps_ceil_floor', 'reps_floor_floor', 'lp_ip'):
            d = s[m]['over_hindsight']
            extra = '   %6.1f' % s[m]['attempts_per_slot'] if 'attempts_per_slot' in s[m] else ''
            print('   %-20s %s [%s, %s]%s%s' % (LABELS[m], pct(d[0]), pct(d[1]), pct(d[2]), ' ' * 27 if extra else '',
                                               extra))
        b = s['best_without_foresight']
        print('   best without foresight: %s (%s)' % (pct(b['over_hindsight']), b['policy']))
        sv = s['solver']
        print('   solver: hindsight all optimal %s (max %.1f s); LP+IP selection median %.2f s, max %.1f s; '
              'time-limited selection programs %d of %d'
              % (sv['hindsight_all_optimal'], sv['hindsight_time_median_max'][1], *sv['lp_ip_time_median_max'],
                 sv['selection_not_optimal'] + sv['lp_ip_not_optimal'], sv['selection_programs'] + s['slots']))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', default='results', help='folder written by run_case_study.py')
    a = ap.parse_args()
    cells = load(a.results)
    if not cells:
        raise SystemExit('no results in %s/; run run_case_study.py first' % a.results)
    summary = {'%s_n%d' % key: summarize(rs) for key, rs in sorted(cells.items())}
    show(summary)
    with open(os.path.join(a.results, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=1)
    print('\nwrote %s' % os.path.join(a.results, 'summary.json'))


if __name__ == '__main__':
    main()
