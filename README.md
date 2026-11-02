# Concurrent entanglement routing: case study

Code for the case study (Section 9) of

> T. Islam and M. Arifuzzaman, *Quantum Network Routing: Complexity, Guarantees, and Design Trade-offs*, manuscript, 2026.

The case study compares four concurrent entanglement-routing methods with optimization references on a common slotted
network model. In every slot, a method first decides how many channels to **attempt** on each link (without knowing
which attempts will succeed), then **selects** end-to-end connections over the links that succeeded. The references
separate these two decisions from the advantage of knowing link outcomes in advance.

| Methods (reimplemented from the papers) | References (solved with HiGHS) |
|---|---|
| fewest-hop greedy (a simple proactive baseline) | **hindsight optimum**: knows the outcomes before choosing attempts and routes |
| Q-CAST (Shi and Qian, SIGCOMM 2020), from the authors' simulator | **expected-capacity LP**: link capacity W·p, fractional |
| REPS (Zhao and Qiao, INFOCOM 2021), three rounding rules | **selection optimum**: best routing over the links a method obtained |
| reactive greedy (loosely based on Pant et al., npj QI 2019) | **LP+IP**: attempts from the REPS relaxation, then the selection optimum |

REPS and the greedy methods were written from the published descriptions, and Q-CAST from the authors' simulator; the
results describe these implementations under this model. The differences from the papers are listed under [Methods](#methods).

## Quick start

Python 3.11 or later.

```bash
git clone https://github.com/HasanTasdiq/quantum-routing-case-study.git
cd quantum-routing-case-study
python -m pip install -r requirements.txt

python verify.py                     # correctness checks, about a minute
python run_case_study.py --quick     # 20-node regimes, 2 networks x 3 slots, under a minute
python analyze.py --results results_quick
```

## Reproduce the paper

```bash
python run_case_study.py             # all cells: about 1 hour with 2 workers (use --workers N for more cores)
python analyze.py                    # tables on screen and results/summary.json
python path_search.py                # the single-request path-search experiment, under a minute
python verify.py --full              # optional: feasibility of every method on every slot, about 15 minutes
```

`run_case_study.py` writes one file per network to `results/` and skips files that already exist, so an interrupted
run can be resumed. All randomness is seeded; with the versions in `requirements.txt` the delivered values, attempts
and links are reproducible exactly. Solver times depend on the machine.

Expected results on 20-node networks (delivered pairs as a fraction of the hindsight optimum; ten networks of 25 slots;
`analyze.py` also prints 95% intervals):

| regime | LP / hindsight | fewest-hop greedy | Q-CAST | REPS ceil/ceil | reactive | LP+IP |
|---|---|---|---|---|---|---|
| generic | 1.04 | 0.58 | 0.68 | 0.70 | 0.57 | 0.81 |
| few-resources | 1.15 | 0.59 | 0.75 | 0.26 | 0.33 | 0.71 |
| lossy-links | 1.19 | 0.33 | 0.49 | 0.35 | 0.42 | 0.63 |
| lossy-swaps | 1.03 | 0.65 | 0.78 | 0.71 | 0.50 | 0.80 |
| hop-bound | 1.04 | 0.65 | 0.72 | 0.70 | 0.56 | 0.75 |
| combined | 1.38 | 0.54 | 0.76 | 0.12 | 0.27 | 0.65 |

## Repository layout

```
qrcs/model.py        regimes, random networks, slots and channel outcomes
qrcs/methods.py      fewest-hop greedy, REPS and reactive greedy
qrcs/qcast.py        Q-CAST, transcribed from the authors' simulator
qrcs/programs.py     the hop-layered program behind all optimization references
run_case_study.py    runs every method and reference on every slot -> results/*.jsonl
analyze.py           ratios to the hindsight optimum with network-bootstrap intervals -> results/summary.json
path_search.py       single-request path search under a fidelity threshold (exact algorithm and heuristics)
verify.py            correctness checks
```

## Model

- **Network.** n nodes uniform in the unit square; nodes u, v are linked with probability
  β·exp(−d_uv/(αL)), α = 0.4, L the largest node distance, β set for an expected mean degree of four (capped at 1);
  disconnected draws are rejected. Channels per link W_e and memory qubits per node Q_v are uniform integers in the
  regime's ranges.
- **Link generation.** An attempted channel on link e succeeds in a slot with probability p_e = exp(−d_e/ℓ); ℓ is set
  per network by bisection so that the mean of p_e equals the regime's target.
- **Resources.** Each attempted channel holds one memory qubit at each end of its link for the whole slot:
  a_e ≤ W_e and Σ_{e at v} a_e ≤ Q_v.
- **Slot.** Six distinct source–destination pairs; a request may receive any number of connections. Before any decision,
  the slot draws an outcome for every channel of every link; a method attempting a_e channels on e sees the first a_e.
- **Throughput.** A connection over k links needs k − 1 swaps, each succeeding with probability q; it counts
  q^(k−1). Methods and references are scored the same way. Pairs do not decay and the slot has no duration.

| regime | mean p | swap q | channels per link | qubits per node | hop bound |
|---|---|---|---|---|---|
| generic | 0.8 | 0.9 | 3–5 | 10–14 | none |
| few-resources | 0.8 | 0.9 | 1–2 | 2–4 | none |
| lossy-links | 0.3 | 0.9 | 3–5 | 10–14 | none |
| lossy-swaps | 0.8 | 0.5 | 3–5 | 10–14 | none |
| hop-bound | 0.8 | 0.9 | 3–5 | 10–14 | 3 links |
| combined | 0.3 | 0.5 | 1–2 | 2–4 | 3 links |

Every regime is run on 20 nodes, and the generic regime also on 10 and 30 nodes. Network k with n nodes uses seed
1000k + n; slot t of it uses seed 10000k + 100n + t; methods use seed t for tie-breaking and randomized rounding.

## Methods

Each method returns its delivered value, its connections, the links it obtained and its attempts
(`qrcs/methods.py`). The table lists what was implemented and where it departs from the published description.

| method | what the code does | differences from the paper |
|---|---|---|
| fewest-hop greedy | reserves width-1 fewest-hop paths for the requests in turn until resources run out; a path delivers only if all its links succeed | a simple proactive baseline, not taken from a specific paper |
| Q-CAST | transcribed from the authors' simulator ([sshi27/QuantumRouting](https://github.com/sshi27/QuantumRouting), `OnlineAlgorithm.kt`): for each request, the extended Dijkstra search at decreasing widths keeps the first width with a path; the path with the largest expected throughput over all requests is reserved, repeatedly; recovery paths by the same search between nodes 1 to 3 hops apart on each major path; after generation, each width-w major path is repaired w times with non-overlapping recovery paths and swaps are attempted along the shortest resulting route | swaps are not sampled: each completed chain of swaps counts its success probability (the mean of the simulator's count, since no decision depends on swap outcomes); the hop bound is applied in the search and to repaired routes; recovery paths picked for one repair are taken in order of position (the simulator iterates over a hash set). Like the simulator, and unlike the paper's text, there is no maximum path length h_m and P4 uses whole-network link states |
| REPS | PFT: LP relaxation of the provisioning problem on the remaining resources, path decomposition, integer parts then fractional parts added unit by unit while channels and memory allow, up to 30 rounds; EPS: LP over the links obtained and randomized rounding; ELS: greedy selection, then fewest-hop paths over leftover links | the expected flow f of a link is turned into channels with ceil/ceil (main), ceil/floor (as the paper writes it: reserve ⌈f/p⌉, attempt ⌊f/p⌋) or floor/floor; the relaxation has no hop limit (the bound is enforced in selection) |
| reactive greedy | spreads memory over the links (one more channel per link per round, random order) while resources allow; after generation, fewest-hop paths over the links obtained, requests in turn | loosely based on the global-link-state protocol of Pant et al., which attempts every channel with one memory per channel and shares several flows in time or space |

## Optimization references

All references solve one hop-layered program (`qrcs/programs.py`). Variable x[r, k, u→v] ≥ 0 counts connections of
request r whose k-th link goes from u to v, k = 1..H (H = 3 with the hop bound, n − 1 otherwise). The objective is
Σ q^(k−1) over connections, subject to flow conservation between layers, link capacity c_e and, where stated, node
memory (each use of a link holds one qubit at each end).

| reference | capacity c_e | memory constraint | variables |
|---|---|---|---|
| hindsight optimum | channels that would succeed, Σ_c ω_ec | yes | integer |
| expected-capacity LP | W_e·p_e | yes | continuous |
| selection optimum of a method | links the method obtained | no (already committed) | integer |
| LP+IP | links obtained by the REPS-relaxation attempts | no | integer |

Solver settings: SciPy `milp`/`linprog` (HiGHS), relative MIP gap 1e−6, time limits 120 s (hindsight) and 60 s
(selection). The recorded times are those of the solver call only. If a selection program hits its time limit, its
best feasible value is used and its gap is recorded.

## Output

Each line of `results/<regime>_n<nodes>_net<k>.jsonl` is one slot: the requests, `hindsight`, `lp`,
`lp_relaxed_hindsight`, and for each method `<name>`, `<name>_attempts`, `<name>_links`, `<name>_time` and
`<name>_selection` (with `_optimal`, `_bound`, `_gap`, `_time`); also `reps_ceil_floor`, `reps_floor_floor`, `reps_x`
and `lp_ip`.

## Citation

```bibtex
@unpublished{IslamArifuzzaman2026,
  author = {Tasdiqul Islam and Md Arifuzzaman},
  title  = {Quantum Network Routing: Complexity, Guarantees, and Design Trade-offs},
  note   = {Manuscript},
  year   = {2026}
}
```

## License

MIT (see `LICENSE`).
