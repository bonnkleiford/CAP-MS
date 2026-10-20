# CAP-MS: a benchmark for multistage reliable facility location under disruptions

[![Data: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey.svg)](LICENSE-DATA)
[![Code: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)

**CAP-MS** extends the capacitated facility location instances *CAP* of the [OR-Library](http://people.brunel.ac.uk/~mastjjb/jeb/orlib/capinfo.html) (Beasley 1990) to a **multistage, stochastic** setting: facilities fail at random over a planning horizon and the planner decides, stage by stage, which facilities to (re-)open, which to close and how to serve demand.

To our knowledge, CAP-MS is the first public benchmark for multistage capacitated facility location with facility disruptions. It contains **135 instances and 540 configurations**, three sets of disruption paths per instance (for training and out-of-sample testing), and reference results of two solution methods, SDDiP and shadow price approximation (SPA).

---

## Contents

- [The problem](#the-problem)
- [The benchmark](#the-benchmark)
- [Quick start](#quick-start)
- [Repository layout and file formats](#repository-layout-and-file-formats)
- [Reference results](#reference-results)
- [Generating new disruption paths](#generating-new-disruption-paths)
- [Reporting results](#reporting-results)
- [How to cite](#how-to-cite)
- [License](#license)

---

## The problem

A set of facilities $i = 1,\dots,F$ serves customers $j = 1,\dots,B$ over stages $t = 1,\dots,T$. All facilities start closed. In every stage, the planner first observes which facilities are disrupted ($\xi_{it} = 1$), then decides:

- which facilities to (re-)open, $y_{it} \in \{0,1\}$, at fixed cost $f_i$ (a disrupted facility cannot be opened);
- in **Model B** only, which open facilities to close voluntarily, $v_{it} \in \{0,1\}$, at closing cost $c_i$;
- how to serve demand $D_j$ from the open facilities, $x_{ijt} \ge 0$, at unit cost $d_{ij}$ and within capacity $C_i$.

A disruption closes an open facility; it stays closed until the planner re-opens it. Every open facility costs $h_i$ per stage. The state of facility $i$ follows

$$u_{it} = u_{i,t-1}(1-\xi_{it}) + y_{it} - v_{it}, \qquad u_{i0} = 0,$$

with $v \equiv 0$ in **Model A** (disruption-only closure). The objective is the expected total cost

$$\mathbb E\Big[\sum_{t=1}^T \Big(\sum_i (f_i y_{it} + h_i u_{it} + c_i v_{it}) + \sum_{i,j} d_{ij} x_{ij t}\Big)\Big],$$

subject to $\sum_j x_{ijt} \le C_i u_{it}$ and $\sum_i x_{ijt} = D_j$ (all demand is served). The full formulation is in Section 2 of the paper.

## The benchmark

| Parameter | Values | Count |
|---|---|---|
| Base instance | *CAP44* ($F=16$), *CAP84* ($F=25$), *CAP114* ($F=50$); $B = 50$ customers | 3 |
| Planning horizon $T$ | 12, 24, 36, 52, 100 | 5 |
| Disruption level $p$ | High 30%, Medium 20%, Low 10% | 3 |
| Capacity scheme | Tight, Medium, Loose (total capacity ≈ 1.4–1.5, 2.5–2.6, 4.1–4.3 × total demand) | 3 |
| **Instances** | 3 × 5 × 3 × 3 | **135** |
| Model | A (disruption-only closure), B (voluntary closure) | 2 |
| Operating cost level $\kappa$ | 0.5, 4 | 2 |
| **Configurations** | 135 × 2 × 2 | **540** |

Common to all configurations:

| Data | Definition |
|---|---|
| $f_i$, $D_j$ | from ORLib, constant over $t$ (as in ORLib, one facility per instance has $f_i = 0$) |
| $d_{ij}$ | ORLib allocation cost of customer $j$ to facility $i$, divided by $D_j$ (unit cost) |
| $C_i$ | identical for all facilities of an instance, set by the capacity scheme (see `configurations.csv`) |
| Disruptions | in every stage exactly $k = \lfloor pF \rfloor$ facilities are disrupted, uniformly at random and independently across stages, so $p_i = k/F$ |
| $c_i$ | $0.5\,f_i$ (closing cost, used by Model B) |
| $h_i$ | $\kappa\,c_i / O_i$ with $O_i = [1-(1-p_i)^{T-1}]/p_i$, in both models |
| Unmet demand | not allowed ($\phi = \infty$); every stage is feasible because the facilities that are not disrupted can serve all demand |

$O_i$ is the expected number of stages an idle facility stays open before a disruption closes it or the horizon ends. If $c_i \ge h_i O_i$, voluntary closure has no value and Models A and B have the same optimal value (Proposition 3 of the paper). Thus **$\kappa = 0.5$ gives a test with a known answer** (the optimal values of A and B coincide), while with **$\kappa = 4$ closure may pay**.

Each instance comes with three sets of disruption paths:

| Set | Paths | Use in the paper |
|---|---|---|
| `train20` | 20 | training sample, $S = 20$ |
| `train50` | 50 | training sample, $S = 50$ |
| `test20`  | 20 | out-of-sample evaluation of every policy (no path appears in a training set) |

## Quick start

```bash
git clone https://github.com/bonnkleiford/CAP-MS.git
cd CAP-MS
pip install -r requirements.txt     # numpy, pandas
```

```python
import capms

inst = capms.load('CAP44', T=12, disruption='High', capacity='Medium', model='B', kappa=4)
inst['F'], inst['B'], inst['k'], inst['p'], inst['O']   # 16, 50, 4, 0.25, 3.83
inst['f'], inst['C'], inst['c'], inst['h']             # per facility (length F)
inst['D'], inst['d']                                   # demand (B), unit cost (B x F)
inst['train20'].shape                                  # (20, 12, 16): path, stage, facility; 1 = disrupted

cfg = capms.configurations()                           # all 540 configurations as a DataFrame
```

The loader only reads CSV files, so the data can be used from any language; see the formats below.

## Repository layout and file formats

```
CAP-MS/
├── README.md
├── capms.py                  # loader (numpy, pandas)
├── generate_paths.py         # reproduces every disruption file; generates new paths
├── configurations.csv        # the 540 configurations and their parameters
├── reference_results.csv     # SDDiP and SPA results for every configuration (paper)
├── instances/
│   ├── orlib/                # original ORLib files cap44.txt, cap84.txt, cap114.txt
│   ├── CAP44/
│   │   ├── facilities.csv    # facility, fixed_cost (f_i)
│   │   ├── customers.csv     # customer, demand (D_j)
│   │   ├── unit_cost.csv     # customer, f1..fF: unit cost d_ij
│   │   └── disruptions/      # T{T}_{High|Medium|Low}_{train20|train50|test20}.csv
│   ├── CAP84/
│   └── CAP114/
├── CITATION.cff
├── LICENSE                   # MIT (code)
└── LICENSE-DATA              # CC BY 4.0 (data)
```

**Disruption files.** One row per path, $T \cdot F$ columns named `t1_f1, t1_f2, …, tT_fF`, entries 0/1 (1 = disrupted). Column `t{t}_f{i}` is entry `t*F + i` (0-based) of the row, so `row.reshape(T, F)` gives the stages × facilities matrix. Every stage has exactly $k$ ones.

**`configurations.csv`.** One row per configuration with `config_id`, `instance`, `F`, `B`, `T`, `disruption`, `p_nominal`, `k`, `p` ($=k/F$), `O`, `capacity_scheme`, `capacity` ($C_i$), `capacity_ratio` (total capacity / total demand), `model`, `kappa`, `closing_cost_factor` ($c_i$ = factor · $f_i$), `operating_cost_factor` ($h_i$ = factor · $f_i$) and the paths of the three disruption files.

## Reference results

`reference_results.csv` reports, for every configuration and both training sample sizes ($S$ = 20, 50; 1080 rows), the results of the paper after **600 s of training** (one thread, Gurobi; see the paper for details):

| Column | Meaning |
|---|---|
| `sddip_lb` | SDDiP lower bound of the training problem (sample of $S$ paths) |
| `sddip_cost`, `sddip_se` | mean cost of the SDDiP policy on the 20 `test20` paths, and its standard error |
| `spa_cost`, `spa_se` | the same for the SPA policy |
| `best_cost`, `best_method` | the lower of the two costs and the method that achieved it |
| `gap_best_pct` | (`best_cost` − `sddip_lb`) / `best_cost` × 100 |
| `sddip_iterations`, `spa_evaluations` | SDDiP iterations, SPA policy evaluations within the time limit |
| `sddip_secs`, `spa_secs` | training time (a run may exceed 600 s by up to one iteration) |

`sddip_lb` bounds the optimal value of the *sampled* training problem, not of the true problem, and the costs are estimates on 20 paths. A gap is therefore an estimate, and it can be slightly negative (2 of 1080 rows). With $S = 20$, the mean gap is about 3% on *CAP44*, 8% on *CAP84* and 13% on *CAP114*, so all instances leave room for better methods.

## Generating new disruption paths

```bash
python generate_paths.py --check        # regenerates all 135 released files and confirms they are identical
python generate_paths.py --instance CAP84 --T 36 --level Low --S 200 --seed 1 --out CAP84_T36_Low_S200.csv
```

New paths follow the same rule as the released ones (exactly $k$ disruptions per stage).

## Reporting results

To keep results comparable:

1. train on `train20` or `train50`, and evaluate the final policy on **`test20`**;
2. report the mean cost on `test20` with its standard error, the training time and the hardware;
3. compare against `best_cost` (and `sddip_lb`) in `reference_results.csv` for the same `config_id` and `S_train`.

New results, improved bounds and corrections are welcome as issues or pull requests.


## License

The data (`instances/`, `configurations.csv`, `reference_results.csv`) are licensed under [CC BY 4.0](LICENSE-DATA): you may share and adapt them with attribution. The code (`capms.py`, `generate_paths.py`) is licensed under the [MIT License](LICENSE). The base instances are from the OR-Library (Beasley 1990).

**Contact:** Bonn Kleiford Seranilla (bonnkleiford@gmail.com)
