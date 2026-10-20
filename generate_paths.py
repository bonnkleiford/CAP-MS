"""
Disruption paths of CAP-MS.

In every stage, exactly k = floor(p * F) facilities are disrupted, drawn uniformly at random and independently
across stages and paths. A path is stored as one row of T*F zeros and ones (column t{t}_f{i} = 1 if facility i is
disrupted in stage t).

    python generate_paths.py --check                      # regenerate all released files and compare them
    python generate_paths.py --instance CAP44 --T 24 --level High --S 100 --seed 1 --out my_paths.csv

The released sets were generated as follows (seeds are fixed, so --check reproduces every file exactly):
    train20  random.seed(42 + T), each stage a shuffled list of k ones and F-k zeros   (Python's random module)
    train50  numpy default_rng(50000 + 1000*l + T), each stage k facilities chosen without replacement
    test20   numpy default_rng(70000 + 1000*l + 10*T + F), same procedure as train50
with l = 0, 1, 2 for High, Medium, Low.
"""
import argparse
import math
import os
import random

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
F_OF = {'CAP44': 16, 'CAP84': 25, 'CAP114': 50}
T_ALL = [12, 24, 36, 52, 100]
LEVELS = {'High': 0.30, 'Medium': 0.20, 'Low': 0.10}
SETS = {'train20': 20, 'train50': 50, 'test20': 20}


def n_disrupted(F, level):
    return math.floor(F * round(LEVELS[level] * 100) / 100)


def paths_shuffle(S, T, F, k, seed):
    """train20: Python random module, one shuffled stage at a time."""
    random.seed(seed)
    out = np.zeros((S, T, F), dtype=np.int8)
    for s in range(S):
        for t in range(T):
            row = [1] * k + [0] * (F - k)
            random.shuffle(row)
            out[s, t] = row
    return out


def paths_choice(S, T, F, k, seed):
    """train50, test20 and new sets: numpy Generator, k facilities without replacement per stage."""
    rng = np.random.default_rng(seed)
    out = np.zeros((S, T, F), dtype=np.int8)
    for s in range(S):
        for t in range(T):
            out[s, t, rng.choice(F, size=k, replace=False)] = 1
    return out


def released(instance, T, level, which):
    F = F_OF[instance]; k = n_disrupted(F, level); l = list(LEVELS).index(level)
    if which == 'train20':
        return paths_shuffle(20, T, F, k, 42 + T)
    if which == 'train50':
        return paths_choice(50, T, F, k, 50_000 + 1000 * l + T)
    return paths_choice(20, T, F, k, 70_000 + 1000 * l + 10 * T + F)


def save(R, path):
    S, T, F = R.shape
    cols = ['t%d_f%d' % (t, i) for t in range(1, T + 1) for i in range(1, F + 1)]
    pd.DataFrame(R.reshape(S, T * F), columns=cols).to_csv(path, index=False)


def check():
    bad = 0
    for inst in F_OF:
        for T in T_ALL:
            for lv in LEVELS:
                for which, S in SETS.items():
                    fn = os.path.join(HERE, 'instances', inst, 'disruptions', 'T%d_%s_%s.csv' % (T, lv, which))
                    ok = (pd.read_csv(fn).values.reshape(S, T, F_OF[inst]) == released(inst, T, lv, which)).all()
                    bad += not ok
                    if not ok:
                        print('MISMATCH', fn)
    print('all %d files reproduced' % (3 * 5 * 3 * 3) if bad == 0 else '%d files differ' % bad)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--instance', choices=list(F_OF)); ap.add_argument('--T', type=int)
    ap.add_argument('--level', choices=list(LEVELS)); ap.add_argument('--S', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--out')
    a = ap.parse_args()
    if a.check:
        check()
    else:
        F = F_OF[a.instance]
        save(paths_choice(a.S, a.T, F, n_disrupted(F, a.level), a.seed), a.out)
        print('wrote', a.out)
