"""
Loader for the CAP-MS benchmark (multistage stochastic reliable capacitated facility location).

    import capms
    inst = capms.load('CAP44', T=12, disruption='High', capacity='Medium', model='B', kappa=4)
    inst['f'], inst['h'], inst['c'], inst['C']      # per facility, length F
    inst['D']                                       # demand, length B
    inst['d']                                       # unit transport cost, B x F (customer j, facility i)
    R = capms.paths('CAP44', 12, 'High', 'test20')  # disruptions, shape (S, T, F); 1 = disrupted
    capms.configurations()                          # the 540 configurations as a DataFrame

Requires numpy and pandas.
"""
import itertools
import math
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
INSTANCES = ['CAP44', 'CAP84', 'CAP114']
T_ALL = [12, 24, 36, 52, 100]
DISRUPTION = {'High': 0.30, 'Medium': 0.20, 'Low': 0.10}         # nominal p; k = floor(p F) facilities per stage
CAPACITY = {                                                     # C_i, identical for all facilities of an instance
    'CAP44':  {'Tight': 5000, 'Medium': 9000, 'Loose': 15000},
    'CAP84':  {'Tight': 3500, 'Medium': 6000, 'Loose': 9500},
    'CAP114': {'Tight': 1800, 'Medium': 3000, 'Loose': 5000},
}
MODELS = ['A', 'B']             # A: disruption-only closure; B: voluntary closure allowed
KAPPAS = [0.5, 4.0]             # h_i = kappa * c_i / O_i
CLOSING_FACTOR = 0.5            # c_i = 0.5 f_i
PATH_SETS = {'train20': 20, 'train50': 50, 'test20': 20}


def _name(instance):
    name = instance.replace('-', '').upper()
    if name not in INSTANCES:
        raise ValueError('instance must be one of %s' % INSTANCES)
    return name


def base(instance):
    """Static ORLib data: fixed (re-)opening cost f, demand D, unit transport cost d (B x F)."""
    folder = os.path.join(HERE, 'instances', _name(instance))
    f = pd.read_csv(os.path.join(folder, 'facilities.csv')).fixed_cost.to_numpy(float)
    D = pd.read_csv(os.path.join(folder, 'customers.csv')).demand.to_numpy(float)
    d = pd.read_csv(os.path.join(folder, 'unit_cost.csv')).drop(columns='customer').to_numpy(float)
    return f, D, d


def disrupted_per_stage(F, disruption):
    return math.floor(F * round(DISRUPTION[disruption] * 100) / 100)


def expected_open_stages(p, T):
    """O = [1 - (1-p)^(T-1)] / p: expected number of stages an idle facility stays open (T-1 if p = 0)."""
    return (1 - (1 - p) ** (T - 1)) / p if p > 0 else T - 1


def paths(instance, T, disruption, which='test20'):
    """Disruption paths as an int array (S, T, F); which in {'train20', 'train50', 'test20'}."""
    name = _name(instance)
    fn = os.path.join(HERE, 'instances', name, 'disruptions', 'T%d_%s_%s.csv' % (T, disruption, which))
    R = pd.read_csv(fn).to_numpy(np.int8)
    return R.reshape(R.shape[0], T, -1)


def load(instance, T, disruption, capacity, model='A', kappa=0.5, with_paths=True):
    """All data of one configuration as a dict."""
    name = _name(instance)
    f, D, d = base(name)
    F, B = len(f), len(D)
    k = disrupted_per_stage(F, disruption)
    p = k / F
    O = expected_open_stages(p, T)
    c = CLOSING_FACTOR * f
    out = dict(instance=name, F=F, B=B, T=T, disruption=disruption, capacity_scheme=capacity, model=model,
               kappa=kappa, k=k, p=p, O=O, f=f, D=D, d=d, C=np.full(F, float(CAPACITY[name][capacity])),
               c=c, h=kappa * c / O, phi=math.inf, closure_allowed=(model == 'B'))
    if with_paths:
        for which in PATH_SETS:
            out[which] = paths(name, T, disruption, which)
    return out


def configurations():
    """The 540 configurations: 3 instances x 5 horizons x 3 disruption levels x 3 capacity schemes x 2 models x 2 kappas."""
    rows = []
    for name, T, lv, cap, model, kappa in itertools.product(INSTANCES, T_ALL, DISRUPTION, ['Tight', 'Medium', 'Loose'],
                                                            MODELS, KAPPAS):
        f, D, _ = base(name)
        F = len(f); k = disrupted_per_stage(F, lv); p = k / F; O = expected_open_stages(p, T)
        rows.append(dict(
            config_id='%s_T%d_%s_%s_%s_kappa%g' % (name, T, lv, cap, model, kappa),
            instance=name, F=F, B=len(D), T=T, disruption=lv, p_nominal=DISRUPTION[lv], k=k, p=round(p, 6),
            O=round(O, 6), capacity_scheme=cap, capacity=CAPACITY[name][cap],
            capacity_ratio=round(F * CAPACITY[name][cap] / D.sum(), 3), model=model, kappa=kappa,
            closing_cost_factor=CLOSING_FACTOR, operating_cost_factor=round(kappa * CLOSING_FACTOR / O, 8),
            train20='instances/%s/disruptions/T%d_%s_train20.csv' % (name, T, lv),
            train50='instances/%s/disruptions/T%d_%s_train50.csv' % (name, T, lv),
            test20='instances/%s/disruptions/T%d_%s_test20.csv' % (name, T, lv)))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    cfg = configurations()
    cfg.to_csv(os.path.join(HERE, 'configurations.csv'), index=False)
    print('wrote configurations.csv:', len(cfg), 'configurations')
