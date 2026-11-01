"""Round-level statistics of the 13 runs (Sections 5.1-5.3, Table 2 and Supplementary Table S3 of the paper).

Reads results/logs/*.csv and writes:
  results/summary/final_runs.csv   one row per run
  results/summary/penalties.json   per-seed accuracy cost of E=1 vs E=3 under both budgets
and prints the tests reported in the paper (Welch, paired t-tests, two-way ANOVA).
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

from runs import BYTES_PER_ROUND, LOCAL_EPOCHS, N_TEST, RUNS, SEEDS, SUMMARY, log_file

MB_PER_ROUND = BYTES_PER_ROUND / 2**20
D = {k: pd.read_csv(log_file(*k)).set_index("round") for k in RUNS}


def acc(k, r):
    return D[k].accuracy[r] * 100


rows = []
for k, d in D.items():
    a = d.accuracy * 100
    p = a[30] / 100
    ci = 1.96 * np.sqrt(p * (1 - p) / N_TEST) * 100          # normal approximation
    first80 = next((int(r) for r in a.index if a[r] >= 80), None)
    stay80 = next((int(r) for r in a.index if (a.loc[r:] >= 80).all()), None)
    first90 = next((int(r) for r in a.index if a[r] >= 90), None)
    plateau = max(int(r) for r in a.index if a[r] < 20)       # last round below 20 %
    rows.append(dict(
        sc=k[0], seed=k[1], r10=a[10], r30=a[30], ci_lo=a[30] - ci, ci_hi=a[30] + ci,
        last5=a.loc[26:30].mean(), first80=first80, stay80=stay80, first90=first90,
        plateau_rounds=plateau, plateau_epochs=plateau * LOCAL_EPOCHS[k[0]],
        comm80=(first80 * MB_PER_ROUND if first80 else None),
        tt_client_med=d.total_train_time_sec.loc[2:].median() / 5,
        tt_client_mean=d.total_train_time_sec.loc[1:].mean() / 5,
        ram_min=d.parent_ram_gb.min(), ram_max=d.parent_ram_gb.max()))
R = pd.DataFrame(rows)
SUMMARY.mkdir(parents=True, exist_ok=True)
R.to_csv(SUMMARY / "final_runs.csv", index=False)
pd.set_option("display.width", 250)
print(R.round(2).to_string())

print("\n== scenario means at round 30 ==")
for sc, g in R.groupby("sc"):
    sd = g.r30.std(ddof=1) if len(g) > 1 else float("nan")
    print(f"{sc}  n={len(g)}  mean {g.r30.mean():.2f}  sd {sd:.2f}  range {g.r30.min():.2f}-{g.r30.max():.2f}")

S1 = acc(("S1", 42), 30)
print("\n== gap to S1 at round 30 (pp) ==")
for sc in ("S2", "S3", "S4", "S5"):
    v = [S1 - acc((sc, s), 30) for s in SEEDS]
    print(sc, [round(float(x), 2) for x in v], f"mean {np.mean(v):.2f}")

s2 = [acc(("S2", s), 30) for s in SEEDS]
s3 = [acc(("S3", s), 30) for s in SEEDS]
t = stats.ttest_ind(s2, s3, equal_var=False)
print(f"\nWelch S2 vs S3 (round 30): diff {np.mean(s2) - np.mean(s3):.2f} pp, t={t.statistic:.2f}, p={t.pvalue:.4f}")

# accuracy cost of E=1 relative to E=3, seed-matched, under two budgets
BUDGETS = {"comm_a01": ("S3", "S4", 30, 30), "comm_a05": ("S2", "S5", 30, 30),   # equal communication
           "comp_a01": ("S3", "S4", 10, 30), "comp_a05": ("S2", "S5", 10, 30)}   # equal local computation
res = {}
print()
for name, (hi, lo, rh, rl) in BUDGETS.items():
    v = [acc((hi, s), rh) - acc((lo, s), rl) for s in SEEDS]
    t = stats.ttest_1samp(v, 0)
    res[name] = v
    print(f"{name}: E3-E1 per seed {[round(float(x), 2) for x in v]}  mean {np.mean(v):.2f}  "
          f"sd {np.std(v, ddof=1):.2f}  paired t={t.statistic:.2f}  p={t.pvalue:.3f}")

# two-way ANOVA (alpha x local epochs) on round-30 accuracy, 3 seeds per cell
y, a01, e1 = [], [], []
for sc, (al, ep) in {"S2": (0.5, 3), "S3": (0.1, 3), "S4": (0.1, 1), "S5": (0.5, 1)}.items():
    for s in SEEDS:
        y.append(acc((sc, s), 30)); a01.append(al == 0.1); e1.append(ep == 1)
y, a01, e1 = np.array(y), np.array(a01), np.array(e1)
X = np.column_stack([np.ones(12), a01, e1, a01 & e1]).astype(float)
beta = np.linalg.lstsq(X, y, rcond=None)[0]
resid = y - X @ beta
cov = (resid @ resid / 8) * np.linalg.inv(X.T @ X)
tI = beta[3] / np.sqrt(cov[3, 3])
print(f"\nTwo-way ANOVA: interaction {beta[3]:.2f} pp, t={tI:.2f}, p={2 * stats.t.sf(abs(tI), 8):.3f}")

json.dump({k: [float(x) for x in v] for k, v in res.items()}, open(SUMMARY / "penalties.json", "w"), indent=1)
