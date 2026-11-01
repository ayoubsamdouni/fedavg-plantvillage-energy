"""Per-class results of the 13 final models (Section 5.4, Table 2 and Supplementary Table S4 of the paper).

Reads the saved test-set predictions (results/predictions) and writes
results/summary/perclass.json with, for every run: accuracy, macro-F1,
bootstrap 95% CI of accuracy (10,000 resamples), per-class F1 and the classes
with F1 = 0.
"""
import json
from collections import Counter

import numpy as np
from sklearn.metrics import f1_score

from runs import PREDS, RUNS, SUMMARY, pred_file

d = np.load(PREDS / "test_labels.npz")
y, classes = d["y"], [str(c) for c in d["classes"]]
rng = np.random.default_rng(0)
B = rng.integers(0, len(y), size=(10_000, len(y)))

out = {}
for (sc, seed) in RUNS:
    p = np.load(pred_file(sc, seed))
    correct = p == y
    per = f1_score(y, p, average=None, labels=range(len(classes)), zero_division=0)
    boot = correct[B].mean(1) * 100
    out[f"{sc}_{seed}"] = dict(
        acc=correct.mean() * 100,
        macro_f1=f1_score(y, p, average="macro") * 100,
        ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        collapsed=[classes[i] for i in range(len(classes)) if per[i] == 0],
        per_class={classes[i]: float(per[i]) for i in range(len(classes))})

support = {c: int((y == i).sum()) for i, c in enumerate(classes)}
SUMMARY.mkdir(parents=True, exist_ok=True)
json.dump(dict(runs=out, support=support), open(SUMMARY / "perclass.json", "w"), indent=1)

for k, v in sorted(out.items()):
    print(f"{k:6s} acc {v['acc']:.2f}  macro-F1 {v['macro_f1']:.2f}  gap {v['acc'] - v['macro_f1']:.1f}  "
          f"CI [{v['ci'][0]:.2f}, {v['ci'][1]:.2f}]  collapsed {v['collapsed']}")
print("\nruns with F1 = 0, per class:", dict(Counter(c for v in out.values() for c in v["collapsed"])))
