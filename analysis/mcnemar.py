"""Exact McNemar tests between pairs of final models, with Holm correction (Sections 5.1-5.2).

Reads the saved test-set predictions and writes results/summary/mcnemar.csv.
"""
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from runs import PREDS, SUMMARY, pred_file

y = np.load(PREDS / "test_labels.npz")["y"]
PAIRS = [(("S3", 42), ("S3", 43)), (("S3", 42), ("S3", 44)), (("S3", 43), ("S3", 44)),
         (("S3", 42), ("S4", 42)), (("S2", 42), ("S5", 42)),
         (("S4", 42), ("S4", 43)), (("S5", 42), ("S5", 43))]

rows = []
for a, b in PAIRS:
    ca, cb = np.load(pred_file(*a)) == y, np.load(pred_file(*b)) == y
    only_a, only_b = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue
    rows.append(dict(model_a=f"{a[0]} seed {a[1]}", model_b=f"{b[0]} seed {b[1]}",
                     only_a_correct=only_a, only_b_correct=only_b, p=p))
R = pd.DataFrame(rows)

# Holm step-down adjustment over the seven tests
order = np.argsort(R.p.values)
adj, running = np.empty(len(R)), 0.0
for i, j in enumerate(order):
    running = max(running, min(1.0, (len(R) - i) * R.p.values[j]))
    adj[j] = running
R["p_holm"] = adj
SUMMARY.mkdir(parents=True, exist_ok=True)
R.to_csv(SUMMARY / "mcnemar.csv", index=False)
print(R.to_string(formatters={"p": "{:.3g}".format, "p_holm": "{:.3g}".format}))
