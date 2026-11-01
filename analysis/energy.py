"""Estimated energy cost of client training (Section 5.4 of the paper).

The energy was not metered. It is estimated from the logged client training time and the nominal maximum
board power of the GPU (NVIDIA A100 40 GB SXM: 400 W, NVIDIA data sheet), so it is an upper bound on the
energy drawn by the GPU board while the clients were training, not a measurement of the whole system.
Only the nine runs executed on the A100 are used; the four replicate runs on an L4 are excluded.
Transmission energy uses 0.06 kWh/GB (Aslan et al., 2018, fixed-line networks, 2015 estimate).

Reads results/logs/*.csv and writes results/summary/energy.csv.
"""
import pandas as pd

from runs import BYTES_PER_ROUND, LOCAL_EPOCHS, RUNS, SUMMARY, log_file

A100_RUNS = [("S1", 42), ("S2", 42), ("S3", 42), ("S3", 43), ("S3", 44), ("S4", 42), ("S4", 43), ("S5", 42), ("S5", 43)]
P_GPU_W = 400.0                 # A100 40 GB SXM, maximum thermal design power
KWH_PER_GB_NETWORK = 0.06       # Aslan et al. (2018), fixed-line Internet, 2015


def kwh(seconds):
    return seconds * P_GPU_W / 3.6e6


rows = []
for k in A100_RUNS:
    d = pd.read_csv(log_file(*k)).set_index("round")
    t = d.total_train_time_sec.loc[1:30]                 # summed over the five clients, per round
    acc = d.accuracy * 100
    first80 = next((int(r) for r in acc.index if r >= 1 and acc[r] >= 80), None)
    rows.append(dict(
        sc=k[0], seed=k[1], local_epochs=LOCAL_EPOCHS[k[0]],
        client_gpu_hours_30=t.sum() / 3600,
        kwh_30_rounds=kwh(t.sum()),
        kwh_10_rounds=kwh(t.loc[1:10].sum()),
        kwh_per_round_median=kwh(t.loc[2:30].median()),
        first80=first80,
        kwh_to_80=kwh(t.loc[1:first80].sum()) if first80 else None,
        kwh_evaluation_30=kwh(d.eval_time_sec.loc[0:30].sum()),
        kwh_uploads_30=30 * BYTES_PER_ROUND / 1e9 * KWH_PER_GB_NETWORK,
    ))
E = pd.DataFrame(rows)
SUMMARY.mkdir(parents=True, exist_ok=True)
E.to_csv(SUMMARY / "energy.csv", index=False)
pd.set_option("display.width", 250)
print(E.round(3).to_string())

g = E.set_index(["sc", "seed"])
print("\nEqual local computation (30 local epochs): E=1 after 30 rounds vs E=3 after 10 rounds")
for lo, hi, s in (("S4", "S3", 42), ("S4", "S3", 43), ("S5", "S2", 42)):
    a, b = g.loc[(lo, s), "kwh_30_rounds"], g.loc[(hi, s), "kwh_10_rounds"]
    print(f"  {lo} vs {hi}, seed {s}: {a:.2f} kWh vs {b:.2f} kWh  (x{a / b:.2f})")
iid = g.loc[("S1", 42), "kwh_to_80"]
print("\nEnergy to reach 80% relative to IID:")
for k, r in g.iterrows():
    if r.kwh_to_80 == r.kwh_to_80 and r.kwh_to_80 is not None:
        print(f"  {k}: {r.kwh_to_80:.2f} kWh (x{r.kwh_to_80 / iid:.2f})")
