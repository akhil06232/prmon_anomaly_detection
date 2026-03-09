"""
prmon Process Resource Monitoring — Anomaly Detection Pipeline
==============================================================

This script:
1. Simulates prmon time-series output (prmon is Linux-only; we reproduce its CSV
   schema faithfully, using synthetic "burner test" patterns for memory and I/O).
2. Injects artificial anomalies (elevated memory, burst I/O, extra threads).
3. Detects anomalies with two methods:
   - Z-score (statistical baseline)
   - Isolation Forest (unsupervised ML)
4. Evaluates detection quality (precision, recall, F1) and produces plots.

AI disclosure: GitHub Copilot (Claude Opus 4.6) was used to generate code,
design the simulation + detection pipeline, and draft the report narrative.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUT_DIR = Path("prmon_output")
PLOT_DIR = Path("plots")
SEED = 42
N_TIMESTEPS = 600          # 10 minutes at 1-second intervals
ANOMALY_FRACTION = 0.08    # ~8 % anomalous points

rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# 1. Simulate prmon burner-test data
# ---------------------------------------------------------------------------
# prmon records columns: wtime, utime, stime, rss, pss, vmem, rchar, wchar,
#   nthreads, nprocs, ...
# We focus on PSS (proportional set size), RSS, rchar (read bytes),
# wchar (write bytes), and nthreads.

def generate_normal_run(n: int, run_id: int) -> pd.DataFrame:
    """Produce a single normal prmon-like time-series."""
    t = np.arange(n, dtype=float)

    # Memory: gradual ramp-up then plateau (mimics a burner-memory test)
    ramp = np.minimum(t / 120, 1.0)  # ramp over ~2 min
    base_pss = 200_000 + 300_000 * ramp  # 200–500 MB in kB
    pss = base_pss + rng.normal(0, 5000, n)

    rss = pss * (1.0 + rng.uniform(0.02, 0.06, n))  # RSS slightly larger
    vmem = rss * rng.uniform(1.8, 2.2, n)

    # I/O: steady read/write rate with small jitter
    rchar = np.cumsum(rng.normal(50_000, 5_000, n).clip(0))
    wchar = np.cumsum(rng.normal(30_000, 3_000, n).clip(0))

    # CPU time: linear growth
    utime = np.cumsum(rng.uniform(0.8, 1.2, n))
    stime = np.cumsum(rng.uniform(0.05, 0.15, n))

    nthreads = np.full(n, 4) + rng.integers(-1, 2, n)  # 3-5 threads
    nprocs = np.ones(n, dtype=int)

    return pd.DataFrame({
        "wtime": t,
        "utime": utime,
        "stime": stime,
        "rss": rss.astype(int),
        "pss": pss.astype(int),
        "vmem": vmem.astype(int),
        "rchar": rchar.astype(int),
        "wchar": wchar.astype(int),
        "nthreads": nthreads.clip(1).astype(int),
        "nprocs": nprocs,
        "run_id": run_id,
    })


def inject_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inject three kinds of anomalies at random locations:
      A) Memory spike — PSS/RSS jump 2–4× for short bursts
      B) I/O burst    — rchar/wchar rate jumps sharply
      C) Thread surge — nthreads jumps to 15–30
    Returns a copy with an 'anomaly' label column.
    """
    df = df.copy()
    n = len(df)
    n_anom = int(n * ANOMALY_FRACTION)

    # Choose anomaly indices (contiguous blocks of 3-8 to mimic real bursts)
    anomaly_mask = np.zeros(n, dtype=bool)
    placed = 0
    while placed < n_anom:
        start = rng.integers(10, n - 10)
        length = rng.integers(3, 9)
        end = min(start + length, n)
        anomaly_mask[start:end] = True
        placed = int(anomaly_mask.sum())

    anom_idx = np.where(anomaly_mask)[0]

    # Assign anomaly types roughly evenly
    types = rng.choice(["mem", "io", "thread"], size=len(anom_idx))

    for idx, atype in zip(anom_idx, types):
        if atype == "mem":
            factor = rng.uniform(2.0, 4.0)
            df.loc[idx, "pss"] = int(df.loc[idx, "pss"] * factor)
            df.loc[idx, "rss"] = int(df.loc[idx, "rss"] * factor)
        elif atype == "io":
            # Sudden jump in cumulative I/O
            jump = rng.integers(500_000, 2_000_000)
            df.loc[idx:, "rchar"] += jump
            df.loc[idx:, "wchar"] += int(jump * 0.6)
        else:  # thread
            df.loc[idx, "nthreads"] = rng.integers(15, 31)

    df["anomaly"] = anomaly_mask.astype(int)
    return df


# ---------------------------------------------------------------------------
# 2. Generate dataset  (multiple "runs" concatenated)
# ---------------------------------------------------------------------------
print("Generating synthetic prmon data …")
N_RUNS = 5
frames = []
for rid in range(N_RUNS):
    normal = generate_normal_run(N_TIMESTEPS, rid)
    with_anom = inject_anomalies(normal)
    frames.append(with_anom)

data = pd.concat(frames, ignore_index=True)

OUT_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

# Save per-run CSVs (prmon convention: one file per monitored job)
for rid in range(N_RUNS):
    subset = data[data["run_id"] == rid].drop(columns=["anomaly", "run_id"])
    subset.to_csv(OUT_DIR / f"prmon_run_{rid:03d}.csv", index=False, sep="\t")
    print(f"  Saved prmon_run_{rid:03d}.csv  ({len(subset)} rows)")

# Also save the combined labelled dataset for analysis
data.to_csv(OUT_DIR / "combined_labelled.csv", index=False)
print(f"Combined dataset: {len(data)} rows, "
      f"{data['anomaly'].sum()} anomalous ({data['anomaly'].mean()*100:.1f}%)")

# ---------------------------------------------------------------------------
# 3. Feature engineering for detection
# ---------------------------------------------------------------------------
# We compute per-timestep features amenable to anomaly detection:
#   - pss, rss (instantaneous)
#   - d_rchar, d_wchar (I/O rate = diff of cumulative counters)
#   - nthreads

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    feat["pss"] = df["pss"]
    feat["rss"] = df["rss"]
    feat["nthreads"] = df["nthreads"]

    # I/O rate (per-second diff within each run)
    feat["d_rchar"] = df.groupby("run_id")["rchar"].diff().fillna(0)
    feat["d_wchar"] = df.groupby("run_id")["wchar"].diff().fillna(0)

    return feat


features = build_features(data)
y_true = data["anomaly"].values

# ---------------------------------------------------------------------------
# 4a. Z-score detector (statistical baseline)
# ---------------------------------------------------------------------------
print("\n--- Z-score detector ---")

def zscore_detect(features: pd.DataFrame, threshold: float = 3.0) -> np.ndarray:
    """Flag a point as anomalous if ANY feature |z| > threshold."""
    means = features.mean()
    stds = features.std().replace(0, 1)
    z = ((features - means) / stds).abs()
    return (z.max(axis=1) > threshold).astype(int).values

y_pred_z = zscore_detect(features, threshold=2.5)
p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred_z, pos_label=1, average="binary", zero_division=0)
print(f"  Precision: {p:.3f}   Recall: {r:.3f}   F1: {f1:.3f}")
cm_z = confusion_matrix(y_true, y_pred_z)
print(f"  Confusion matrix:\n{cm_z}")

# ---------------------------------------------------------------------------
# 4b. Isolation Forest detector
# ---------------------------------------------------------------------------
print("\n--- Isolation Forest detector ---")

iso = IsolationForest(
    n_estimators=200,
    contamination=ANOMALY_FRACTION,
    random_state=SEED,
    n_jobs=-1,
)
iso.fit(features)
iso_labels = iso.predict(features)          # -1 = anomaly, 1 = normal
y_pred_iso = (iso_labels == -1).astype(int)

p2, r2, f12, _ = precision_recall_fscore_support(y_true, y_pred_iso, pos_label=1, average="binary", zero_division=0)
print(f"  Precision: {p2:.3f}   Recall: {r2:.3f}   F1: {f12:.3f}")
cm_iso = confusion_matrix(y_true, y_pred_iso)
print(f"  Confusion matrix:\n{cm_iso}")

# Anomaly scores (lower = more anomalous)
iso_scores = iso.decision_function(features)

# ---------------------------------------------------------------------------
# 5. Visualisation
# ---------------------------------------------------------------------------
print("\nGenerating plots …")

# --- Plot 1: PSS time-series with anomaly flags (one run) ---
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

run0 = data[data["run_id"] == 0].copy()
run0_feat = features.loc[run0.index]
run0_pred_z = y_pred_z[run0.index]
run0_pred_iso = y_pred_iso[run0.index]
run0_true = y_true[run0.index]

t = run0["wtime"].values

# Panel A: raw PSS
ax = axes[0]
ax.plot(t, run0["pss"] / 1e3, linewidth=0.8, label="PSS (MB)")
anom_t = t[run0_true == 1]
anom_pss = run0["pss"].values[run0_true == 1] / 1e3
ax.scatter(anom_t, anom_pss, c="red", s=18, zorder=5, label="True anomaly")
ax.set_ylabel("PSS (MB)")
ax.set_title("Run 0 — PSS with injected anomalies")
ax.legend(loc="upper left")

# Panel B: Z-score detections
ax = axes[1]
ax.plot(t, run0["pss"] / 1e3, linewidth=0.8, alpha=0.5)
det_z = t[run0_pred_z == 1]
det_z_pss = run0["pss"].values[run0_pred_z == 1] / 1e3
ax.scatter(det_z, det_z_pss, c="orange", s=18, marker="^", zorder=5, label="Z-score flag")
ax.set_ylabel("PSS (MB)")
ax.set_title("Z-score detections")
ax.legend(loc="upper left")

# Panel C: Isolation Forest detections
ax = axes[2]
ax.plot(t, run0["pss"] / 1e3, linewidth=0.8, alpha=0.5)
det_iso = t[run0_pred_iso == 1]
det_iso_pss = run0["pss"].values[run0_pred_iso == 1] / 1e3
ax.scatter(det_iso, det_iso_pss, c="green", s=18, marker="s", zorder=5, label="Isolation Forest flag")
ax.set_xlabel("Time (s)")
ax.set_ylabel("PSS (MB)")
ax.set_title("Isolation Forest detections")
ax.legend(loc="upper left")

plt.subplots_adjust(hspace=0.3, left=0.08, right=0.97, top=0.95, bottom=0.07)
plt.savefig(PLOT_DIR / "pss_anomalies_run0.png", dpi=150)
plt.close()

# --- Plot 2: Multi-metric overview (all runs) ---
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

metrics = [("pss", "PSS (kB)"), ("nthreads", "Threads"),
           ("d_rchar", "Read rate (B/s)"), ("d_wchar", "Write rate (B/s)")]

for ax, (col, label) in zip(axes.flat, metrics):
    vals = features[col].values
    ax.plot(vals, linewidth=0.4, alpha=0.7)
    anom_idx = np.where(y_true == 1)[0]
    ax.scatter(anom_idx, vals[anom_idx], c="red", s=6, zorder=5, alpha=0.6, label="anomaly")
    ax.set_ylabel(label)
    ax.set_xlabel("Global timestep index")
    ax.legend(fontsize=8)

fig.suptitle("All runs — feature time-series with true anomalies", fontsize=13)
plt.subplots_adjust(hspace=0.35, wspace=0.3, top=0.92)
plt.savefig(PLOT_DIR / "multi_metric_overview.png", dpi=150)
plt.close()

# --- Plot 3: Isolation Forest anomaly score distribution ---
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(iso_scores[y_true == 0], bins=60, alpha=0.6, label="Normal", density=True)
ax.hist(iso_scores[y_true == 1], bins=60, alpha=0.6, label="Anomalous", density=True)
ax.axvline(0, color="k", linestyle="--", linewidth=0.8, label="Decision boundary")
ax.set_xlabel("Isolation Forest anomaly score")
ax.set_ylabel("Density")
ax.set_title("Score distribution: normal vs. anomalous points")
ax.legend()
plt.subplots_adjust(left=0.1, right=0.97, top=0.92, bottom=0.15)
plt.savefig(PLOT_DIR / "isoforest_score_distribution.png", dpi=150)
plt.close()

# --- Plot 4: Precision / Recall / F1 comparison bar chart ---
fig, ax = plt.subplots(figsize=(7, 4))
methods = ["Z-score", "Isolation Forest"]
prec_vals = [p, p2]
rec_vals = [r, r2]
f1_vals = [f1, f12]
x = np.arange(len(methods))
w = 0.25
ax.bar(x - w, prec_vals, w, label="Precision")
ax.bar(x, rec_vals, w, label="Recall")
ax.bar(x + w, f1_vals, w, label="F1")
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Score")
ax.set_title("Detection quality comparison")
ax.legend()
plt.subplots_adjust(left=0.1, right=0.97, top=0.90, bottom=0.12)
plt.savefig(PLOT_DIR / "detection_comparison.png", dpi=150)
plt.close()

print(f"Plots saved in {PLOT_DIR}/")

# ---------------------------------------------------------------------------
# 6. Summary table
# ---------------------------------------------------------------------------
summary = pd.DataFrame({
    "Method": methods,
    "Precision": prec_vals,
    "Recall": rec_vals,
    "F1": f1_vals,
})
print("\n" + summary.to_string(index=False))

# Clean up temp scripts
for f in ["inspect_data.py", "check_env.py", "chk2.py"]:
    try:
        os.remove(f)
    except FileNotFoundError:
        pass

print("\nDone.")
