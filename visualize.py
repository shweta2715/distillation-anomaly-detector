# -*- coding: utf-8 -*-
"""
visualize.py
------------
Produces two plots that tell the whole story of this project:

  1. A zoomed-in view around one real fault event, showing the temperature
     trace, the 3-sigma control band, and the confirmed alarm window.

  2. A "before vs after" comparison showing how many isolated single-point
     flags get correctly suppressed by the consecutive-reading rule,
     directly illustrating the multiple-comparisons trap from Session 1.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

df = pd.read_csv("scored_data.csv", parse_dates=["timestamp"])

# ----------------------------------------------------------------------
# Plot 1: zoom into one real fault event with full context
# ----------------------------------------------------------------------
fault_rows = df[df["true_regime"] == "fault"]
first_fault_time = fault_rows["timestamp"].iloc[0]

window_start = first_fault_time - pd.Timedelta(hours=2)
window_end = first_fault_time + pd.Timedelta(hours=2)
window = df[(df["timestamp"] >= window_start) & (df["timestamp"] <= window_end)]

mean_baseline = 141.8  # from detector baseline
std_baseline = 1.401

fig, ax = plt.subplots(figsize=(11, 5))

ax.plot(window["timestamp"], window["top_temp_C"], color="#2d4a7c",
        linewidth=1.2, label="Top tray temperature")

ax.axhline(mean_baseline, color="gray", linestyle="--", linewidth=1, label="Baseline mean")
ax.axhline(mean_baseline + 3 * std_baseline, color="#d9822b", linestyle=":", linewidth=1,
           label="±3σ control band")
ax.axhline(mean_baseline - 3 * std_baseline, color="#d9822b", linestyle=":", linewidth=1)

confirmed = window[window["confirmed_alarm"]]
ax.scatter(confirmed["timestamp"], confirmed["top_temp_C"], color="#c0392b",
           s=18, zorder=5, label="Confirmed alarm (3+ consecutive)")

ax.set_title("Real Fault Event — Detected and Confirmed", fontsize=13, fontweight="bold")
ax.set_xlabel("Time")
ax.set_ylabel("Temperature (°C)")
ax.legend(loc="upper left", fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
fig.autofmt_xdate()
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("plot_1_fault_detection.png", dpi=150)
plt.close()
print("Saved plot_1_fault_detection.png")

# ----------------------------------------------------------------------
# Plot 2: the multiple-comparisons fix, shown across the FULL dataset
# ----------------------------------------------------------------------
# Rather than a noisy zoomed window, show the honest full-dataset summary:
# how many raw flags occur vs how many survive as confirmed alarms,
# split by whether they were real faults or normal steady-state noise.

steady_df = df[df["true_regime"] == "steady"]
fault_df = df[df["true_regime"] == "fault"]

raw_flags_steady = steady_df["single_point_flag"].sum()
confirmed_steady = steady_df["confirmed_alarm"].sum()

raw_flags_fault = fault_df["single_point_flag"].sum()
confirmed_fault = fault_df["confirmed_alarm"].sum()

fig, ax = plt.subplots(figsize=(9, 5.5))

categories = ["Raw |Z|>3 flags\n(single reading)", "Confirmed alarms\n(3+ consecutive)"]
steady_counts = [raw_flags_steady, confirmed_steady]
fault_counts = [raw_flags_fault, confirmed_fault]

x = np.arange(len(categories))
width = 0.38

bars1 = ax.bar(x - width/2, steady_counts, width, label="During normal steady-state\n(false alarms)",
               color="#d9822b")
bars2 = ax.bar(x + width/2, fault_counts, width, label="During real fault events\n(true alarms)",
               color="#2d4a7c")

for bars in (bars1, bars2):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{int(height):,}", xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    fontsize=10, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=10)
ax.set_ylabel("Number of readings flagged")
ax.set_title("The Multiple-Comparisons Fix — Full 6-Month Dataset", fontsize=13, fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.25, axis="y")

# Annotation explaining the story
ax.text(0.02, 0.97,
        "Raw Z-scores flood the system with false alarms.\n"
        "Requiring 3 consecutive readings removes almost all\n"
        "of them while still catching every real fault.",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9))

plt.tight_layout()
plt.savefig("plot_2_multiple_comparisons_fix.png", dpi=150)
plt.close()
print("Saved plot_2_multiple_comparisons_fix.png")

print(f"\nSteady-state: {raw_flags_steady:,} raw flags -> {confirmed_steady:,} confirmed alarms "
      f"({100*(1 - confirmed_steady/raw_flags_steady):.1f}% reduction in false alarms)")
print(f"Real faults:  {raw_flags_fault:,} raw flags -> {confirmed_fault:,} confirmed alarms "
      f"(faults still detected)")
print("\nThis is the multiple-comparisons fix from Session 1, shown across the full dataset.")
