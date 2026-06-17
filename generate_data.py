# -*- coding: utf-8 -*-
"""
generate_data.py
-----------------
Simulates 6 months of top-tray temperature data for a distillation column,
sampled every minute. Includes:
  - Steady-state normal operation (with realistic Gaussian noise)
  - Startup and shutdown periods (should be excluded from baseline stats)
  - A handful of injected real anomalies (process faults)
  - Sensor noise spikes (single-point glitches, NOT real faults)

This mirrors the real-world messiness discussed in Session 1:
you cannot just throw all the data into mean/std -- you must filter
to steady-state first.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
MINUTES_PER_DAY = 1440
DAYS = 180                     # 6 months
N = MINUTES_PER_DAY * DAYS

NORMAL_MEAN = 141.8            # degrees C, true steady-state mean
NORMAL_STD  = 1.4              # degrees C, true steady-state std

START_TEMP  = 30.0             # ambient temperature at startup
STARTUP_DURATION = 180         # minutes to ramp up to operating temp
SHUTDOWN_DURATION = 150        # minutes to ramp down

# ----------------------------------------------------------------------
# Build a timeline with regime labels: 'startup', 'steady', 'shutdown', 'fault'
# ----------------------------------------------------------------------
timestamps = pd.date_range("2025-01-01", periods=N, freq="1min")
temp = np.full(N, NORMAL_MEAN)
regime = np.full(N, "steady", dtype=object)

# --- Inject 4 startup/shutdown cycles spread across the 6 months ---
cycle_starts = [0, N // 4, N // 2, 3 * N // 4]

for start in cycle_starts:
    # Startup ramp: ambient -> operating temperature
    end_startup = start + STARTUP_DURATION
    if end_startup < N:
        ramp = np.linspace(START_TEMP, NORMAL_MEAN, STARTUP_DURATION)
        temp[start:end_startup] = ramp
        regime[start:end_startup] = "startup"

    # Shutdown ramp, a bit later in the cycle (simulate a planned outage)
    shutdown_start = start + N // 8
    shutdown_end = shutdown_start + SHUTDOWN_DURATION
    if shutdown_end < N:
        ramp_down = np.linspace(NORMAL_MEAN, START_TEMP, SHUTDOWN_DURATION)
        temp[shutdown_start:shutdown_end] = ramp_down
        regime[shutdown_start:shutdown_end] = "shutdown"

        # Brief restart after shutdown
        restart_end = shutdown_end + STARTUP_DURATION
        if restart_end < N:
            ramp_up = np.linspace(START_TEMP, NORMAL_MEAN, STARTUP_DURATION)
            temp[shutdown_end:restart_end] = ramp_up
            regime[shutdown_end:restart_end] = "startup"

# --- Add realistic Gaussian sensor noise on top of steady-state periods ---
steady_mask = regime == "steady"
temp[steady_mask] += np.random.normal(0, NORMAL_STD, steady_mask.sum())

# --- Inject 6 REAL faults: sustained deviations lasting 15-40 minutes ---
fault_events = []
rng = np.random.default_rng(7)
fault_start_candidates = np.where(steady_mask)[0]
chosen_faults = rng.choice(fault_start_candidates, size=6, replace=False)

for f_start in chosen_faults:
    duration = rng.integers(15, 40)
    f_end = min(f_start + duration, N)
    # Real fault: sustained shift of 4-7 std devs, not a single spike
    shift = rng.choice([-1, 1]) * rng.uniform(4.0, 7.0) * NORMAL_STD
    temp[f_start:f_end] += shift
    regime[f_start:f_end] = "fault"
    fault_events.append((timestamps[f_start], timestamps[f_end - 1]))

# --- Inject 25 single-point sensor noise spikes (NOT real faults) ---
spike_candidates = np.where(regime == "steady")[0]
chosen_spikes = rng.choice(spike_candidates, size=25, replace=False)
for s in chosen_spikes:
    spike_shift = rng.choice([-1, 1]) * rng.uniform(3.2, 4.5) * NORMAL_STD
    temp[s] += spike_shift   # single point only -- regime label stays "steady"

# ----------------------------------------------------------------------
# Save to CSV
# ----------------------------------------------------------------------
df = pd.DataFrame({
    "timestamp": timestamps,
    "top_temp_C": temp.round(3),
    "true_regime": regime,   # ground truth label, used only for evaluation
})

df.to_csv("historian_data.csv", index=False)

print(f"Generated {N:,} readings over {DAYS} days")
print(f"Regime breakdown:\n{df['true_regime'].value_counts()}")
print(f"\nInjected {len(fault_events)} real fault events:")
for s, e in fault_events:
    print(f"  {s} to {e}")
print(f"\nInjected 25 single-point sensor noise spikes (not real faults)")
print(f"\nSaved to historian_data.csv")
