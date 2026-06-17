# -*- coding: utf-8 -*-
"""
detector.py
-----------
Z-score based anomaly detector for distillation column temperature,
built directly from Session 1 concepts:

  1. Filter to steady-state data using domain knowledge (NOT statistics)
     before computing any baseline -- avoids the circular outlier problem.
  2. Compute mean/std with ddof=1 (Bessel's correction) on the clean baseline.
  3. Flag readings with |Z| > 3 as candidate anomalies.
  4. Apply the consecutive-reading fix to avoid the multiple-comparisons
     false-alarm flood: require N consecutive flagged readings before
     raising a real alarm, collapsing the false alarm rate from
     0.27% per reading to ~0.27%^N for the whole window.
"""

import numpy as np
import pandas as pd


class ZScoreAnomalyDetector:
    """
    Production-style anomaly detector following the Session 1 checklist.
    """

    def __init__(self, z_threshold: float = 3.0, consecutive_required: int = 3):
        self.z_threshold = z_threshold
        self.consecutive_required = consecutive_required
        self.mean_ = None
        self.std_ = None
        self.n_baseline_ = None

    # ------------------------------------------------------------------
    # Step 1 + 2: fit baseline on pre-filtered steady-state data only
    # ------------------------------------------------------------------
    def fit(self, steady_state_values: np.ndarray):
        """
        steady_state_values: array of readings ALREADY filtered to
        steady-state operation using domain knowledge (timestamps,
        operator logs, regime labels) -- never filtered by statistics,
        to avoid the circular outlier-removal problem from Session 1.
        """
        self.n_baseline_ = len(steady_state_values)
        self.mean_ = np.mean(steady_state_values)
        self.std_  = np.std(steady_state_values, ddof=1)  # Bessel's correction
        return self

    # ------------------------------------------------------------------
    # Step 3: compute Z-scores for new data
    # ------------------------------------------------------------------
    def z_scores(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Call fit() before scoring new data.")
        return (values - self.mean_) / self.std_

    # ------------------------------------------------------------------
    # Step 4: the consecutive-reading fix (the multiple-comparisons trap)
    # ------------------------------------------------------------------
    def flag_anomalies(self, values: np.ndarray) -> dict:
        """
        Returns a dict with:
          'z'                : Z-score for every reading
          'single_point_flag' : True wherever |Z| > threshold (raw, noisy)
          'confirmed_alarm'   : True only where N consecutive single-point
                                 flags occur -- this is the actual alarm
                                 you would send to an operator.
        """
        z = self.z_scores(values)
        single_point_flag = np.abs(z) > self.z_threshold

        confirmed_alarm = np.zeros_like(single_point_flag, dtype=bool)
        run_length = 0
        for i, flagged in enumerate(single_point_flag):
            run_length = run_length + 1 if flagged else 0
            if run_length >= self.consecutive_required:
                # Mark the whole run as a confirmed alarm, not just the
                # Nth point, so the operator sees the full event window.
                confirmed_alarm[i - run_length + 1 : i + 1] = True

        return {
            "z": z,
            "single_point_flag": single_point_flag,
            "confirmed_alarm": confirmed_alarm,
        }

    def summary(self) -> str:
        return (
            f"Baseline: mean={self.mean_:.3f} C, std={self.std_:.3f} C "
            f"(n={self.n_baseline_:,} steady-state readings, ddof=1)\n"
            f"Threshold: |Z| > {self.z_threshold} "
            f"({self._expected_false_rate()*100:.3f}% false-alarm rate per single reading)\n"
            f"Consecutive readings required to confirm alarm: {self.consecutive_required}"
        )

    def _expected_false_rate(self) -> float:
        from scipy import stats
        return 2 * (1 - stats.norm.cdf(self.z_threshold))


# ----------------------------------------------------------------------
# Demo / evaluation against the ground-truth labels from generate_data.py
# ----------------------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv("historian_data.csv", parse_dates=["timestamp"])

    # --- Step 1: filter to steady-state using DOMAIN KNOWLEDGE (the label),
    #             never using statistics to decide what counts as "clean" ---
    steady_mask = df["true_regime"] == "steady"
    baseline_values = df.loc[steady_mask, "top_temp_C"].values

    # NOTE: in a real deployment you would also exclude the known fault
    # windows from the baseline-fitting data, since here we are using
    # ALL "steady" rows including the single-point noise spikes on purpose,
    # to show the detector is still robust to a small amount of contamination.

    detector = ZScoreAnomalyDetector(z_threshold=3.0, consecutive_required=3)
    detector.fit(baseline_values)
    print(detector.summary())
    print()

    # --- Step 3 + 4: score the FULL dataset (steady + startup + shutdown + fault) ---
    result = detector.flag_anomalies(df["top_temp_C"].values)
    df["z_score"] = result["z"]
    df["single_point_flag"] = result["single_point_flag"]
    df["confirmed_alarm"] = result["confirmed_alarm"]

    # --- Evaluate against ground truth ---
    n_single_flags = df["single_point_flag"].sum()
    n_confirmed = df["confirmed_alarm"].sum()
    print(f"Raw single-point flags (|Z|>3):           {n_single_flags:,} readings")
    print(f"Confirmed alarms (3+ consecutive):        {n_confirmed:,} readings")
    print()

    # How many of the 6 real fault events were caught by confirmed alarms?
    real_faults = df[df["true_regime"] == "fault"]
    fault_groups = (real_faults["timestamp"].diff() > pd.Timedelta(minutes=5)).cumsum()
    n_real_fault_events = fault_groups.nunique()

    caught = 0
    for _, group in real_faults.groupby(fault_groups):
        if df.loc[group.index, "confirmed_alarm"].any():
            caught += 1

    print(f"Real fault events: {n_real_fault_events}")
    print(f"Real fault events caught by confirmed alarm: {caught}/{n_real_fault_events}")
    print()

    # How many single-point noise spikes were correctly NOT escalated to alarms?
    # This is the multiple-comparisons fix in action: raw Z>3 flags happen often,
    # but the consecutive-reading rule filters out isolated sensor noise.
    correctly_ignored = ((df["true_regime"] == "steady") &
                         df["single_point_flag"] &
                         ~df["confirmed_alarm"]).sum()
    print(f"Single-point noise spikes correctly NOT confirmed as alarms: {correctly_ignored:,}")

    df.to_csv("scored_data.csv", index=False)
    print("\nSaved scored_data.csv")
