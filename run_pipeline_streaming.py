# --- Guard snippet: force baseline combine if anomaly median is suspiciously high ---
import os

# compute median anomaly safely (df_all should be the DataFrame holding results for this run)
try:
    anom_median = float(df_all['anomaly_score'].median())
except Exception:
    # If df_all or the column isn't available, skip the guard (fail-safe)
    anom_median = None

# configurable threshold via environment variable (defaults to 0.6)
_guard_th = os.getenv('ANOM_MEDIAN_GUARD', '0.6')
try:
    guard_threshold = float(_guard_th)
except Exception:
    guard_threshold = 0.6

# determine chosen_combine_mode: default to whatever the CLI requested (chosen_combine_mode),
# but force 'baseline' if anomaly median is above the threshold
chosen_combine_mode = getattr(args, 'combine_mode', None) if 'args' in globals() else None
if anom_median is not None and guard_threshold is not None:
    if anom_median > guard_threshold:
        print(f"[guard] anomaly median={anom_median:.3f} > {guard_threshold} — forcing combine_mode='baseline'")
        chosen_combine_mode = 'baseline'
# else chosen_combine_mode remains whatever was configured

# Use chosen_combine_mode in the combine/recombine call below
# Example: replace any call like
#    recombine(df_all, chosen_combine_mode, ...)
# with:
#    recombine(df_all, chosen_combine_mode, ...)
# -------------------------------------------------------------------------------