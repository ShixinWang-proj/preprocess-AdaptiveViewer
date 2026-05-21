"""
Feature extraction from preprocessed PPG (ied_final) signal.

Pipeline:
  1. Smooth the signal (Savitzky-Golay) to remove high-frequency noise
  2. Detect positive peaks and negative troughs as heartbeat anchors
  3. For each peak pair, find cycle boundaries (zero-crossings or fallback)
  4. Extract 4 features per cycle: timestamp, RRI, area_up, area_down, motion

Outputs a CSV with one row per heartbeat cycle.
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from scipy.integrate import trapezoid


def smooth_signal(signal, window_length=21, polyorder=3):
    """Savitzky-Golay smoothing that preserves peak positions.

    Parameters
    ----------
    signal : 1D array
        Input signal.
    window_length : int
        Smoothing window in samples. 21 at 100Hz = 0.21s,
        well below the ~0.8s pulse period so peaks are not shifted.
    polyorder : int
        Polynomial order (must be < window_length).

    Returns
    -------
    smoothed : 1D array
        Smoothed signal, same length as input.
    """
    # Ensure odd window length
    if window_length % 2 == 0:
        window_length += 1
    # Must have window_length < len(signal)
    if len(signal) < window_length:
        return signal.copy()
    return savgol_filter(signal, window_length, polyorder)


def detect_positive_peaks(signal, fs=100, min_height=None, min_rr=0.5):
    """Detect positive peaks as heartbeat anchors.

    Parameters
    ----------
    signal : 1D array
        Smoothed ied_final values (no NaN within segment).
    fs : int
        Sampling rate.
    min_height : float or None
        Absolute minimum peak height. If None, uses adaptive threshold.
    min_rr : float
        Minimum inter-beat interval in seconds. Set to 0.5 since
        a normal pulse is ~0.8s, so two peaks cannot be closer than 0.5s.

    Returns
    -------
    peak_indices : 1D array of int
        Indices of detected positive peaks in the input signal array.
    peak_heights : 1D array
        Heights of detected peaks.
    """
    min_distance = int(min_rr * fs)  # 0.5s * 100Hz = 50 samples
    if min_height is None:
        pos_vals = signal[signal > 0]
        if len(pos_vals) > 0:
            min_height = max(3000, np.percentile(pos_vals, 30))
        else:
            min_height = 3000

    peaks, props = find_peaks(signal, height=min_height, distance=min_distance)
    return peaks, props['peak_heights']


def find_zero_crossings(signal, fs=100, noise_tol=3):
    """Find robust zero-crossing points.

    A zero-crossing is confirmed only when the signal stays on each side
    for at least `noise_tol` consecutive samples (filters baseline noise).

    Parameters
    ----------
    signal : 1D array
        Signal segment.
    noise_tol : int
        Minimum consecutive samples above/below zero to confirm a crossing.

    Returns
    -------
    crossings : list of (index, direction) tuples
        direction: 'up' (neg->pos) or 'down' (pos->neg)
        index is the interpolated crossing point.
    """
    crossings = []
    n = len(signal)

    # Find sign-change candidates
    signs = np.sign(signal)
    candidates = np.where(np.diff(signs) != 0)[0]

    for idx in candidates:
        # Determine direction
        if signs[idx] < 0 and signs[idx + 1] > 0:
            direction = 'up'
        elif signs[idx] > 0 and signs[idx + 1] < 0:
            direction = 'down'
        else:
            continue

        # Verify: need noise_tol consecutive samples on each side
        # Before the crossing
        left_ok = False
        if idx >= noise_tol:
            left_signs = np.sign(signal[idx - noise_tol:idx])
            if direction == 'down' and np.all(left_signs >= 0):
                left_ok = True
            elif direction == 'up' and np.all(left_signs <= 0):
                left_ok = True
        elif idx > 0 and ((direction == 'down' and signal[idx] >= 0) or
                          (direction == 'up' and signal[idx] <= 0)):
            left_ok = True

        if not left_ok:
            continue

        # After the crossing
        right_ok = False
        if idx + 1 + noise_tol <= n:
            right_signs = np.sign(signal[idx + 1:idx + 1 + noise_tol])
            if direction == 'down' and np.all(right_signs <= 0):
                right_ok = True
            elif direction == 'up' and np.all(right_signs >= 0):
                right_ok = True

        if not right_ok:
            continue

        # Linear interpolation for precise crossing position
        frac = -signal[idx] / (signal[idx + 1] - signal[idx])
        exact_idx = idx + frac
        crossings.append((exact_idx, direction))

    return crossings


def compute_cycle_features(signal, peak_indices, peak_heights, fs=100,
                           accX=None, accY=None, accZ=None):
    """Extract per-cycle features between consecutive positive peaks.

    For each pair of consecutive peaks, finds:
    - timestamp: time of the first peak
    - RRI: inter-beat interval (seconds)
    - area_up: integral of positive portion between the two peaks
    - area_down: absolute integral of negative portion
    - motion: mean vector magnitude of 3-axis accelerometer in the window

    Parameters
    ----------
    signal : 1D array
        Preprocessed ied_final signal.
    peak_indices : 1D array
        Indices of detected positive peaks.
    peak_heights : 1D array
        Heights of detected peaks.
    fs : int
        Sampling rate.
    accX, accY, accZ : 1D arrays or None
        Accelerometer data (same length as signal).

    Returns
    -------
    features : list of dicts
        One dict per cycle with keys: timestamp, rri, area_up, area_down, motion,
        peak_height, cycle_start, cycle_end, quality.
    """
    features = []
    n_cycles = len(peak_indices) - 1

    for i in range(n_cycles):
        p0 = peak_indices[i]
        p1 = peak_indices[i + 1]
        h0 = peak_heights[i]
        h1 = peak_heights[i + 1]
        rri = (p1 - p0) / fs

        # Slice the cycle
        cycle = signal[p0:p1 + 1]

        # Find robust zero-crossings within the cycle
        crossings = find_zero_crossings(cycle, fs, noise_tol=3)

        # Identify the first 'down' crossing (peak -> negative)
        # and the last 'up' crossing (negative -> next peak)
        down_crossings = [c for c in crossings if c[1] == 'down']
        up_crossings = [c for c in crossings if c[1] == 'up']

        quality = 'good'
        cycle_min = cycle.min()

        if len(down_crossings) > 0 and len(up_crossings) > 0:
            # Standard case: clean biphasic cycle
            zc_down = down_crossings[0][0]
            zc_up = up_crossings[-1][0]

            # AreaUp: from start to down-crossing (positive portion)
            end_up = int(np.ceil(zc_down)) + 1
            end_up = min(end_up, len(cycle))
            area_up = trapezoid(np.maximum(cycle[:end_up], 0), dx=1.0 / fs)

            # AreaDown: from down-crossing to up-crossing (negative portion)
            start_down = int(np.floor(zc_down))
            end_down = int(np.ceil(zc_up)) + 1
            end_down = min(end_down, len(cycle))
            area_down = trapezoid(np.abs(np.minimum(cycle[start_down:end_down], 0)), dx=1.0 / fs)

        elif cycle_min < 0:
            # No valid zero-crossings but signal does go negative
            # Use the minimum point as the split
            quality = 'degraded'
            neg_mask = cycle < 0
            area_up = trapezoid(np.maximum(cycle, 0), dx=1.0 / fs)
            area_down = trapezoid(np.abs(np.minimum(cycle, 0)), dx=1.0 / fs)

        else:
            # Entire cycle is positive (no negative lobe)
            quality = 'degraded'
            area_up = trapezoid(np.maximum(cycle, 0), dx=1.0 / fs)
            area_down = 0.0

        # Motion: mean vector magnitude over the cycle window
        if accX is not None:
            mv = np.sqrt(accX[p0:p1 + 1] ** 2 +
                         accY[p0:p1 + 1] ** 2 +
                         accZ[p0:p1 + 1] ** 2)
            motion = float(np.mean(mv))
        else:
            motion = 0.0

        # Timestamp: time of the first peak
        timestamp = p0 / fs

        features.append({
            'timestamp': round(timestamp, 4),
            'rri': round(rri, 4),
            'area_up': round(area_up, 4),
            'area_down': round(area_down, 4),
            'motion': round(motion, 4),
            'peak_height': round(h0, 2),
            'cycle_start': p0,
            'cycle_end': p1,
            'quality': quality,
        })

    return features


def extract_features(input_path, output_path=None, fs=100,
                     smooth_window=21, smooth_polyorder=3):
    """Full feature extraction pipeline.

    Parameters
    ----------
    input_path : str
        Path to preprocessed CSV file.
    output_path : str or None
        Path for output feature CSV. If None, no file is written.
    fs : int
        Sampling rate.
    smooth_window : int
        Savitzky-Golay smoothing window length (samples).
    smooth_polyorder : int
        Savitzky-Golay polynomial order.

    Returns
    -------
    df_features : DataFrame
        Feature table with one row per heartbeat cycle.
    """
    df = pd.read_csv(input_path)

    # Handle NaN: process in clean segments
    clean_mask = ~df['ied_final'].isna()
    y = df['ied_final'].values

    # Find clean segments
    segments = []
    in_clean = False
    seg_start = 0
    for i, v in enumerate(clean_mask):
        if v and not in_clean:
            seg_start = i
            in_clean = True
        elif not v and in_clean:
            if i - seg_start > int(2 * fs):
                segments.append((seg_start, i))
            in_clean = False
    if in_clean and len(clean_mask) - seg_start > int(2 * fs):
        segments.append((seg_start, len(clean_mask)))

    all_features = []

    for s, e in segments:
        seg_raw = y[s:e]

        # Step 1: smooth the signal
        seg_signal = smooth_signal(seg_raw, smooth_window, smooth_polyorder)

        # Step 2: detect positive peaks
        peaks, peak_heights = detect_positive_peaks(seg_signal, fs)

        if len(peaks) < 2:
            continue

        # Extract features (uses smoothed signal for area calc)
        accX = df['accX'].values[s:e] if 'accX' in df.columns else None
        accY = df['accY'].values[s:e] if 'accY' in df.columns else None
        accZ = df['accZ'].values[s:e] if 'accZ' in df.columns else None

        cycle_features = compute_cycle_features(
            seg_signal, peaks, peak_heights, fs, accX, accY, accZ
        )

        # Adjust cycle_start/cycle_end to global indices
        for feat in cycle_features:
            feat['cycle_start'] += s
            feat['cycle_end'] += s

        all_features.extend(cycle_features)

    df_features = pd.DataFrame(all_features)

    if output_path is not None and df_features is not None:
        df_features.to_csv(output_path, index=False)
        print(f"Features saved: {output_path}")
        print(f"Total cycles: {len(df_features)}")
        print(f"  good: {(df_features['quality'] == 'good').sum()}")
        print(f"  degraded: {(df_features['quality'] == 'degraded').sum()}")

    return df_features


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PPG feature extraction')
    parser.add_argument('--input_path', '-i', required=True, help='Input preprocessed CSV')
    parser.add_argument('--output_path', '-o', default=None, help='Output features CSV')
    parser.add_argument('--fs', type=int, default=100, help='Sampling rate (default: 100)')
    args = parser.parse_args()

    if args.output_path is None:
        base = args.input_path.replace('.csv', '')
        args.output_path = f'{base}_features.csv'

    extract_features(args.input_path, args.output_path, fs=args.fs)
