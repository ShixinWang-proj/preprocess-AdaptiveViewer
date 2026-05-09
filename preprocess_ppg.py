"""
PPG preprocessing: detect polluted events (jumps + spikes),
mark them as NaN (margin=100 samples), then per-segment high-pass.
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, medfilt


def detect_events(ied, fs=100, jump_thresh=500000, spike_window_sec=5, spike_n_mad=10):
    """Detect polluted events: baseline jumps AND single-point spikes.

    Returns a set of sample indices representing event centers.

    Parameters
    ----------
    ied : 1D array, raw IED samples
    fs : int, sampling rate
    jump_thresh : float, minimum |1-sec diff| to flag a jump
    spike_window_sec : int, seconds for rolling median window (odd)
    spike_n_mad : float, MAD multiplier for spike threshold

    Returns
    -------
    event_centers : set of sample indices
    """
    events = set()
    n = len(ied)

    # --- 1. Second-level baseline jumps ---
    step = fs
    n_sec = n // step
    ied_sec = ied[: n_sec * step].reshape(-1, step).mean(axis=1)
    diff_1s = np.abs(np.diff(ied_sec))
    jump_secs = np.where(diff_1s > jump_thresh)[0]
    for j in jump_secs:
        events.add(int(j * fs))

    # --- 2. Single-point spikes (MAD from rolling median) ---
    w = spike_window_sec * fs
    if w % 2 == 0:
        w += 1
    local_median = medfilt(ied.astype(float), kernel_size=w)
    deviation = np.abs(ied.astype(float) - local_median)
    mad = np.median(deviation)
    spike_mask = deviation > spike_n_mad * mad
    for si in np.where(spike_mask)[0]:
        events.add(int(si))

    return events, diff_1s, ied_sec


def build_polluted_segments(event_centers, n, margin=100):
    """Expand each event by ±margin, merge overlaps.

    Parameters
    ----------
    event_centers : set of sample indices
    n : int, total samples
    margin : int, samples to expand around each event center

    Returns
    -------
    merged : list of (start_idx, end_idx) tuples (inclusive)
    """
    segments = []
    for center in event_centers:
        s = max(0, center - margin)
        e = min(n - 1, center + margin)
        segments.append((s, e))

    segments.sort()
    merged = []
    if segments:
        ms, me = segments[0]
        for ns, ne in segments[1:]:
            if ns <= me + margin:
                me = max(me, ne)
            else:
                merged.append((ms, me))
                ms, me = ns, ne
        merged.append((ms, me))
    return merged


def mark_nan(ied, merged_segments):
    ied_masked = ied.astype(float)
    for s, e in merged_segments:
        ied_masked[s: e + 1] = np.nan
    return ied_masked


def highpass_per_segment(ied_masked, fs=100, fc=0.5, order=4, min_seg=2):
    ied_hp = np.full(len(ied_masked), np.nan)
    Wn = fc / (fs / 2)
    b, a = butter(order, Wn, btype='high')

    valid_mask = ~np.isnan(ied_masked)
    in_valid = False
    seg_count = 0
    min_samples = int(min_seg * fs)

    for i in range(len(ied_masked)):
        if valid_mask[i] and not in_valid:
            seg_start = i
            in_valid = True
        elif not valid_mask[i] and in_valid:
            seg_len = i - seg_start
            if seg_len > min_samples:
                ied_hp[seg_start:i] = filtfilt(b, a, ied_masked[seg_start:i])
                seg_count += 1
            in_valid = False
    if in_valid:
        seg_len = len(ied_masked) - seg_start
        if seg_len > min_samples:
            ied_hp[seg_start:] = filtfilt(b, a, ied_masked[seg_start:])
            seg_count += 1

    return ied_hp


def preprocess(ied, fs=100, jump_thresh=500000, margin=100, fc=0.5,
               spike_window_sec=5, spike_n_mad=10):
    """Full pipeline. Returns dict with all intermediate results."""
    n = len(ied)

    event_centers, diff_1s, ied_sec = detect_events(
        ied, fs, jump_thresh, spike_window_sec, spike_n_mad)
    merged = build_polluted_segments(event_centers, n, margin)
    ied_masked = mark_nan(ied, merged)
    ied_final = highpass_per_segment(ied_masked, fs, fc)

    return {
        'event_centers': event_centers,
        'diff_1s': diff_1s,
        'ied_sec': ied_sec,
        'merged_segments': merged,
        'ied_masked': ied_masked,
        'ied_final': ied_final,
    }


def save_preprocessed_csv(input_path, output_path, fs=100):
    df = pd.read_csv(input_path)
    ied = df['ied'].values.astype(float)
    n = len(ied)

    df['elapsed_s'] = np.arange(n) / fs

    result = preprocess(ied, fs=fs)

    df['ied_masked'] = result['ied_masked']
    df['ied_final'] = result['ied_final']
    df.to_csv(output_path, index=False)

    total_nan = np.sum(np.isnan(result['ied_masked']))
    print(f"Events detected: {len(result['event_centers'])}")
    print(f"Polluted segments: {len(result['merged_segments'])}")
    for i, (s, e) in enumerate(result['merged_segments']):
        print(f"  Seg {i+1}: idx {s}~{e} ({s/fs:.1f}~{e/fs:.1f}s), len={e-s+1}")
    print(f"NaN points: {total_nan}/{n} ({total_nan/n*100:.1f}%)")
    print(f"Saved: {output_path}")
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PPG preprocessing pipeline')
    parser.add_argument('--input_path', '-i', required=True, help='Input CSV path')
    parser.add_argument('--output_path', '-o', required=True, help='Output CSV path')
    parser.add_argument('--fs', type=int, default=100, help='Sampling rate (default: 100)')
    args = parser.parse_args()
    save_preprocessed_csv(args.input_path, args.output_path, fs=args.fs)
