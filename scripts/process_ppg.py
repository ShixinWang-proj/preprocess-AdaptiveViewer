"""
PPG Data Processing Pipeline
Following TODO_PPG_Processing.md steps
"""

import pandas as pd
import numpy as np
from scipy import signal
from scipy.stats import zscore
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1: 预处理 (Preprocessing)
# ============================================================

def load_and_preprocess(filepath):
    """加载并预处理数据"""
    print("=" * 60)
    print("STEP 1: 预处理")
    print("=" * 60)

    df = pd.read_csv(filepath)
    print(f"[1.1] 加载数据: {df.shape}")

    # 解析时间戳
    df['datetime'] = pd.to_datetime(df['datetime'])
    print(f"[1.2] 时间戳解析完成: {df['datetime'].min()} ~ {df['datetime'].max()}")

    # 缺失值检测
    missing = df.isnull().sum()
    print(f"[1.3] 缺失值:\n{missing}")

    # 异常值检测（Z-score > 3 视为异常）
    for col in ['ied', 'motion']:
        z = np.abs(zscore(df[col].dropna()))
        outliers = (z > 3).sum()
        print(f"[1.4] {col} 异常值数量 (|z|>3): {outliers} ({outliers/len(df)*100:.2f}%)")

    # 创建处理后的数据副本
    df_clean = df.copy()
    return df_clean


# ============================================================
# STEP 2: 信号滤波 (Filtering)
# ============================================================

def filter_signal(df_clean):
    print("\n" + "=" * 60)
    print("STEP 2: 信号滤波")
    print("=" * 60)

    # 假设采样率约为 30-50 Hz（根据数据时间间隔估算）
    # 估算采样率
    time_diff = (df_clean['datetime'].diff().dt.total_seconds()).median()
    fs = 1 / time_diff if time_diff > 0 else 30
    print(f"[2.1] 估算采样率: {fs:.2f} Hz")

    ied_signal = df_clean['ied'].values
    motion_signal = df_clean['motion'].values

    # 2.2 去除直流分量
    ied_dc_removed = ied_signal - np.mean(ied_signal)
    print(f"[2.2] 去除直流分量: mean={np.mean(ied_signal):.2f} -> {np.mean(ied_dc_removed):.2f}")

    # 2.3 带通滤波 (0.5-4 Hz, 对应 30-240 BPM)
    low_freq = 0.5  # Hz
    high_freq = 4.0  # Hz
    nyquist = fs / 2

    if high_freq < nyquist:
        b, a = signal.butter(4, [low_freq/nyquist, high_freq/nyquist], btype='band')
        ied_filtered = signal.filtfilt(b, a, ied_dc_removed)
        print(f"[2.3] 带通滤波: {low_freq}-{high_freq} Hz")
    else:
        ied_filtered = ied_dc_removed
        print(f"[2.3] 带通滤波: 采样率不足，跳过")

    # 2.4 可选：使用 motion 信号去除运动伪影（简单相减）
    # 运动伪影通常与加速度信号相关
    motion_normalized = (motion_signal - np.mean(motion_signal)) / np.std(motion_signal)
    ied_normalized = (ied_filtered - np.mean(ied_filtered)) / np.std(ied_filtered)
    correlation = np.corrcoef(motion_normalized, ied_normalized)[0, 1]
    print(f"[2.4] motion与ied相关性: {correlation:.4f}")

    df_clean['ied_filtered'] = ied_filtered
    df_clean['ied_dc_removed'] = ied_dc_removed

    return df_clean, fs


# ============================================================
# STEP 3: 信号质量评估 (Signal Quality Index)
# ============================================================

def assess_quality(df_clean):
    print("\n" + "=" * 60)
    print("STEP 3: 信号质量评估")
    print("=" * 60)

    ied = df_clean['ied_filtered'].values

    # 3.1 信噪比（SNR）估算
    # 信号 = 峰值部分的功率，噪声 = 非峰值部分的功率
    signal_power = np.mean(ied ** 2)
    noise_power = np.var(ied)
    snr = 10 * np.log10(signal_power / noise_power)
    print(f"[3.1] SNR: {snr:.2f} dB")

    # 3.2 信号幅度
    amplitude = np.max(ied) - np.min(ied)
    print(f"[3.2] 信号幅度: {amplitude:.2f}")

    # 3.3 周期一致性（通过自相关）
    autocorr = np.correlate(ied, ied, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    peak_indices, _ = signal.find_peaks(autocorr[:1000], height=0.1*autocorr[0])
    if len(peak_indices) > 1:
        avg_period = np.mean(np.diff(peak_indices))
        print(f"[3.3] 平均周期（样本数）: {avg_period:.1f}")

    df_clean['snr'] = snr
    return df_clean


# ============================================================
# STEP 4: 脉搏波检测 (Pulse Wave Detection)
# ============================================================

def detect_pulse_waves(df_clean, fs):
    print("\n" + "=" * 60)
    print("STEP 4: 脉搏波检测")
    print("=" * 60)

    ied = df_clean['ied_filtered'].values

    # 4.1 峰值检测
    # 寻找 R 峰/收缩峰
    peaks, peak_properties = signal.find_peaks(ied, distance=int(fs*0.3), height=np.percentile(ied, 50))
    print(f"[4.1] 检测到峰值数量: {len(peaks)}")

    # 4.2 谷值检测
    valleys, valley_properties = signal.find_peaks(-ied, distance=int(fs*0.3))
    print(f"[4.2] 检测到谷值数量: {len(valleys)}")

    # 4.3 周期分割（每两个峰值之间为一个心跳周期）
    if len(peaks) > 1:
        peak_intervals = np.diff(peaks)
        heart_rate = fs / peak_intervals * 60  # BPM
        heart_rate = heart_rate[(heart_rate > 40) & (heart_rate < 200)]  # 合理范围
        print(f"[4.3] 心率估计: {np.mean(heart_rate):.1f} ± {np.std(heart_rate):.1f} BPM")
    else:
        heart_rate = []
        print(f"[4.3] 心率估计: 数据不足")

    df_clean['peaks'] = None  # 存储峰值索引
    df_clean['valleys'] = None

    return df_clean, peaks, valleys, heart_rate


# ============================================================
# STEP 5: 特征提取 (Feature Extraction)
# ============================================================

def extract_features(df_clean, peaks, valleys, fs):
    print("\n" + "=" * 60)
    print("STEP 5: 特征提取")
    print("=" * 60)

    ied = df_clean['ied_filtered'].values
    ied_raw = df_clean['ied'].values

    features = []

    if len(peaks) > 0:
        # 时域特征
        peak_heights = ied[peaks]
        print(f"[5.1] 峰值幅度: mean={np.mean(peak_heights):.2f}, std={np.std(peak_heights):.2f}")

        # 脉搏波宽度（半高宽）
        half_max = (np.max(ied) + np.min(ied)) / 2
        pulse_widths = []
        for peak in peaks[:100]:  # 只计算前100个
            left_idx = np.where(ied[:peak] < half_max)[0]
            right_idx = np.where(ied[peak:] < half_max)[0]
            if len(left_idx) > 0 and len(right_idx) > 0:
                width = (right_idx[0] + peak - left_idx[-1]) / fs * 1000  # ms
                pulse_widths.append(width)

        if pulse_widths:
            print(f"[5.2] 脉搏波宽度: {np.mean(pulse_widths):.1f} ± {np.std(pulse_widths):.1f} ms")

        # 上升时间和下降时间
        if len(peaks) > 1 and len(valleys) > 0:
            upstrokes = []
            downstrokes = []
            for i in range(min(len(peaks), len(valleys))):
                if valleys[i] < peaks[i]:
                    up_time = (peaks[i] - valleys[i]) / fs * 1000
                    upstrokes.append(up_time)
                if i < len(valleys) - 1 and peaks[i] < valleys[i+1]:
                    down_time = (valleys[i+1] - peaks[i]) / fs * 1000
                    downstrokes.append(down_time)

            if upstrokes:
                print(f"[5.3] 上升时间: {np.mean(upstrokes):.1f} ± {np.std(upstrokes):.1f} ms")
            if downstrokes:
                print(f"[5.4] 下降时间: {np.mean(downstrokes):.1f} ± {np.std(downstrokes):.1f} ms")

        # 收缩期/舒张期比率
        systolic_amplitudes = []
        diastolic_amplitudes = []
        for i in range(min(len(peaks), len(valleys)) - 1):
            if valleys[i] < peaks[i] < valleys[i+1]:
                systolic = ied[peaks[i]] - ied[valleys[i]]
                diastolic = ied[peaks[i]] - ied[valleys[i+1]]
                systolic_amplitudes.append(systolic)
                diastolic_amplitudes.append(diastolic)

        if systolic_amplitudes and diastolic_amplitudes:
            ratio = np.mean(systolic_amplitudes) / np.mean(diastolic_amplitudes)
            print(f"[5.5] 收缩期/舒张期比率: {ratio:.3f}")

        # HRV 分析（频域）
        if len(peaks) > 10:
            rr_intervals = np.diff(peaks) / fs * 1000  # ms
            # 计算 HRV 指标
            rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
            print(f"[5.6] RMSSD (HRV): {rmssd:.2f} ms")

    return features


# ============================================================
# STEP 6: 后处理与验证 (Post-processing)
# ============================================================

def postprocess_and_validate(df_clean, peaks, valleys, heart_rate, fs):
    print("\n" + "=" * 60)
    print("STEP 6: 后处理与验证")
    print("=" * 60)

    # 6.1 心率验证
    if len(heart_rate) > 0:
        valid_hr = heart_rate[(heart_rate > 40) & (heart_rate < 200)]
        print(f"[6.1] 有效心率数量: {len(valid_hr)}/{len(heart_rate)}")

    # 6.2 异常心跳标记
    if len(heart_rate) > 0:
        hr_z = np.abs(zscore(heart_rate))
        anomalous = np.where(hr_z > 2)[0]
        print(f"[6.2] 异常心跳数量: {len(anomalous)}")

    # 6.3 可视化
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # 原始信号
    axes[0].plot(df_clean['datetime'], df_clean['ied'], alpha=0.7, label='Raw IED')
    axes[0].set_title('Raw PPG Signal (IED)')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend()

    # 滤波后信号
    axes[1].plot(df_clean['datetime'], df_clean['ied_filtered'], color='orange', label='Filtered IED')
    if len(peaks) > 0:
        axes[1].scatter(df_clean['datetime'].iloc[peaks], df_clean['ied_filtered'].iloc[peaks],
                       color='red', s=10, label='Peaks')
    if len(valleys) > 0:
        axes[1].scatter(df_clean['datetime'].iloc[valleys], df_clean['ied_filtered'].iloc[valleys],
                       color='green', s=10, label='Valleys')
    axes[1].set_title('Filtered PPG Signal with Detected Peaks/Valleys')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Amplitude')
    axes[1].legend()

    # 心率时序
    if len(heart_rate) > 0:
        axes[2].plot(heart_rate, alpha=0.7)
        axes[2].axhline(y=np.mean(heart_rate), color='r', linestyle='--', label=f'Mean: {np.mean(heart_rate):.1f} BPM')
        axes[2].set_title('Heart Rate Over Time')
        axes[2].set_xlabel('Beat Index')
        axes[2].set_ylabel('BPM')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('data/processed/ppg_processing_results.png', dpi=150)
    print(f"[6.3] 可视化结果已保存: data/processed/ppg_processing_results.png")

    return df_clean


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    filepath = 'data/raw/may_piece.csv'
    output_dir = 'data/processed'

    # 创建输出目录
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "#" * 60)
    print("# PPG 数据处理流水线")
    print("#" * 60)

    # Step 1: 预处理
    df = load_and_preprocess(filepath)

    # Step 2: 滤波
    df, fs = filter_signal(df)

    # Step 3: 信号质量评估
    df = assess_quality(df)

    # Step 4: 脉搏波检测
    df, peaks, valleys, heart_rate = detect_pulse_waves(df, fs)

    # Step 5: 特征提取
    features = extract_features(df, peaks, valleys, fs)

    # Step 6: 后处理与验证
    df = postprocess_and_validate(df, peaks, valleys, heart_rate, fs)

    # 保存处理后的数据
    df.to_csv('data/processed/may_piece_processed.csv', index=False)
    print(f"\n处理后数据已保存: data/processed/may_piece_processed.csv")

    print("\n" + "#" * 60)
    print("# 处理完成!")
    print("#" * 60)

    return df, peaks, valleys, heart_rate


if __name__ == "__main__":
    df, peaks, valleys, heart_rate = main()
