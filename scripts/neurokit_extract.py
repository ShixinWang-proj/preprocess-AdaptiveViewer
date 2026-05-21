#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import gc
import numpy as np
import pandas as pd
import neurokit2 as nk

def process_robust_ppg(df, sampling_rate=100, trim_seconds=2):
    """
    分段重采样并裁切边缘的长时序 PPG 处理逻辑
    :param df: 包含 'datetime' 和 'ied' 的 pandas DataFrame
    :param sampling_rate: 目标重采样频率
    :param trim_seconds: 剔除每个连续段首尾的秒数
    """
    
    # 1. 基于原始时间戳识别连续段 (间隙 > 1秒视为断档)
    # 计算相邻时间戳的时间差（秒）
    time_diffs = df['datetime'].diff().dt.total_seconds()
    
    # 巧妙利用 cumsum 生成段落 ID：每次遇到 >1 秒的断档，ID 就加 1
    # 这样就把整个 25 小时的数据切成了独立的数据岛
    segment_ids = (time_diffs > 1.0).cumsum()
    
    results = []
    
    # 计算首尾需要裁切的点数
    trim_points = int(sampling_rate * trim_seconds)
    # 一个片段至少需要的长度：首尾裁切 + 至少 3 秒的有效数据
    min_required_points = trim_points * 2 + (sampling_rate * 3) 
    
    grouped = df.groupby(segment_ids)
    print(f"[*] 初步检测到 {len(grouped)} 个连续数据岛，开始独立重采样与提取...")

    for seg_id, group in grouped:
        # 如果原始数据量连裁切的底线都达不到，直接跳过
        if len(group) < min_required_points:
            continue
            
        try:
            # ---------------------------------------------------------
            # 2. 局部重采样 (消除当前片段内的采样抖动)
            # '10ms' 对应 100Hz (1秒 / 100)
            resample_rule = f"{int(1000/sampling_rate)}ms"
            
            # 将 datetime 设为索引并执行重采样插值
            group_indexed = group.set_index('datetime')
            resampled = group_indexed.resample(resample_rule).mean().interpolate(method='linear')
            resampled = resampled.reset_index()
            
            # ---------------------------------------------------------
            # 3. 边缘裁切 (剔除首尾不稳定的数据)
            if len(resampled) <= min_required_points:
                continue
                
            trimmed = resampled.iloc[trim_points : -trim_points].reset_index(drop=True)
            
            current_sig = trimmed['ied'].values
            current_times = trimmed['datetime'].values
            
            # ---------------------------------------------------------
            # 4. NeuroKit2 信号处理
            cleaned = nk.ppg_clean(current_sig, sampling_rate=sampling_rate, method='elgendi')
            
            # 波峰与心率
            peaks_info = nk.ppg_findpeaks(cleaned, sampling_rate=sampling_rate, method='elgendi')
            peaks = peaks_info["PPG_Peaks"]
            
            if len(peaks) > 1:
                hr_segment = nk.signal_rate(peaks, sampling_rate=sampling_rate, desired_length=len(current_sig))
                hr_at_peaks = hr_segment[peaks]
            else:
                hr_at_peaks = np.full(len(peaks), np.nan)
                
            # 波谷
            troughs_info = nk.ppg_findpeaks(-cleaned, sampling_rate=sampling_rate, method='elgendi')
            troughs = troughs_info["PPG_Peaks"]
            
            # ---------------------------------------------------------
            # 5. 将结果映射回绝对时间戳并记录
            for p, hr in zip(peaks, hr_at_peaks):
                results.append({
                    'datetime': current_times[p], 
                    'point_type': 'Peak', 
                    'ied_value': current_sig[p], 
                    'hr_value': hr
                })
                
            for t in troughs:
                results.append({
                    'datetime': current_times[t], 
                    'point_type': 'Trough', 
                    'ied_value': current_sig[t], 
                    'hr_value': np.nan
                })
                
            # 内存管理
            del group_indexed, resampled, trimmed, current_sig, cleaned
            if len(peaks) > 1:
                del hr_segment
            gc.collect()
            
        except Exception as e:
            print(f"[!] 数据岛 {seg_id} 处理失败: {e}")
            continue

    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description="高级分段重采样 PPG 提取工具")
    parser.add_argument("input_csv", help="输入的 CSV 文件路径")
    parser.add_argument("-o", "--output", default="ppg_results_advanced.csv", help="输出路径")
    parser.add_argument("-s", "--sr", type=int, default=100, help="目标重采样频率 Hz (默认: 100)")
    parser.add_argument("-t", "--trim", type=int, default=2, help="首尾裁切的秒数 (默认: 2)")
    
    args = parser.parse_args()
    
    print(f"[*] 正在加载数据: {args.input_csv}")
    try:
        df = pd.read_csv(args.input_csv)
    except FileNotFoundError:
        print(f"[错误] 找不到文件: {args.input_csv}")
        sys.exit(1)
        
    if 'datetime' not in df.columns or 'ied' not in df.columns:
        print("[错误] CSV 必须包含 'datetime' 和 'ied' 列！")
        sys.exit(1)
        
    df['datetime'] = pd.to_datetime(df['datetime'])
    # 剔除原始数据中彻底为空的行，防止影响时间差计算
    df = df.dropna(subset=['datetime', 'ied']).copy()
    
    # 核心逻辑
    df_results = process_robust_ppg(df, sampling_rate=args.sr, trim_seconds=args.trim)
    
    if df_results.empty:
        print("[!] 警告: 未能提取到任何有效数据，请检查信号质量或裁切参数。")
        sys.exit(1)
        
    # 按时间戳排序并保存
    df_results = df_results.sort_values(by='datetime').reset_index(drop=True)
    df_results.to_csv(args.output, index=False)
    
    peaks_count = len(df_results[df_results['point_type'] == 'Peak'])
    troughs_count = len(df_results[df_results['point_type'] == 'Trough'])
    
    print(f"[*] 处理完成！共提取波峰 {peaks_count} 个，波谷 {troughs_count} 个。")
    print(f"[*] 结果已保存至: {args.output}")

if __name__ == "__main__":
    main()