# PPG 数据处理步骤

## 1. 预处理 (Preprocessing)

- [ ] 数据加载与格式检查
- [ ] 缺失值检测与处理
- [ ] 异常值/离群点检测与处理
- [ ] 时间戳解析与同步

## 2. 信号滤波 (Filtering)

- [ ] 去除直流分量（DC offset removal）
- [ ] 带通滤波（Bandpass Filter）
  - 典型范围：0.5-4 Hz（对应 30-240 BPM）
  - 去除高频噪声和低频漂移
- [ ] 工频干扰去除（50/60 Hz Notch Filter）
- [ ] 运动伪影去除（如有IMU数据，可使用加速度信号辅助）

## 3. 信号质量评估 (Signal Quality Index)

- [ ] 信噪比（SNR）计算
- [ ] 信号幅度评估
- [ ] 周期一致性检测

## 4. 脉搏波检测 (Pulse Wave Detection)

- [ ] 峰值检测（Peak Detection）
  - 寻找 R 峰/收缩峰位置
- [ ] 谷值检测（Dicrotic Notch / Diastolic Peak）
- [ ] 周期分割（Segment into individual heartbeats）

## 5. 特征提取 (Feature Extraction)

- [ ] 时域特征
  - [ ] 峰值幅度、谷值幅度
  - [ ] 脉搏波宽度（PW, Pulse Width）
  - [ ] 上升时间、下降时间
  - [ ] 收缩期/舒张期比率
- [ ] 频域特征
  - [ ] HRV（心率变异性）分析
  - [ ] 功率谱密度

## 6. 后处理与验证 (Post-processing)

- [ ] 心率计算与验证
- [ ] 异常心跳标记
- [ ] 结果可视化检查
