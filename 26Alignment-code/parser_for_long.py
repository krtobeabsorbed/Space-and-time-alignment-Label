import numpy as np
import matplotlib.pyplot as plt

# --- 核心配置 ---
# 文件路径
file_name = 'adc_data_0.bin'

# 硬件参数配置 (根据 mmWave Studio 截图确认)
num_adc_samples = 256    # 
num_rx = 4               # 
# 注意：基于 TDM-MIMO 配置 (3Tx * 128 Loops)，原始数据中的 Chirp 维度应为 384 
num_chirps_raw = 384
# 根据计算，该文件中包含 682 个完整帧
num_frames = 682

print(f"正在处理文件: {file_name}")
print(f"配置参数: Frames={num_frames}, Chirps(Raw)={num_chirps_raw}, Rx={num_rx}, Samples={num_adc_samples}")

# --- 1. 读取和预处理 ---
# 读取二进制文件 (16位有符号整数)
adc_data_raw = np.fromfile(file_name, dtype=np.int16)

# 组合成复数 (I + jQ)
adc_data_raw = adc_data_raw.reshape(-1, 2)
adc_data_complex = adc_data_raw[:, 0] + 1j * adc_data_raw[:, 1]
print(f"文件中读取到的总复数点数: {len(adc_data_complex)}")

# --- [关键步骤] 数据截取 ---
# 计算 682 帧所需的理论复数点总数
expected_points_total = num_frames * num_chirps_raw * num_rx * num_adc_samples
print(f"预期完整帧的复数点数: {expected_points_total}")

if len(adc_data_complex) >= expected_points_total:
    # 截取掉文件末尾多余的不完整帧数据
    adc_data_complex = adc_data_complex[:expected_points_total]
    print(f"已截取前 {expected_points_total} 个点进行处理。")

    # --- 2. 重新塑形 (Reshape) ---
    try:
        # 原始排列通常为: [帧, Chirp, Rx通道, 采样点]
        adc_data_cube = adc_data_complex.reshape(num_frames, num_chirps_raw, num_rx, num_adc_samples)
        # 转置为更方便的格式: [Rx, Sample, Chirp, Frame]
        adc_data_cube = adc_data_cube.transpose(2, 3, 1, 0)
        
        print("-" * 30)
        print("数据解析成功！数据立方体形状:", adc_data_cube.shape)
        print("-" * 30)

        # --- 3. 可视化验证 (查看第1帧) ---
        # 选择第1帧 (索引0), 第1个Rx天线 (索引0) 的所有Chirp数据
        # 形状为 (采样点 256, Chirp 384)
        frame1_rx1_data = adc_data_cube[0, :, :, 0]

        # Range FFT
        range_fft_data = np.fft.fft(frame1_rx1_data, axis=0)
        # 取模并转换为dB
        range_profile = 20 * np.log10(np.abs(range_fft_data[:num_adc_samples//2, :]))

        # 绘制热力图
        plt.figure(figsize=(10, 6))
        # 注意：这里的横轴是原始的 384 个 Chirp (包含了 3 个 Tx 的交替发射)
        plt.imshow(range_profile, aspect='auto', cmap='jet', origin='lower')
        plt.title(f'Range Profile Map (Rx1, Frame1) - Raw TDM Data')
        plt.xlabel('Raw Chirp Index (0-383)')
        plt.ylabel('Range Bin Index')
        plt.colorbar(label='Magnitude (dB)')
        plt.show()

    except ValueError as e:
        print(f"\n解析失败 (Reshape错误): {e}")
else:
    print(f"\n错误：文件数据不足以构成 {num_frames} 帧。请检查计算或文件。")