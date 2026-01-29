import numpy as np
import matplotlib.pyplot as plt


# 文件路径
file_name = 'adc_data.bin'
# 硬件参数配置
num_adc_samples = 256
num_rx = 4
num_chirps = 128
num_frames = 24

# 1. 读取二进制文件
# 数据是16位有符号整数 (int16)
adc_data_raw = np.fromfile(file_name, dtype=np.int16)

# 2. 将数据组合成复数 (I + jQ)
# 将数组重塑为 (-1, 2)，即每两个点一组
adc_data_raw = adc_data_raw.reshape(-1, 2)
# 第一列是实部(I)，第二列是虚部(Q)
adc_data_complex = adc_data_raw[:, 0] + 1j * adc_data_raw[:, 1]

# 3. 重新塑形矩阵 (Reshape)
# 注意: Python通常是行优先，MATLAB是列优先，reshape的顺序可能需要调整
# 这里假设一种常见的排列: [帧, Chirp, Rx通道, 采样点]
try:
    adc_data_cube = adc_data_complex.reshape(num_frames, num_chirps, num_rx, num_adc_samples)
    # 为了方便后续处理，通常会转置为 [Rx, Sample, Chirp, Frame] 或类似格式
    adc_data_cube = adc_data_cube.transpose(2, 3, 1, 0)

    print("数据解析成功，形状为:", adc_data_cube.shape)
    # 查看第一帧、第一个chirp、第一个Rx通道的前10个采样点
    print(adc_data_cube[0, :10, 0, 0])

    # 获取第 10 帧的所有数据
    frame_10_data = adc_data_cube[:, :, :, 9]
    print("第10帧数据的形状:", frame_10_data.shape) # 输出应为 (4, 256, 128)


    # 选择第1个Rx天线 (索引0), 第1帧 (索引0), 第1个Chirp (索引0)
    chirp_data = adc_data_cube[0, :, 0, 0]
    # 绘制实部和虚部
    plt.figure(figsize=(10, 4))
    plt.plot(np.real(chirp_data), label='Real (I)')
    plt.plot(np.imag(chirp_data), label='Imag (Q)')
    plt.title('Raw ADC Data - Rx1, Frame1, Chirp1')
    plt.xlabel('Sample Index (Fast Time)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 选择第1帧 (索引0), 第1个Rx天线 (索引0) 的所有Chirp数据
    # 形状为 (256, 128) -> (采样点, Chirp)
    frame1_rx1_data = adc_data_cube[0, :, :, 0]
    # 1. 沿着快时间维 (axis=0) 做 FFT
    # 结果是一个 Range-Chirp 图 (也叫 Range-Doppler 的中间状态)
    range_fft_data = np.fft.fft(frame1_rx1_data, axis=0)
    # 2. 取模值 (Magnitude) 来看信号强度，并转换为对数刻度 (dB) 以便观察
    # 通常只看前一半的采样点 (Nyquist采样定理)
    num_samples = range_fft_data.shape[0]
    range_profile = 20 * np.log10(np.abs(range_fft_data[:num_samples//2, :]))
    # 3. 绘制 Range Profile 热力图
    plt.figure(figsize=(8, 6))
    # 使用 imshow 绘制热力图，纵轴是距离(Range Bin)，横轴是Chirp索引
    plt.imshow(range_profile, aspect='auto', cmap='jet', origin='lower')
    plt.title('Range Profile Map (Rx1, Frame1)')
    plt.xlabel('Chirp Index (Slow Time)')
    plt.ylabel('Range Bin Index')
    plt.colorbar(label='Magnitude (dB)')
    plt.show()

except ValueError as e:
    print(f"解析失败: {e}")
    print("请检查您的硬件参数配置是否与文件大小匹配。")
    # 计算预期的数据点总数
    expected_points = num_frames * num_chirps * num_rx * num_adc_samples
    print(f"文件包含复数点数: {len(adc_data_complex)}")
    print(f"预期复数点数: {expected_points}")