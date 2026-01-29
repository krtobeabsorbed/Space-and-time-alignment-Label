import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 配置参数
# ==========================================
CONFIG = {
    'file_path': 'adc_data_Full_p1.bin',  # <--- 文件名
    'num_adc_samples': 256,
    'num_chirps_per_frame': 384,
    'num_rx_antennas': 4,
    'num_tx_antennas': 3,
    'fps': 16.13,
}

def generate_doppler_time_map(config):
    print(f"正在处理: {config['file_path']} ...")
    
    # 1. 读取与预处理
    try:
        raw_data = np.fromfile(config['file_path'], dtype=np.int16)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {config['file_path']}")
        return

    raw_data = raw_data.astype(np.float32)
    complex_data = raw_data[0::2] + 1j * raw_data[1::2]
    
    frame_size = config['num_adc_samples'] * config['num_chirps_per_frame'] * config['num_rx_antennas']
    num_frames = len(complex_data) // frame_size
    
    if num_frames == 0:
        print("错误: 数据不足一帧")
        return

    complex_data = complex_data[:num_frames * frame_size]
    
    # Reshape
    data_cube = complex_data.reshape(num_frames, 
                                     config['num_chirps_per_frame'], 
                                     config['num_rx_antennas'], 
                                     config['num_adc_samples'])

    # 2. 分离 TX 天线
    num_loops = config['num_chirps_per_frame'] // config['num_tx_antennas']
    
    # Reshape以分离TX
    data_cube = data_cube.reshape(num_frames, num_loops, config['num_tx_antennas'], 
                                  config['num_rx_antennas'], config['num_adc_samples'])
    
    # 取 TX0, RX0 
    radar_data_1tx_1rx = data_cube[:, :, 0, 0, :] # Shape: [Frames, Loops, Samples]

    # 3. 信号处理
    print("执行 2D-FFT (Range -> Doppler)...")
    
    # 3.1 Range FFT
    range_fft = np.fft.fft(radar_data_1tx_1rx, axis=-1)
    
    # 去掉静止杂波 (Clutter Removal)
    range_fft = range_fft - np.mean(range_fft, axis=0, keepdims=True)

    # 3.2 Doppler FFT
    doppler_fft = np.fft.fft(range_fft, axis=1)
    
    # Shift zero frequency to center (这里修正了参数名为 axes)
    doppler_fft = np.fft.fftshift(doppler_fft, axes=1)

    # 4. 生成 Micro-Doppler 图
    # 只取前几米的有效距离
    valid_range_bins = config['num_adc_samples'] // 2
    
    # 取模并对 Range 维度求和
    mag_data = np.abs(doppler_fft[:, :, :valid_range_bins])
    time_doppler_map = np.sum(mag_data, axis=-1)
    
    # 转置绘图
    time_doppler_map = time_doppler_map.T 
    
    # Log scale
    time_doppler_map = 20 * np.log10(time_doppler_map + 1e-9)

    # 5. 绘图
    print("正在绘图...")
    plt.figure(figsize=(12, 6))
    
    max_time = num_frames / config['fps']
    
    plt.imshow(time_doppler_map, aspect='auto', cmap='jet', origin='lower',
               extent=[0, max_time, -num_loops/2, num_loops/2])
    
    plt.title(f'Micro-Doppler Spectrogram ({config["file_path"]})')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Doppler Velocity (Bin Index)')
    plt.colorbar(label='Magnitude (dB)')
    
    plt.axhline(0, color='white', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    generate_doppler_time_map(CONFIG)