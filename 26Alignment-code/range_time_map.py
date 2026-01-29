import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 核心配置
# ==========================================
CONFIG = {
    # 文件路径
    'file_path': 'adc_data_Full_p1.bin', 
    
    # 核心解析参数
    'num_adc_samples': 256,      # # of ADC Samples
    'num_chirps_per_frame': 384, # 128 Loops * 3 TX Antennas = 384
    'num_rx_antennas': 4,        # IWR6843/ISK 板载固定为 4 个接收天线
    
    # 辅助参数
    'num_tx_antennas': 3,        # 实际上上面的 384 已经隐含了这个信息，但留着备用
    'fps': 16.13,                # 1000ms / 62ms
    'range_resolution': 0.044    # 默认值，不影响能否出图
}

def process_radar_data(config):
    print(f"正在读取文件: {config['file_path']} ...")
    
    # 1. 读取二进制文件
    # DCA1000 保存的是 int16 格式
    try:
        raw_data = np.fromfile(config['file_path'], dtype=np.int16)
    except FileNotFoundError:
        print("错误：找不到文件，请检查路径。")
        return

    # 2. 处理复数 (I/Q)
    # TI DCA1000 的原始数据通常是实部虚部交替
    # 格式转换: I, Q, I, Q ... -> I + jQ
    # 注意：某些旧版本固件可能是 Q, I 顺序，但这不影响我们看能量峰值
    raw_data = raw_data.astype(np.float32)
    complex_data = raw_data[0::2] + 1j * raw_data[1::2]
    
    print(f"原始数据读取完毕，总样本数 (Complex): {len(complex_data)}")

    # 3. 计算这一分钟总共有多少帧 (Frame)
    # 单帧大小 = Samples * Chirps (包含Tx和Rx的总Chirps) * RX
    # 注意：这里的 chirps_per_frame 通常指 loops * tx_antennas
    frame_size = config['num_adc_samples'] * config['num_chirps_per_frame'] * config['num_rx_antennas']
    
    num_frames = len(complex_data) // frame_size
    print(f"检测到完整帧数: {num_frames} 帧 (约 {num_frames/config['fps']:.1f} 秒)")
    
    if num_frames == 0:
        print("错误：数据量不足一帧，请检查配置参数是否设置过大。")
        return

    # 截取完整帧的数据
    complex_data = complex_data[:num_frames * frame_size]

    # 4. 重塑矩阵形状
    # 目标形状: [Frames, Chirps(all), RX, Samples]
    try:
        data_cube = complex_data.reshape(num_frames, 
                                         config['num_chirps_per_frame'], 
                                         config['num_rx_antennas'], 
                                         config['num_adc_samples'])
    except ValueError as e:
        print(f"Reshape 失败: {e}")
        print("这通常是因为 CONFIG 中的参数(Samples/Chirps/RX)与实际文件不匹配。")
        return

    # 5. Range-FFT 处理
    print("正在执行 Range-FFT ...")
    
    # 对最后一个维度 (Samples) 做 FFT
    range_fft = np.fft.fft(data_cube, axis=-1)
    
    # 这是一个1分钟的长视频，为了方便看清楚，我们只取前一半距离
    # 因为通常室内实验只关心前几米，后面的都是高频噪声
    range_bins_to_keep = config['num_adc_samples'] // 2
    range_fft = range_fft[:, :, :, :range_bins_to_keep]

    # 6. 生成 Range-Time Map (非相干积累)
    # 也就是把 chirps 和 antennas 维度的能量加起来，只保留 [Time, Range]
    # 取模 -> 也就是信号强度
    radar_magnitude = np.abs(range_fft)
    
    # 对 Chirps 和 RX 维度求平均
    # 形状变化: [Frames, Chirps, RX, Samples] -> [Frames, Samples]
    range_time_map = np.mean(radar_magnitude, axis=(1, 2))
    
    # 转为对数坐标 (dB)，让微弱信号也能看清
    range_time_map_log = 20 * np.log10(range_time_map + 1e-9) # 加微小值防止log0
    
    # 为了绘图，我们需要转置: X轴是时间，Y轴是距离
    range_time_map_log = range_time_map_log.T

    # 7. 绘图
    print("正在绘图...")
    plt.figure(figsize=(12, 6))
    
    # 翻转Y轴，让0米在最下面
    # extent参数设置坐标轴刻度: [开始时间, 结束时间, 开始距离, 结束距离]
    max_time = num_frames / config['fps']
    max_range = range_bins_to_keep * config['range_resolution']
    
    plt.imshow(range_time_map_log, aspect='auto', cmap='jet', origin='lower',
               extent=[0, max_time, 0, max_range])
    
    plt.colorbar(label='Signal Strength (dB)')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Range (Meters)')
    plt.title(f'Radar Range-Time Heatmap ({config["file_path"]})')
    
    # 标记: 这里可以帮你大致看清哪里是"入场"
    plt.grid(False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    process_radar_data(CONFIG)