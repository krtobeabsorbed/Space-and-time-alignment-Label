import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# ==========================================
# 1. 雷达配置 (和之前一样)
# ==========================================
CONFIG = {
    'file_path': 'adc_data_Full_p1_aligned.bin', 
    'num_adc_samples': 256,
    'num_chirps_per_frame': 384,
    'num_rx_antennas': 4,
    'num_tx_antennas': 3,
    'fps': 16.13,
    'range_resolution': 0.044,
    'max_range': 5.0, # 只需要5米内的数据
}

def generate_point_cloud(config):
    print("正在处理雷达点云...")  
    # 读取数据 (同前)
    raw_data = np.fromfile(config['file_path'], dtype=np.int16)
    raw_data = raw_data.astype(np.float32)
    complex_data = raw_data[0::2] + 1j * raw_data[1::2]

    frame_size = config['num_adc_samples'] * config['num_chirps_per_frame'] * config['num_rx_antennas']
    num_frames = len(complex_data) // frame_size
    complex_data = complex_data[:num_frames * frame_size]

    # Reshape
    data_cube = complex_data.reshape(num_frames, config['num_chirps_per_frame'], 
                                     config['num_rx_antennas'], config['num_adc_samples'])
    
    # 分离 TX (384 chirps -> 128 loops * 3 TX)
    num_loops = config['num_chirps_per_frame'] // config['num_tx_antennas']
    data_cube = data_cube.reshape(num_frames, num_loops, config['num_tx_antennas'], 
                                  config['num_rx_antennas'], config['num_adc_samples'])

    # === 极简版点云生成 (Range-Azimuth Heatmap Peak Finding) ===
    # 为了简化计算，我们这里只做 Range-FFT 和 Angle-FFT (Capon/Music太慢了)
    
    # 1. 虚拟孔径数组构建 (Virtual Antenna Array)
    # TDM-MIMO: 3TX * 4RX = 12 Virtual Antennas
    # 需要根据天线布局拼接。通常 IWR6843 是水平排列
    # 简单拼接: [TX0RX0...TX0RX3, TX1RX0...TX1RX3, ...]
    
    radar_centroids = [] # 存储每一帧的人体坐标 [x, y, z]

    for frame_idx in range(num_frames):
        # 拿一帧数据
        frame_data = data_cube[frame_idx] # [Loops, TX, RX, Samples]
        
        # 2D FFT (Range-Doppler)
        # Range FFT
        range_fft = np.fft.fft(frame_data, axis=-1)
        
        # 咱们取多普勒维度的0频附近? 不，直接非相干积累看能量
        # 这里的处理为了速度，我们简化为：直接在 Virtual Array 上做 Angle FFT
        
        # 构建虚拟天线数据 [12, Samples] (取第0个Loop，或者平均所有Loop)
        virtual_ant_data = range_fft.mean(axis=0) # [TX, RX, Samples]
        virtual_ant_data = virtual_ant_data.reshape(-1, config['num_adc_samples']) # [12, Samples]
        
        # Angle FFT (沿着天线维度)
        angle_fft = np.fft.fft(virtual_ant_data, axis=0, n=64) # 补零到64点提高分辨率
        angle_fft = np.fft.fftshift(angle_fft, axes=0)
        
        # 得到 [Angle, Range] 热力图
        heatmap = np.abs(angle_fft).T # [Range, Angle]
        
        # 阈值过滤 (CFAR的简化版)
        threshold = np.percentile(heatmap, 99.5) # 只取最亮的0.5%的点
        peaks = np.argwhere(heatmap > threshold)
        
        points_xyz = []
        for r_idx, a_idx in peaks:
            r = r_idx * config['range_resolution']
            if r > config['max_range'] or r < 0.5: continue # 过滤太远或太近
            
            # 角度索引转弧度 (-pi/2 到 pi/2)
            # 64点FFT，中间是0度
            angle = (a_idx - 32) / 64 * np.pi 
            
            x = r * np.sin(angle)
            y = r * np.cos(angle)
            z = 0 # 2D雷达假设z=0，如果是3D雷达需要Elevation FFT
            
            points_xyz.append([x, y, z])
        
        points_xyz = np.array(points_xyz)
        
        # 聚类找人 (DBSCAN)
        if len(points_xyz) > 0:
            clustering = DBSCAN(eps=0.5, min_samples=3).fit(points_xyz)
            # 假设点最多的那个类是人
            labels = clustering.labels_
            if len(set(labels)) > 1 or (len(set(labels))==1 and -1 not in labels):
                # 找最大的簇
                unique_labels, counts = np.unique(labels[labels>=0], return_counts=True)
                if len(unique_labels) > 0:
                    dominant_label = unique_labels[np.argmax(counts)]
                    person_points = points_xyz[labels == dominant_label]
                    centroid = np.mean(person_points, axis=0)
                    radar_centroids.append(centroid)
                else:
                    radar_centroids.append([np.nan, np.nan, np.nan])
            else:
                radar_centroids.append([np.nan, np.nan, np.nan])
        else:
            radar_centroids.append([np.nan, np.nan, np.nan])

        if frame_idx % 50 == 0:
            print(f"处理进度: {frame_idx}/{num_frames}")

    # 保存结果
    radar_centroids = np.array(radar_centroids)
    np.savetxt("radar_track.txt", radar_centroids, fmt="%.4f")
    print("雷达轨迹已保存到 radar_track.txt")
    
    # 画个图看看轨迹对不对
    plt.figure()
    plt.plot(radar_centroids[:, 0], radar_centroids[:, 1], '.-')
    plt.title("Extracted Radar Trajectory (Top View)")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    generate_point_cloud(CONFIG)