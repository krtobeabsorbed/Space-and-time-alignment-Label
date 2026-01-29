import numpy as np
import matplotlib.pyplot as plt

# ============================
# 填入你想检查的一对文件
# ============================
RADAR_FILE = 'radar_track1.txt'   # 或者是 p2
CAM_FILE = 'camera_track1.txt'    # 对应的相机
FPS_RATIO = 30.0 / 16.13  # Video_FPS / Radar_FPS

def check_shape():
    # 1. 读取
    r_data = np.loadtxt(RADAR_FILE) # [x, y, z]
    c_data = np.loadtxt(CAM_FILE, skiprows=1) # [frame, u, v]
    
    # 2. 简单的对齐提取
    matched_r = []
    matched_c = []
    
    for row in c_data:
        frame_cam = row[0]
        idx_r = int((frame_cam / 30.0) * 16.13) # 时间对齐
        
        if idx_r < len(r_data):
            # 只有当雷达数据不为 nan 且不全为0时
            if not np.isnan(r_data[idx_r][0]) and np.linalg.norm(r_data[idx_r]) > 0.1:
                matched_r.append(r_data[idx_r])
                matched_c.append(row[1:]) # u, v

    matched_r = np.array(matched_r)
    matched_c = np.array(matched_c)

    if len(matched_r) == 0:
        print("没有匹配到任何数据！检查文件名或时间对齐。")
        return

    # 3. 画图对比 (归一化到 0-1 之间方便比较形状)
    def normalize(d):
        return (d - d.min(axis=0)) / (d.max(axis=0) - d.min(axis=0) + 1e-6)

    norm_r = normalize(matched_r)
    norm_c = normalize(matched_c)

    plt.figure(figsize=(12, 6))
    
    # 子图1: 雷达轨迹 (Top View)
    plt.subplot(1, 2, 1)
    # 注意：雷达的 X 是左右，Y 是深度
    plt.plot(norm_r[:, 0], norm_r[:, 1], 'b.-', label='Radar Trajectory')
    plt.title(f"Radar Shape ({len(matched_r)} points)")
    plt.xlabel("X (Normalized)")
    plt.ylabel("Y (Normalized)")
    plt.grid(True)
    plt.legend()

    # 子图2: 相机轨迹 (Image View)
    plt.subplot(1, 2, 2)
    # 注意：相机的 u 是左右，v 是上下(深度)
    # 为了视觉上好对比，通常把 v 反转一下，或者只看 u-v 形状
    plt.plot(norm_c[:, 0], norm_c[:, 1], 'r.-', label='Camera Trajectory')
    plt.title("Camera Trajectory Shape")
    plt.xlabel("u (Pixel)")
    plt.ylabel("v (Pixel)")
    plt.gca().invert_yaxis() # 图片坐标系原点在左上，翻转Y轴符合直觉
    plt.grid(True)
    plt.legend()

    plt.suptitle(f"Shape Comparison: {RADAR_FILE} vs {CAM_FILE}")
    plt.show()

if __name__ == "__main__":
    check_shape()