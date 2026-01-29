import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 调试配置：填入你刚才报错的那一组文件
# ==========================================
# 比如看看 P2 和 C3 (虽然误差很大，但至少算出来了，适合分析)
RADAR_FILE = 'radar_track2.txt' 
CAM_FILE = 'camera_track3.txt'

# 视频和雷达的帧率
VIDEO_FPS = 30.0
RADAR_FPS = 16.13

def check_data_shape():
    if not os.path.exists(RADAR_FILE) or not os.path.exists(CAM_FILE):
        print("错误：文件找不到")
        return

    # 1. 读取原始数据
    r_data = np.loadtxt(RADAR_FILE) # [x, y, z]
    # 跳过第一行表头读取相机数据
    try:
        c_data = np.loadtxt(CAM_FILE, skiprows=1) # [frame, u, v]
    except:
        c_data = np.loadtxt(CAM_FILE) # 如果没有表头

    # 2. 提取配对点
    matched_rx = []
    matched_ry = []
    matched_u = []
    matched_v = []
    
    print(f"雷达数据行数: {len(r_data)}")
    print(f"相机数据行数: {len(c_data)}")

    for row in c_data:
        frame_id = row[0]
        # 时间对齐
        t = frame_id / VIDEO_FPS
        rad_idx = int(t * RADAR_FPS)
        
        if rad_idx < len(r_data):
            r_pt = r_data[rad_idx]
            # 排除无效点 (雷达经常会在没人的时候输出 0,0,0 或 nan)
            if not np.isnan(r_pt[0]) and (abs(r_pt[0]) > 0.1 or abs(r_pt[1]) > 0.1):
                matched_rx.append(r_pt[0]) # 雷达 X (左右)
                matched_ry.append(r_pt[1]) # 雷达 Y (深度)
                
                matched_u.append(row[1])   # 相机 U (左右)
                matched_v.append(row[2])   # 相机 V (上下)

    if len(matched_rx) < 5:
        print("严重警告：有效匹配点少于 5 个！无法分析形状。")
        print("可能原因：时间没对齐，或者雷达在相机有人那个时间段全是空数据。")
        return

    # 3. 归一化 (让大家都在 0-1 之间，只看形状，不看大小)
    def normalize(arr):
        arr = np.array(arr)
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)

    # 4. 绘图对比
    plt.figure(figsize=(12, 6))

    # 左图：雷达轨迹 (上帝视角)
    plt.subplot(1, 2, 1)
    plt.scatter(matched_rx, matched_ry, c=range(len(matched_rx)), cmap='viridis', marker='o')
    plt.plot(matched_rx, matched_ry, 'k-', alpha=0.3)
    plt.title(f"Radar Shape (Top View)\nN={len(matched_rx)}")
    plt.xlabel("Radar X (Left/Right)")
    plt.ylabel("Radar Y (Range/Depth)")
    plt.grid(True)
    plt.axis('equal') # 保持比例

    # 右图：相机轨迹 (图像视角)
    plt.subplot(1, 2, 2)
    # 注意：相机的 V 轴通常要翻转一下才符合直觉 (因为图像原点在左上角)
    plt.scatter(matched_u, matched_v, c=range(len(matched_u)), cmap='viridis', marker='x')
    plt.plot(matched_u, matched_v, 'k-', alpha=0.3)
    plt.title(f"Camera Shape (Image View)\nN={len(matched_u)}")
    plt.xlabel("Camera U (Left/Right)")
    plt.ylabel("Camera V (Up/Down)")
    plt.gca().invert_yaxis() # 翻转Y轴
    plt.grid(True)
    
    print("请观察弹出的图像：")
    print("1. 颜色代表时间：从深色(开始)到亮色(结束)。")
    print("2. 两个图的【颜色走势】必须一致 (比如都是从左往右)。")
    print("3. 【形状】必须相似 (比如都是一条斜线，或者都有一个弯)。")
    
    plt.show()

if __name__ == "__main__":
    check_data_shape()