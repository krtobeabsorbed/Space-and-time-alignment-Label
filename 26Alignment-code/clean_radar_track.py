import numpy as np
import matplotlib.pyplot as plt

# ============================
# 配置 去除墙壁反射的静止点
# ============================
INPUT_FILE = 'radar_track2.txt'   # 清洗的文件
OUTPUT_FILE = 'radar_track2_clean.txt' # 清洗后的文件

# 容忍半径 (米)
# 如果一个点距离“静止中心”小于这个距离，就把它当垃圾删掉
FILTER_RADIUS = 0.3 

def clean_data():
    print(f"正在读取 {INPUT_FILE} ...")
    data = np.loadtxt(INPUT_FILE)
    
    # 1. 找到“钉子户” (出现频率最高的那个位置)
    # 我们把空间划分成小格子，看哪个格子点最多
    # 简单做法：计算中位数或众数，这里用 Histogram 找峰值
    
    # 仅统计有效点
    valid_data = data[~np.isnan(data).any(axis=1)]
    valid_data = valid_data[np.abs(valid_data[:,0]) > 0.01] # 去掉0,0点
    
    if len(valid_data) == 0:
        print("数据为空！")
        return

    # 寻找聚类中心 (Static Clutter Center)
    # 这里用简单的平均值可能不准，我们假设那个死锁点是最密集的
    # 我们取所有点的平均值作为初始猜测，但如果人一直在动，静止点应该方差很小
    
    # 让我们用一种更暴力的方法：直接看日志里那个 0.24, 4.72
    # 或者自动计算：
    # 计算每个点到所有其他点的距离之和？太慢。
    # 让我们假设日志里出现最多的就是静止点。
    
    # 统计 X 和 Y 的直方图峰值
    hist_x, bins_x = np.histogram(valid_data[:, 0], bins=50)
    peak_x = bins_x[np.argmax(hist_x)]
    
    hist_y, bins_y = np.histogram(valid_data[:, 1], bins=50)
    peak_y = bins_y[np.argmax(hist_y)]
    
    static_center = np.array([peak_x, peak_y])
    print(f"检测到静止杂波中心 (墙壁): X≈{peak_x:.2f}, Y≈{peak_y:.2f}")
    
    # 2. 开始过滤
    cleaned_data = []
    removed_count = 0
    
    for row in data:
        x, y = row[0], row[1]
        
        # 计算到静止中心的距离
        dist = np.sqrt((x - static_center[0])**2 + (y - static_center[1])**2)
        
        if dist < FILTER_RADIUS:
            # 这是墙，删掉 (填 nan)
            cleaned_data.append([np.nan, np.nan, np.nan])
            removed_count += 1
        else:
            # 这是人 (或者是其他噪声，但至少不是那个墙)
            cleaned_data.append(row)
            
    # 3. 保存
    np.savetxt(OUTPUT_FILE, np.array(cleaned_data), fmt='%.4f')
    
    print("-" * 30)
    print(f"清洗完成！")
    print(f"共处理 {len(data)} 帧")
    print(f"删除了 {removed_count} 个静止帧 ({(removed_count/len(data))*100:.1f}%)")
    print(f"结果已保存至: {OUTPUT_FILE}")
    print("-" * 30)
    
    # 4. 画个图对比一下
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original (With Wall)")
    plt.scatter(data[:,0], data[:,1], s=1, alpha=0.5)
    plt.scatter(static_center[0], static_center[1], c='r', marker='x', s=100, label='Clutter')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.title("Cleaned (Person Only)")
    cl_arr = np.array(cleaned_data)
    plt.scatter(cl_arr[:,0], cl_arr[:,1], s=5, c='g', alpha=0.8)
    # plt.xlim(np.min(data[:,0]), np.max(data[:,0]))
    # plt.ylim(np.min(data[:,1]), np.max(data[:,1]))
    plt.xlim(np.nanmin(data[:, 0]), np.nanmax(data[:, 0]))
    plt.ylim(np.nanmin(data[:, 1]), np.nanmax(data[:, 1]))

    plt.show()

if __name__ == "__main__":
    clean_data()