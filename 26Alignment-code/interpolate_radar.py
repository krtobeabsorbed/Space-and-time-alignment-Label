import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# ==========================================
# 1. 雷达配置 (和之前一样)
#  FFT处理点云，同时生成俯视图对照，以及track.txt痕迹文件
# ==========================================
INPUT_FILE = 'radar_track2_clean.txt'
OUTPUT_FILE = 'radar_track2_final_smooth.txt'

def fill_gaps():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到文件: {INPUT_FILE}")
        return

    print(f"正在读取 {INPUT_FILE} ...")
    # 读取数据，保留 NaN
    data = np.loadtxt(INPUT_FILE)
    
    # 转换为 Pandas DataFrame 以便处理
    df = pd.DataFrame(data, columns=['x', 'y', 'z'])
    
    print(f"原始数据行数: {len(df)}")
    print(f"空值(NaN)行数: {df['x'].isna().sum()}")

    # 1. 线性插值 (Linear Interpolation)
    # limit=50: 如果中间断了超过 50 帧 (约3秒)，就不补了（避免瞎猜）
    # direction='both': 头尾缺的也尽量补一下
    df_filled = df.interpolate(method='linear', limit=50, limit_direction='both')
    
    # 2. 对于实在补不上的（比如开头结尾太长），用 0 填充或者保持上一帧
    df_filled = df_filled.fillna(method='bfill').fillna(method='ffill')
    
    # 3. 平滑处理 (可选，让轨迹更圆润)
    # 使用移动平均窗口
    df_smooth = df_filled.rolling(window=5, min_periods=1, center=True).mean()

    # 保存
    np.savetxt(OUTPUT_FILE, df_smooth.values, fmt='%.4f')
    
    print("-" * 30)
    print("✅ 插值补全完成！")
    print(f"已生成平滑后的轨迹文件: {OUTPUT_FILE}")
    print("请在 interactive_tuner 和 generate_fusion 中使用这个新文件。")
    print("-" * 30)
    # ==========================================
    # 4. 可视化对比 (新增部分)
    # ==========================================
    plt.figure(figsize=(14, 6))

    # 左图：修复前 (断断续续)
    plt.subplot(1, 2, 1)
    plt.plot(df['x'], df['y'], 'r.', markersize=2, label='Original Points')  # 用点画，能看出断裂
    # 为了连线看清楚顺序，也可以画淡淡的线
    plt.plot(df['x'], df['y'], 'r-', alpha=0.2)
    plt.title(f"Before: Broken Trajectory ({len(df)} pts)")
    plt.xlabel("X (Left/Right)")
    plt.ylabel("Y (Depth)")
    plt.grid(True)
    plt.axis('equal')  # 保持比例

    # 右图：修复后 (丝滑连接)
    plt.subplot(1, 2, 2)
    plt.plot(df_smooth['x'], df_smooth['y'], 'b-', linewidth=2, label='Smoothed')  # 用线画，看连贯性
    plt.plot(df_smooth['x'], df_smooth['y'], 'b.', markersize=4)  # 显示补出来的点
    plt.title(f"After: Interpolated & Smoothed")
    plt.xlabel("X (Left/Right)")
    plt.ylabel("Y (Depth)")
    plt.grid(True)
    plt.axis('equal')

    plt.suptitle(f"Effect of Interpolation: {INPUT_FILE}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    fill_gaps()