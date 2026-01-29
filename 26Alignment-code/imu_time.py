import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 配置
# ==========================================
CONFIG = {
    # Phyphox 数据路径
    'acc_file': 'Accelerometer.csv', 
    
    # 限制显示前 N 秒 (方便看清起步动作)
    'duration_limit': 60.0 
}

def plot_imu_entry(config):
    print(f"正在读取: {config['acc_file']} ...")
    
    try:
        df = pd.read_csv(config['acc_file'])
    except Exception as e:
        print(f"读取失败: {e}")
        return

    # 1. 自动匹配列名 (根据你的截图修正)
    cols = df.columns
    
    # 找到时间列 (包含 'Time')
    try:
        time_col = [c for c in cols if 'Time' in c][0]
    except IndexError:
        print("错误: 找不到时间列 (Time)")
        return

    # 找到 X, Y, Z 列
    # 逻辑: 包含 'X' 且不包含 'Time'，依此类推
    try:
        acc_x = [c for c in cols if 'X' in c.upper() and 'TIME' not in c.upper()][0]
        acc_y = [c for c in cols if 'Y' in c.upper() and 'TIME' not in c.upper()][0]
        acc_z = [c for c in cols if 'Z' in c.upper() and 'TIME' not in c.upper()][0]
        
        print(f"已识别列名: Time='{time_col}', X='{acc_x}', Y='{acc_y}', Z='{acc_z}'")
        
    except IndexError:
        print("错误: 无法识别 X/Y/Z 列，请检查 CSV 表头。")
        print(f"当前表头: {cols.tolist()}")
        return

    # 2. 截取时间
    if config['duration_limit']:
        df = df[df[time_col] <= config['duration_limit']]

    # 3. 计算合加速度 (Resultant Acceleration)
    # 你的数据看起来是带重力的 (Z轴约为9.8或-9.8)
    # 公式: sqrt(x^2 + y^2 + z^2)
    x_val = df[acc_x]
    y_val = df[acc_y]
    z_val = df[acc_z]
    
    magnitude = np.sqrt(x_val**2 + y_val**2 + z_val**2)

    # 4. 绘图
    plt.figure(figsize=(12, 6))
    
    plt.plot(df[time_col], magnitude, label='Resultant Acceleration', linewidth=1, color='blue')
    
    plt.title('IMU Acceleration Magnitude (Find the Entry Step)')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s^2)')
    
    # 画一条 9.8 的参考线 (重力)
    plt.axhline(9.8, color='red', linestyle='--', alpha=0.5, label='Gravity (Static)')
    
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    print("-" * 30)
    print("【操作指南】")
    print("1. 图中红线是重力 (9.8)，静止时曲线应该贴着红线走。")
    print("2. 找到曲线第一次出现【大幅度锯齿状波动】的时间点。")
    print("   这就是被试者【开始走路/进场】的时刻 (T_imu_entry)。")
    print("-" * 30)
    
    plt.show()

if __name__ == "__main__":
    plot_imu_entry(CONFIG)