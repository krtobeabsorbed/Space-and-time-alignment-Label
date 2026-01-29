import pandas as pd
import os

# ==========================================
# IMU 裁剪配置 (1.5s 到 57.5s)
# ==========================================
CONFIG = {
    # 裁剪起始时间 (原文件中的秒数)
    'imu_start_time': 1.5,
    
    # 裁剪结束时间 (原文件中的秒数)
    'imu_end_time': 57.5,

    # 文件路径 (同时处理加速度计和陀螺仪)
    'acc_file': 'Accelerometer.csv',
    'gyro_file': 'Gyroscope.csv'
}

def trim_imu_exact(config):
    start_t = config['imu_start_time']
    end_t = config['imu_end_time']
    expected_duration = end_t - start_t
    
    print(f"准备裁剪 IMU 数据...")
    print(f"  - 保留区间: {start_t}s ~ {end_t}s")
    print(f"  - 预期总时长: {expected_duration:.2f}s")
    print("-" * 30)

    files_to_process = [config['acc_file'], config['gyro_file']]

    for file_path in files_to_process:
        if not os.path.exists(file_path):
            print(f"跳过 (文件不存在): {file_path}")
            continue
            
        print(f"正在处理: {file_path} ...")
        
        try:
            # 读取 CSV
            df = pd.read_csv(file_path)
            
            # 找时间列
            cols = df.columns
            # 模糊匹配 'Time' 列 (Phyphox 导出通常叫 'Time (s)')
            time_col = [c for c in cols if 'Time' in c][0]
            
            # === 核心裁剪逻辑 ===
            # 筛选条件: 大于等于 start 且 小于等于 end
            mask = (df[time_col] >= start_t) & (df[time_col] <= end_t)
            df_trimmed = df[mask].copy()
            
            if df_trimmed.empty:
                print(f"  警告: 裁剪后为空！请检查时间范围是否超出了文件总长。")
                continue

            # === 时间归零 ===
            # 让 1.5s 变成 0s
            df_trimmed[time_col] = df_trimmed[time_col] - start_t
            
            # 保存新文件
            new_filename = file_path.replace('.csv', '_aligned_56s.csv')
            df_trimmed.to_csv(new_filename, index=False)
            
            # 打印统计信息
            actual_duration = df_trimmed[time_col].max() - df_trimmed[time_col].min()
            print(f"  - 原数据行数: {len(df)}")
            print(f"  - 裁剪后行数: {len(df_trimmed)}")
            print(f"  - 新文件时长: {actual_duration:.2f}s")
            print(f"  - 已保存: {new_filename}")
            
        except Exception as e:
            print(f"  处理失败: {e}")
            
    print("-" * 30)
    print("全部完成！新文件的第0秒对应原始数据的1.5秒。")

if __name__ == "__main__":
    trim_imu_exact(CONFIG)