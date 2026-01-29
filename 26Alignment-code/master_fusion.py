import pandas as pd
import numpy as np
import os

# ==========================================
# 1. 文件名配置 (请确保文件名正确)
# ==========================================
# 视频融合数据 (Radar 1 + Cam 1/2/3/4)
CSV_FILES = {
    'C1': 'dataset_fusion_final_r1_c1.csv',
    'C2': 'dataset_fusion_final_r1_c2.csv',
    'C3': 'dataset_fusion_final_r1_c3.csv',
    'C4': 'dataset_fusion_final_r1_c4.csv'
}

# IMU 数据
IMU_ACC_FILE = 'Accelerometer_aligned_56s.csv'
IMU_GYRO_FILE = 'Gyroscope_aligned_56s.csv'

# 输出
OUTPUT_FILE = 'dataset_fusioned.csv'

def main():
    print("🚀 开始最终数据融合...")

    # ------------------------------------------------
    # Step 1: 读取主相机数据 (C1) 作为基准
    # ------------------------------------------------
    if not os.path.exists(CSV_FILES['C1']):
        print(f"❌ 致命错误: 找不到主文件 {CSV_FILES['C1']}")
        return

    print(f"📂 读取主数据 C1: {CSV_FILES['C1']}")
    master_df = pd.read_csv(CSV_FILES['C1'])
    
    # 重命名 C1 的列
    # 原列名: Video_Frame, Radar_Time, Pixel_U, Pixel_V, Real_X, Real_Y, Real_Z
    master_df = master_df.rename(columns={
        'Video_Frame': 'Frame_ID',
        'Radar_Time': 'Timestamp',
        'Pixel_U': 'C1_U',
        'Pixel_V': 'C1_V',
        'Real_X': 'Radar_X',
        'Real_Y': 'Radar_Y',
        'Real_Z': 'Radar_Z'
    })
    
    # ------------------------------------------------
    # Step 2: 融合其他相机 (C2, C3, C4)
    # ------------------------------------------------
    for cam_name in ['C2', 'C3', 'C4']:
        file_path = CSV_FILES[cam_name]
        if os.path.exists(file_path):
            print(f"📂 正在融合 {cam_name}...")
            sub_df = pd.read_csv(file_path)
            
            # 只提取需要的列: Video_Frame, Pixel_U, Pixel_V
            # 假设所有视频是帧对齐的 (Frame ID 一致)
            sub_df = sub_df[['Video_Frame', 'Pixel_U', 'Pixel_V']]
            
            # 重命名
            sub_df = sub_df.rename(columns={
                'Video_Frame': 'Frame_ID',
                'Pixel_U': f'{cam_name}_U',
                'Pixel_V': f'{cam_name}_V'
            })
            
            # 合并到主表
            master_df = pd.merge(master_df, sub_df, on='Frame_ID', how='left')
        else:
            print(f"⚠️ 跳过 {cam_name} (文件不存在)")

    # ------------------------------------------------
    # Step 3: 融合 IMU 数据 (Acc + Gyro)
    # ------------------------------------------------
    if os.path.exists(IMU_ACC_FILE) and os.path.exists(IMU_GYRO_FILE):
        print("📂 处理 IMU 数据...")
        
        # 读取原始 CSV
        acc_df = pd.read_csv(IMU_ACC_FILE)
        gyro_df = pd.read_csv(IMU_GYRO_FILE)
        
        # 重命名 (严格按照你提供的列名)
        # Acc: Time (s), X (m/s^2), Y (m/s^2), Z (m/s^2)
        acc_df = acc_df.rename(columns={
            'Time (s)': 'Time',
            'X (m/s^2)': 'Acc_X',
            'Y (m/s^2)': 'Acc_Y',
            'Z (m/s^2)': 'Acc_Z'
        })
        
        # Gyro: Time (s), X (rad/s), Y (rad/s), Z (rad/s)
        gyro_df = gyro_df.rename(columns={
            'Time (s)': 'Time',
            'X (rad/s)': 'Gyro_X',
            'Y (rad/s)': 'Gyro_Y',
            'Z (rad/s)': 'Gyro_Z'
        })
        
        # 先合并 Acc 和 Gyro (基于时间)
        imu_df = pd.merge(acc_df, gyro_df, on='Time', how='inner')
        
        # 按照时间戳融合到主表 (merge_asof)
        # 必须先排序
        master_df = master_df.sort_values('Timestamp')
        imu_df = imu_df.sort_values('Time')
        
        # 执行最近邻匹配 (tolerance=0.05s)
        master_df = pd.merge_asof(master_df, imu_df, 
                                  left_on='Timestamp', 
                                  right_on='Time', 
                                  direction='nearest', 
                                  tolerance=0.05)
        
        # 删掉多余的 IMU Time 列
        if 'Time' in master_df.columns:
            master_df = master_df.drop(columns=['Time'])
            
        print("✅ IMU 数据融合成功")
    else:
        print("⚠️ 未找到 IMU 文件，跳过融合")

    # ------------------------------------------------
    # Step 4: 保存结果
    # ------------------------------------------------
    # 按帧号排序
    master_df = master_df.sort_values('Frame_ID')
    
    # 调整列顺序 (把重要的放前面)
    cols = ['Frame_ID', 'Timestamp', 
            'Radar_X', 'Radar_Y', 'Radar_Z',
            'C1_U', 'C1_V', 'C2_U', 'C2_V', 'C3_U', 'C3_V', 'C4_U', 'C4_V']
            
    # 把剩下的列 (IMU等) 加到后面
    remaining_cols = [c for c in master_df.columns if c not in cols]
    final_cols = cols + remaining_cols
    
    # 过滤掉不存在的列 (防止报错)
    final_cols = [c for c in final_cols if c in master_df.columns]
    
    master_df = master_df[final_cols]
    
    master_df.to_csv(OUTPUT_FILE, index=False)
    print("-" * 30)
    print(f"🎉 大功告成！总表已生成: {OUTPUT_FILE}")
    print(f"📊 数据行数: {len(master_df)}")
    print(f"📄 包含列名: {master_df.columns.tolist()}")

if __name__ == "__main__":
    main()