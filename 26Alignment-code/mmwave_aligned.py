import numpy as np
import os

# ==========================================
# 裁剪配置 (截取 3.0s ~ 59.0s)
# ==========================================
TRIM_CONFIG = {
    # 截取起始时间 (秒)
    'start_time': 3.0,

    # 截取结束时间 (秒)
    'end_time': 59.0,

    # 文件路径
    'p1_file': 'adc_data_Full_p1.bin',
    'p2_file': 'adc_data_Full_p2.bin',
    
    # 雷达参数 (必须正确!)
    'num_adc_samples': 256,
    'num_chirps_per_frame': 384,
    'num_rx_antennas': 4,
    'fps': 16.13
}

def trim_bin_exact_range(file_path, config):
    start_t = config['start_time']
    end_t = config['end_time']
    
    # 检查参数合理性
    if start_t >= end_t:
        print("错误: 结束时间必须大于开始时间")
        return

    print(f"正在处理 {file_path} ...")
    print(f"  - 目标区间: {start_t}s ~ {end_t}s")
    print(f"  - 预期时长: {end_t - start_t:.2f}s")

    # 1. 计算单帧大小 (bytes)
    # Sample(4 bytes) * Samples * Chirps * RX
    frame_size_bytes = config['num_adc_samples'] * config['num_chirps_per_frame'] * config['num_rx_antennas'] * 4
    
    # 2. 计算起始和结束的帧索引
    # int() 向下取整，确保从第3秒开始
    start_frame_idx = int(start_t * config['fps'])
    end_frame_idx = int(end_t * config['fps'])
    
    # 计算需要读取的总帧数
    frames_to_read = end_frame_idx - start_frame_idx
    
    # 3. 计算字节偏移量
    start_byte_offset = start_frame_idx * frame_size_bytes
    bytes_to_read = frames_to_read * frame_size_bytes
    
    print(f"  - 起始帧: {start_frame_idx}, 结束帧: {end_frame_idx}")
    print(f"  - 总共截取: {frames_to_read} 帧")
    print(f"  - 数据量: {bytes_to_read / 1024 / 1024:.2f} MB")

    # 4. 读取并写入
    new_filename = file_path.replace('.bin', '_3s_to_59s.bin')
    
    try:
        with open(file_path, 'rb') as f_in:
            # 跳到开始位置
            f_in.seek(start_byte_offset)
            
            # 只读取指定长度的数据
            data = f_in.read(bytes_to_read)
            
            # 检查读取是否完整
            if len(data) != bytes_to_read:
                print(f"  警告: 文件末尾数据不足! 实际读取了 {len(data)} 字节 (预期 {bytes_to_read})")
                print(f"  可能原文件总时长不足 {end_t} 秒。")

        with open(new_filename, 'wb') as f_out:
            f_out.write(data)
            
        print(f"  - 成功! 已保存: {new_filename}")
        
    except FileNotFoundError:
        print(f"  错误: 找不到文件 {file_path}")
    except Exception as e:
        print(f"  发生错误: {e}")
        
    print("-" * 30)

if __name__ == "__main__":
    # 处理 P1
    trim_bin_exact_range(TRIM_CONFIG['p1_file'], TRIM_CONFIG)
                  
    # 处理 P2
    trim_bin_exact_range(TRIM_CONFIG['p2_file'], TRIM_CONFIG)
                  
    print("全部完成！新文件的第0秒对应原始数据的第3秒。")