import numpy as np
import cv2
import os

# ==========================================
# 手动标定配置 (这里全是我们的猜测值!)
# ==========================================
OUTPUT_NPZ = 'calib_manual_hack.npz'

# 海康相机内参 (保持不变)
W, H = 3200, 1800
F_mm = 4.0
Sensor_W_mm = 5.9
fx = F_mm * W / Sensor_W_mm
fy = fx
cx = W / 2
cy = H / 2
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

def create_manual_calibration():
    print("开始生成手动硬编码标定参数...")

    # --- 1. 定义平移向量 T (相机相对于雷达的位置) ---
    # 坐标系参考：雷达的 [x(右), y(前), z(上)]
    # 假设：相机在雷达正上方 1.5米，且往后退了 0.5米
    # T = [x_shift, y_shift, z_shift]
    # 注意：这里需要在 OpenCV 相机坐标系下定义 T。
    # OpenCV 相机系: X右, Y下, Z前
    # 这是一个非常粗略的估计，目的是让点散开
    tvec = np.array([0.0, 1.5, 0.5], dtype=np.float32) 
    print(f"手动设置平移 T: {tvec}")

    # --- 2. 定义旋转矩阵 R (相机的姿态) ---
    # 我们需要一个让相机“向下看”的旋转。
    # 绕 X 轴旋转一个正角度 (Pitch Down)
    pitch_angle_deg = 25.0 # 向下看 25 度
    pitch_angle_rad = np.deg2rad(pitch_angle_deg)

    # 旋转向量 (绕 X 轴旋转)
    rvec = np.array([pitch_angle_rad, 0, 0], dtype=np.float32)

    # 转为矩阵
    R, _ = cv2.Rodrigues(rvec)
    print(f"手动设置旋转 R (俯拍 {pitch_angle_deg} 度):\n{R}")

    # --- 3. 保存 ---
    np.savez(OUTPUT_NPZ, R=R, T=tvec, K=K)
    print(f"\n已生成手动标定文件: {OUTPUT_NPZ}")
    print("请使用 verify_calibration_with_offset.py 加载此文件进行验证。")

if __name__ == "__main__":
    create_manual_calibration()