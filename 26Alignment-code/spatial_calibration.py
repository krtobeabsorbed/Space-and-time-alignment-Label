import numpy as np
import cv2
import os

# ==========================================
# 1. 智能标定配置
# ==========================================
PAIRS = [
    ('radar_track1.txt', 'camera_track1.txt', 'calib_r1_c1.npz'),
    ('radar_track1.txt', 'camera_track2.txt', 'calib_r1_c2.npz'), # 之前失败的那个
    ('radar_track1.txt', 'camera_track3.txt', 'calib_r1_c3.npz'),
    ('radar_track1.txt', 'camera_track4.txt', 'calib_r1_c4.npz'),
]

# 海康相机参数
W, H = 3200, 1800
F_mm = 4.0 
Sensor_W_mm = 5.9
fx = F_mm * W / Sensor_W_mm
fy = fx 
cx = W / 2
cy = H / 2

K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((4,1)) 

RADAR_FPS = 16.13
VIDEO_FPS = 30.0

def try_calibrate(object_points, image_points, description):
    """ 尝试一种特定的坐标变换，返回 (成功否, 误差, rvec, tvec) """
    if len(object_points) < 6:
        return False, 99999, None, None

    # 使用 RANSAC
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points, 
        image_points, 
        K, 
        dist_coeffs,
        iterationsCount=200,      # 增加迭代次数
        reprojectionError=15.0    # 稍微严格一点
    )
    
    if success:
        projected_pts, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist_coeffs)
        projected_pts = projected_pts.reshape(-1, 2)
        error = np.linalg.norm(image_points - projected_pts, axis=1).mean()
        return True, error, rvec, tvec
    else:
        return False, 99999, None, None

def solve_smart_pair(radar_file, cam_file, out_name):
    print(f"\n>>> 正在处理: {radar_file} <---> {cam_file}")
    
    if not os.path.exists(radar_file) or not os.path.exists(cam_file):
        print("  错误: 文件不存在，跳过。")
        return

    # 1. 读取原始数据
    try:
        r_raw = np.loadtxt(radar_file)
        c_raw = np.loadtxt(cam_file, skiprows=1)
    except:
        print("  读取错误，跳过。")
        return

    # 2. 原始匹配 (只做时间对齐，不做坐标变换)
    raw_matches = [] # 存 [rx, ry, u, v]
    
    for row in c_raw:
        vid_idx = int(row[0])
        u, v = row[1], row[2]
        t = vid_idx / VIDEO_FPS
        rad_idx = int(t * RADAR_FPS)
        
        if rad_idx < len(r_raw):
            r_pt = r_raw[rad_idx]
            if not np.isnan(r_pt[0]) and not np.all(r_pt==0):
                raw_matches.append([r_pt[0], r_pt[1], u, v]) # 只取 x, y (忽略z=0)
    
    if len(raw_matches) < 6:
        print("  匹配点过少 (<6)，跳过。")
        return

    raw_matches = np.array(raw_matches)
    rx = raw_matches[:, 0]
    ry = raw_matches[:, 1]
    uv = raw_matches[:, 2:]

    # 3. 定义 4 种坐标变换策略
    # 格式: 名字, lambda函数生成3D点
    strategies = [
        ("原始 [x, y, 0]",    lambda x, y: np.column_stack((x, y, np.zeros_like(x)))),
        ("深度修正 [x, 0, y]", lambda x, y: np.column_stack((x, np.zeros_like(x), y))),
        ("X反转 [ -x, 0, y]",  lambda x, y: np.column_stack((-x, np.zeros_like(x), y))),
        ("Y反转 [x, 0, -y]",   lambda x, y: np.column_stack((x, np.zeros_like(x), -y))),
    ]

    best_error = 99999
    best_res = None
    best_name = ""

    print(f"  匹配到 {len(raw_matches)} 个点，开始尝试 4 种坐标变换...")

    for name, transform_func in strategies:
        # 生成 3D 点
        obj_pts = transform_func(rx, ry).astype(np.float32)
        img_pts = uv.astype(np.float32)
        
        success, error, rvec, tvec = try_calibrate(obj_pts, img_pts, name)
        
        if success:
            print(f"    - 尝试 {name}: 误差 = {error:.2f} px")
            if error < best_error:
                best_error = error
                best_res = (rvec, tvec)
                best_name = name
        else:
            print(f"    - 尝试 {name}: PnP 求解失败")

    # 4. 保存最佳结果
    if best_res:
        rvec, tvec = best_res
        R, _ = cv2.Rodrigues(rvec)
        
        print("-" * 20)
        print(f"  [胜出策略] {best_name}")
        print(f"  [最终误差] {best_error:.2f} 像素")
        print(f"  [平移向量] {tvec.flatten()}")
        
        # 这是一个简单的物理检查
        if np.abs(tvec[2]) > 20: # 如果Z轴平移超过20米，通常是错的
            print("  [警告] 平移向量 Z 值过大，可能仍有物理异常，请检查数据。")
            
        np.savez(out_name, R=R, T=tvec, K=K, error=best_error, strategy=best_name)
        print(f"  已保存至 {out_name}")
    else:
        print("  所有策略均失败，无法标定该组相机。")

if __name__ == "__main__":
    print("开始智能标定...")
    for r, c, o in PAIRS:
        solve_smart_pair(r, c, o)