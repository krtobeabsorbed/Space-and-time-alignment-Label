import numpy as np
import cv2
import os

# ==========================================
# 配置
# ==========================================
RADAR_FILE = 'radar_track2_final_smooth.txt'  # 确保是对应的雷达文件
VIDEO_FILE = 'a1.mp4'      # 确保是对应的视频文件
OUTPUT_NPZ = 'calib_r2_a1_tuned.npz'

# 初始猜测值 (可以基于之前的 hack 值)
INIT_X = 0.0
INIT_Y = 1.5  # 高度
INIT_Z = 0.5  # 深度
INIT_PITCH = 25.0 # 度
INIT_YAW = 0.0
INIT_ROLL = 0.0
INIT_TIME_OFFSET = 0
INIT_MIRROR = True # 默认开启镜像试试

# 相机内参
W, H = 3200, 1800
F_mm = 4.0
Sensor_W_mm = 5.9
fx = F_mm * W / Sensor_W_mm
fy = fx
cx = W / 2
cy = H / 2
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((4,1))

RADAR_FPS = 16.13
VIDEO_FPS = 30.0
DISPLAY_WIDTH = 1280

def get_rotation_matrix(pitch, yaw, roll):
    # 将角度转换为弧度
    rx, ry, rz = np.deg2rad(pitch), np.deg2rad(yaw), np.deg2rad(roll)
    
    # 旋转矩阵 Euler Angles
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    
    # R = Rz * Ry * Rx
    return Rz @ Ry @ Rx

def main():
    if not os.path.exists(RADAR_FILE) or not os.path.exists(VIDEO_FILE):
        print("文件缺失！")
        return

    radar_data = np.loadtxt(RADAR_FILE)
    cap = cv2.VideoCapture(VIDEO_FILE)
    
    # 状态变量
    params = {
        'tx': INIT_X, 'ty': INIT_Y, 'tz': INIT_Z,
        'pitch': INIT_PITCH, 'yaw': INIT_YAW, 'roll': INIT_ROLL,
        'time_offset': INIT_TIME_OFFSET,
        'mirror_x': INIT_MIRROR
    }
    
    frame_idx = 0
    paused = False
    
    print(">>> 启动交互式调试 <<<")
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # 循环播放
                frame_idx = 0
                continue
            frame_idx += 1
        else:
            # 暂停时只刷新参数，不读新帧
            pass

        # 1. 计算当前时间对应的雷达帧
        t_vid = frame_idx / VIDEO_FPS
        t_rad = t_vid + params['time_offset']
        rad_idx = int(t_rad * RADAR_FPS)
        
        # 2. 获取当前的 R, T
        R = get_rotation_matrix(params['pitch'], params['yaw'], params['roll'])
        T = np.array([params['tx'], params['ty'], params['tz']], dtype=np.float32)
        rvec, _ = cv2.Rodrigues(R)

        # 3. 准备绘制
        display_frame = frame.copy()
        points_to_draw = []
        
        # 取前后几帧雷达数据
        for i in range(rad_idx - 2, rad_idx + 3):
            if 0 <= i < len(radar_data):
                r_pt = radar_data[i]
                if not np.isnan(r_pt[0]) and (abs(r_pt[0])>0.1 or abs(r_pt[1])>0.1):
                    x_r, y_r = r_pt[0], r_pt[1]
                    
                    # === 坐标映射 ===
                    # 这里的逻辑对应之前的 manual hack
                    # 默认: 雷达X -> 相机X, 雷达Y(深度) -> 相机Z
                    
                    final_x = -x_r if params['mirror_x'] else x_r
                    
                    # 构造物体坐标系下的点 (假设雷达在地板上)
                    # 我们让 Z=0 (地板), Y=Depth (前方)
                    # 相机坐标系惯例: Y向下, Z向前
                    # 这里我们手动把雷达点转到相机系前，先假设一个物体坐标系
                    
                    # 修正策略：直接构造点 [x, 0, y] (x左右, 0高, y深)
                    obj_pt = np.array([final_x, 0, y_r], dtype=np.float32)
                    points_to_draw.append(obj_pt)

        # 4. 投影
        if len(points_to_draw) > 0:
            img_pts, _ = cv2.projectPoints(np.array(points_to_draw), rvec, T, K, dist_coeffs)
            for pt in img_pts.reshape(-1, 2):
                try:
                    cv2.circle(display_frame, (int(pt[0]), int(pt[1])), 15, (0, 0, 255), 3)
                    cv2.circle(display_frame, (int(pt[0]), int(pt[1])), 5, (0, 255, 255), -1)
                except: pass

        # 5. 显示 UI 信息
        h, w = display_frame.shape[:2]
        scale = DISPLAY_WIDTH / w
        disp = cv2.resize(display_frame, (DISPLAY_WIDTH, int(h*scale)))
        
        info = [
            f"[W/S] Pitch: {params['pitch']:.1f}",
            f"[A/D] Yaw:   {params['yaw']:.1f}",
            f"[J/L] Tx:    {params['tx']:.2f}",
            f"[I/K] Ty:    {params['ty']:.2f} (Height)",
            f"[U/O] Tz:    {params['tz']:.2f} (Depth)",
            f"[M]   Mirror X: {params['mirror_x']}",
            f"[Z/C] Time:  {params['time_offset']:.2f}s",
            f"[SPACE] Pause/Play | [ESC] Save"
        ]
        
        for i, line in enumerate(info):
            cv2.putText(disp, line, (20, 40 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Interactive Calibration Tuner', disp)
        
        # 6. 处理按键
        key = cv2.waitKey(1 if not paused else 30) & 0xFF
        
        step_angle = 1.0
        step_dist = 0.1
        step_time = 0.1
        
        if key == 27: # ESC
            print("保存并退出...")
            np.savez(OUTPUT_NPZ, R=R, T=T, K=K, params=params)
            break
        elif key == 32: paused = not paused
        elif key == ord('w'): params['pitch'] += step_angle
        elif key == ord('s'): params['pitch'] -= step_angle
        elif key == ord('a'): params['yaw'] -= step_angle
        elif key == ord('d'): params['yaw'] += step_angle
        elif key == ord('j'): params['tx'] -= step_dist
        elif key == ord('l'): params['tx'] += step_dist
        elif key == ord('i'): params['ty'] -= step_dist
        elif key == ord('k'): params['ty'] += step_dist
        elif key == ord('u'): params['tz'] -= step_dist
        elif key == ord('o'): params['tz'] += step_dist
        elif key == ord('z'): params['time_offset'] -= step_time
        elif key == ord('c'): params['time_offset'] += step_time
        elif key == ord('m'): params['mirror_x'] = not params['mirror_x']

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()