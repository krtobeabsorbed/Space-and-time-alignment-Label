import numpy as np
import cv2
import os

# ==========================================
# 诊断模式配置
# ==========================================
RADAR_FILE = 'radar_track1.txt'  #先试 Radar 1
VIDEO_FILE = 'a3.mp4'            
OUTPUT_NPZ = 'calib_diagnostic.npz'

# 初始参数
INIT_PARAMS = {
    'tx': 0.0, 'ty': 1.5, 'tz': 1.0,
    'pitch': 20.0, 'yaw': 0.0, 'roll': 0.0,
    'time_offset': 0.0,          
    'mirror_x': False
}

# 相机内参
W, H = 3200, 1800
F_mm = 4.0
Sensor_W_mm = 5.9
fx = F_mm * W / Sensor_W_mm
fy = fx
cx, cy = W / 2, H / 2
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((4,1))

RADAR_FPS = 16.13
VIDEO_FPS = 30.0
DISPLAY_WIDTH = 1280

def get_rotation_matrix(pitch, yaw, roll):
    rx, ry, rz = np.deg2rad(pitch), np.deg2rad(yaw), np.deg2rad(roll)
    Rx = np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]])
    Ry = np.array([[np.cos(ry),0,np.sin(ry)],[0,1,0],[-np.sin(ry),0,np.cos(ry)]])
    Rz = np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]])
    return Rz @ Ry @ Rx

def main():
    if not os.path.exists(RADAR_FILE):
        print(f"❌ 找不到文件: {RADAR_FILE}")
        return

    print(f"📂 正在读取雷达文件: {RADAR_FILE} ...")
    radar_data = np.loadtxt(RADAR_FILE)
    print(f"✅ 雷达数据读取成功，共 {len(radar_data)} 行")
    
    # 检查数据是否真的在动
    x_std = np.std(radar_data[:, 0])
    y_std = np.std(radar_data[:, 1])
    print(f"📊 数据活跃度检查: X轴变化量={x_std:.3f}, Y轴变化量={y_std:.3f}")
    if x_std < 0.1 and y_std < 0.1:
        print("⚠️⚠️⚠️ 警告：整个雷达文件的数据几乎没有变化！是不是选错文件了？")

    cap = cv2.VideoCapture(VIDEO_FILE)
    params = INIT_PARAMS.copy()
    frame_idx = 0
    paused = False

    print("\n>>> 启动诊断监控 <<<")
    print("请按【空格键】播放，然后观察控制台输出的数值变化！")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                continue
            frame_idx += 1

        t_vid = frame_idx / VIDEO_FPS
        t_rad = t_vid + params['time_offset']
        rad_idx = int(t_rad * RADAR_FPS)

        # === 核心诊断打印 ===
        # 每隔 10 帧打印一次，防止刷屏太快
        if not paused and frame_idx % 10 == 0:
            print(f"Time: {t_rad:.2f}s | Idx: {rad_idx} | ", end="")
            
            # 检查这一刻的数据
            if 0 <= rad_idx < len(radar_data):
                raw_pt = radar_data[rad_idx]
                print(f"Raw Radar: [{raw_pt[0]:.2f}, {raw_pt[1]:.2f}] <--- 这里的数字在变吗？")
            else:
                print("❌ 越界 (无数据)")

        # 计算变换
        R = get_rotation_matrix(params['pitch'], params['yaw'], params['roll'])
        T = np.array([params['tx'], params['ty'], params['tz']], dtype=np.float32)
        rvec, _ = cv2.Rodrigues(R)

        # 绘图逻辑
        display_frame = frame.copy()
        points_to_draw = []
        
        # 宽容模式：取前后 5 帧，只要有点就画出来
        for i in range(rad_idx - 5, rad_idx + 6):
            if 0 <= i < len(radar_data):
                r_pt = radar_data[i]
                if not np.isnan(r_pt[0]) and (abs(r_pt[0])>0.1 or abs(r_pt[1])>0.1):
                    x_r, y_r = r_pt[0], r_pt[1]
                    final_x = -x_r if params['mirror_x'] else x_r
                    # 默认映射：x->x, y->z
                    obj_pt = np.array([final_x, 0, y_r], dtype=np.float32)
                    points_to_draw.append(obj_pt)

        if len(points_to_draw) > 0:
            img_pts, _ = cv2.projectPoints(np.array(points_to_draw), rvec, T, K, dist_coeffs)
            for pt in img_pts.reshape(-1, 2):
                try:
                    cv2.circle(display_frame, (int(pt[0]), int(pt[1])), 15, (0, 0, 255), 3)
                    cv2.circle(display_frame, (int(pt[0]), int(pt[1])), 5, (0, 255, 255), -1)
                except: pass
        else:
            # 如果当前没点，画个大叉提示
            cv2.putText(display_frame, "NO DATA HERE", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

        # 显示
        h, w = display_frame.shape[:2]
        scale = DISPLAY_WIDTH / w
        disp = cv2.resize(display_frame, (DISPLAY_WIDTH, int(h*scale)))
        
        cv2.putText(disp, f"Radar Time: {t_rad:.2f}s (Offset: {params['time_offset']:.1f})", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(disp, "[Z/C] Change Time", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow('Diagnostic Mode', disp)
        
        key = cv2.waitKey(1 if not paused else 30) & 0xFF
        if key == 27: break
        elif key == 32: paused = not paused
        elif key == ord('z'): params['time_offset'] -= 0.5 # 加大步长，快速翻页
        elif key == ord('c'): params['time_offset'] += 0.5

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()