import numpy as np
import cv2
import os

# ==========================================
# 验证配置
# ==========================================
# 【修改 1】使用刚才生成的手动标定文件
NPZ_FILE = 'calib_manual_hack.npz' 
RADAR_FILE = 'radar_track2.txt'
VIDEO_FILE = 'a3.mp4'

# 【核心参数】时间偏移量 (秒)
# 之前的问题是空间不对，现在空间大概对上了，可能还需要调时间
TIME_OFFSET = -3

# 其他参数
RADAR_FPS = 16.13
VIDEO_FPS = 30.0
DISPLAY_WIDTH = 1280

# 【修改 2】配合手动标定的坐标变换
def transform_radar_point(r_pt):
    x_radar = r_pt[0] # 雷达左右
    y_radar = r_pt[1] # 雷达深度(前方)

    # 这是一个经验性的映射，旨在配合俯拍相机
    # 雷达的 X -> 相机的 X (左右)
    # 雷达的 Y (前方) -> 相机的 Z (深度)
    # 雷达的 Z (地面0) -> 相机的 -Y (下方)
    # 我们稍微抬高一点点 z，假设雷达探测的是脚踝高度
    obj_pt = np.array([x_radar, 0.2, y_radar], dtype=np.float32)
    return obj_pt

def verify_calibration():
    # ... (此处省略的代码与上一条回复中的 verify_calibration_with_offset.py 完全一致)
    # ... 请直接复制上一条回复中的 verify_calibration 主函数代码到这里 ...
    if not os.path.exists(NPZ_FILE) or not os.path.exists(RADAR_FILE):
        print("找不到必要文件")
        return

    # 1. 加载标定
    data = np.load(NPZ_FILE)
    R, T, K = data['R'], data['T'], data['K']
    rvec, _ = cv2.Rodrigues(R) # 转回向量形式供 projectPoints 使用
    
    print(f"当前应用时间偏移: {TIME_OFFSET} 秒")
    print(f"加载标定文件: {NPZ_FILE}")

    # 2. 读取雷达
    radar_data = np.loadtxt(RADAR_FILE)
    
    # 3. 打开视频
    cap = cv2.VideoCapture(VIDEO_FILE)
    if not cap.isOpened():
        print(f"无法打开视频: {VIDEO_FILE}")
        return
        
    frame_idx = 0
    print("开始播放... 按 'q' 退出，按空格暂停")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        t_vid = frame_idx / VIDEO_FPS
        t_rad_target = t_vid + TIME_OFFSET
        rad_idx = int(t_rad_target * RADAR_FPS)
        
        points_to_draw = []
        for i in range(rad_idx - 1, rad_idx + 2):
            if 0 <= i < len(radar_data):
                r_pt = radar_data[i]
                if not np.isnan(r_pt[0]) and (abs(r_pt[0])>0.1 or abs(r_pt[1])>0.1):
                    # 使用新的变换函数
                    obj_pt = transform_radar_point(r_pt)
                    points_to_draw.append(obj_pt)

        if len(points_to_draw) > 0:
            object_points = np.array(points_to_draw)
            # 使用加载的手动 R, T
            img_pts_proj, _ = cv2.projectPoints(object_points, rvec, T, K, np.zeros(4))
            img_pts_proj = img_pts_proj.reshape(-1, 2)

            for pt in img_pts_proj:
                try:
                    u, v = int(pt[0]), int(pt[1])
                    cv2.circle(frame, (u, v), 15, (0, 0, 255), 3)
                    cv2.circle(frame, (u, v), 6, (0, 255, 255), -1)
                except: pass
        
        h, w = frame.shape[:2]
        scale = DISPLAY_WIDTH / w
        frame_disp = cv2.resize(frame, (DISPLAY_WIDTH, int(h*scale)))
        cv2.putText(frame_disp, f"Time Offset: {TIME_OFFSET:.2f}s", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        cv2.imshow('Sync Check (Manual Hack)', frame_disp)
        
        key = cv2.waitKey(20)
        if key == ord('q'): break
        elif key == ord(' '): cv2.waitKey(0)

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    verify_calibration()