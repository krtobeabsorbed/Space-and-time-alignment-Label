import numpy as np
import cv2
import os
import csv

# ==========================================
# 1. 再次确认文件名 (必须完全一致!)
# ==========================================
NPZ_FILE = 'calib_r2_a1_tuned.npz'     # <--- 必须是你刚才按ESC保存的那个文件名
RADAR_FILE = 'radar_track2_final_smooth.txt'     # 雷达文件
VIDEO_FILE = 'a1.mp4'               # 视频文件

OUTPUT_VIDEO = 'output_fusion_final_r2_c1.mp4'
OUTPUT_CSV = 'dataset_fusion_final_r2_c1.csv'

# 海康相机内参 (必须与 Tuner 里的完全一致)
W, H = 3200, 1800
F_mm = 4.0
Sensor_W_mm = 5.9
fx = F_mm * W / Sensor_W_mm
fy = fx
cx = W / 2
cy = H / 2
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

RADAR_FPS = 16.13
VIDEO_FPS = 30.0

def generate_strict():
    if not os.path.exists(NPZ_FILE):
        print(f"❌ 错误：找不到文件 {NPZ_FILE}")
        print("请回到 interactive_tuner.py，按 ESC 确保保存成功！")
        return

    print(f"📂 正在加载标定文件: {NPZ_FILE} ...")
    data = np.load(NPZ_FILE, allow_pickle=True)
    
    # --- 核心排查点：读取 T, R ---
    R = data['R']
    T = data['T']
    
    # --- 核心排查点：读取 params ---
    # 我们不使用 try-except，如果出错直接报错，方便找原因
    if 'params' not in data:
        print("❌ 严重错误：npz 文件里没有 'params' 字段！")
        print("原因：你之前的 Tuner 代码可能版本过旧，或者保存时没存进去。")
        print("解决：请重新运行 Tuner 并按 ESC 保存。")
        return

    params = data['params'].item()
    time_offset = params['time_offset']
    mirror_x = params['mirror_x']

    # --- 🚨 打印出来给你看！必须核对！ 🚨 ---
    print("="*40)
    print("✅ 参数加载成功！请核对以下数值是否熟悉：")
    print(f"   ▶ 平移向量 T (I/K/J/L调的): {T}")
    print(f"   ▶ 时间偏移 (Z/C调的):       {time_offset} 秒")
    print(f"   ▶ 镜像开启 (M键调的):       {mirror_x}")
    print("="*40)
    
    if abs(T[1] - 1.5) < 0.01 and abs(T[2] - 0.5) < 0.01:
        print("⚠️ 警告：你的 T 看起来像是初始默认值 (1.5, 0.5)。")
        print("   如果你在 Tuner 里大改过位置，这说明保存没成功！")

    # 开始生成
    rvec, _ = cv2.Rodrigues(R)
    radar_data = np.loadtxt(RADAR_FILE)
    cap = cv2.VideoCapture(VIDEO_FILE)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), VIDEO_FPS, (width, height))
    csv_file = open(OUTPUT_CSV, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['Video_Frame', 'Radar_Time', 'Pixel_U', 'Pixel_V', 'Real_X', 'Real_Y', 'Real_Z'])

    print(f"🚀 开始渲染 {total_frames} 帧...")
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        t_vid = frame_idx / VIDEO_FPS
        t_rad_target = t_vid + time_offset
        rad_idx = int(t_rad_target * RADAR_FPS)
        
        points_3d = []
        points_raw = []

        for i in range(rad_idx - 1, rad_idx + 2):
            if 0 <= i < len(radar_data):
                r_pt = radar_data[i]
                if not np.isnan(r_pt[0]) and (abs(r_pt[0])>0.1 or abs(r_pt[1])>0.1):
                    x_r, y_r = r_pt[0], r_pt[1]
                    
                    # === 必须与 Tuner 逻辑完全一致 ===
                    final_x = -x_r if mirror_x else x_r
                    obj_pt = np.array([final_x, 0, y_r], dtype=np.float32)
                    
                    points_3d.append(obj_pt)
                    points_raw.append([final_x, y_r, 0])

        if len(points_3d) > 0:
            img_pts, _ = cv2.projectPoints(np.array(points_3d), rvec, T, K, np.zeros(4))
            for j, pt in enumerate(img_pts.reshape(-1, 2)):
                u, v = int(pt[0]), int(pt[1])
                if 0 <= u < width and 0 <= v < height:
                    cv2.circle(frame, (u, v), 10, (0, 0, 255), 2)
                    cv2.circle(frame, (u, v), 4, (0, 255, 255), -1)
                    rx, ry, rz = points_raw[j]
                    writer.writerow([frame_idx, f"{t_rad_target:.3f}", u, v, f"{rx:.3f}", f"{ry:.3f}", f"{rz:.3f}"])

        out.write(frame)
        if frame_idx % 50 == 0:
            print(f"进度: {frame_idx}/{total_frames}", end='\r')
        frame_idx += 1

    cap.release()
    out.release()
    csv_file.close()
    print("\n✅ 处理完成！请查看 output_fusion_final.mp4")

if __name__ == "__main__":
    generate_strict()