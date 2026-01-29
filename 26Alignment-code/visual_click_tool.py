import cv2
import numpy as np
import os

# ==========================================
# 配置
# ==========================================
VIDEO_PATH = "a4.mp4" # 你的视频路径
OUTPUT_FILE = "camera_track.txt"

# 显示时的目标宽度 (建议 1280 或 1600，取决于你的屏幕)
# 这只是为了让你能看全画面，不影响保存的数据精度
DISPLAY_WIDTH = 1280 

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # x, y 是你在缩小的画面上点击的坐标
        # 我们需要把它还原回原始分辨率
        scale = param['scale']
        
        real_x = int(x * scale)
        real_y = int(y * scale)
        
        frame_idx = param['frame_idx']
        
        print(f"Frame {frame_idx}: 点击屏幕({x},{y}) -> 还原坐标({real_x}, {real_y})")
        
        # 保存真实的原始坐标 [Frame_ID, u, v]
        param['coords'].append([frame_idx, real_x, real_y])
        
        # 在显示的画面上画个圈 (为了视觉反馈)
        cv2.circle(param['img_display'], (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Video_Calibration_Tool', param['img_display'])

def extract_visual_track():
    if not os.path.exists(VIDEO_PATH):
        print(f"错误: 找不到视频文件 {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    coords = []
    frame_idx = 0
    step = 10 
    
    # 获取原始视频分辨率
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 计算缩放比例
    scale_factor = orig_w / DISPLAY_WIDTH
    disp_h = int(orig_h / scale_factor)
    
    print(f"原始分辨率: {orig_w}x{orig_h}")
    print(f"显示分辨率: {DISPLAY_WIDTH}x{disp_h} (缩放倍数: {scale_factor:.2f})")
    print("===" * 20)
    print("【操作说明】")
    print("1. 画面已自动缩小适配屏幕，但保存的坐标会自动还原为原始高清坐标。")
    print("2. 没人的帧：直接按【空格键】跳过。")
    print("3. 有人的帧：点击脚底中心，然后按【空格键】继续。")
    print("4. 按 'q' 键保存并退出。")
    print("===" * 20)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        if frame_idx % step == 0:
            # 1. 缩放图片用于显示
            frame_display = cv2.resize(frame, (DISPLAY_WIDTH, disp_h))
            
            # 2. 传递参数 (包括缩放比例)
            param = {
                'frame_idx': frame_idx, 
                'coords': coords, 
                'img_display': frame_display, # 只在缩放图上画圈
                'scale': scale_factor         # 用于还原坐标
            }
            
            cv2.imshow('Video_Calibration_Tool', frame_display)
            cv2.setMouseCallback('Video_Calibration_Tool', click_event, param)
            
            key = cv2.waitKey(0)
            if key == ord('q'):
                print("用户请求退出...")
                break
        
        frame_idx += 1
        
    cap.release()
    cv2.destroyAllWindows()
    
    if len(coords) > 0:
        # 保存时记得带个 Header 说明，防止忘记
        np.savetxt(OUTPUT_FILE, np.array(coords), fmt="%d", header="Frame_ID u_real v_real")
        print(f"\n成功! 视觉轨迹已保存到 {OUTPUT_FILE}")
        print(f"共采集了 {len(coords)} 个点 (坐标已还原为 {orig_w}x{orig_h} 分辨率)。")
    else:
        print("\n未采集到任何点。")

if __name__ == "__main__":
    extract_visual_track()