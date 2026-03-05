# video_processor.py
import cv2
import os
from seatbelt_detector import detect_single_frame

# 视频路径与输出设置
video_path = "/home/sutpc/sjh/project03/video/test5.mp4"
output_dir = "/home/sutpc/sjh/project03/runs/track/car_inside_detection"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "result_output.mp4")

# 获取视频属性
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

# 初始化视频写入器
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# 逐帧处理（每帧调用一次检测模块）
cap = cv2.VideoCapture(video_path)
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    # 每帧都调用检测（根据需求可调整为每N帧调用）
    result = detect_single_frame(frame)
    
    # 保存处理后的帧
    out.write(result['frame'])
    
    # 打印进度
    if frame_count % 10 == 0:
        print(f"Processed frame {frame_count}")

cap.release()
out.release()
print(f"Processing completed. Results saved to: {output_path}")