# 视频处理模块（优化后的高效实现）
import cv2
import os
import argparse
from seatbelt_detector import detect_single_frame

def process_video(video_path, output_dir, skip_frames=0):
    """
    视频处理核心逻辑（优化后的高效实现）
    
    Args:
        video_path: 输入视频路径
        output_dir: 输出目录
        skip_frames: 跳过帧数（0表示不跳过，1表示每2帧处理1帧）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "result_output.mp4")
    
    # 获取视频属性
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 逐帧处理（优化：跳过部分帧）
    cap = cv2.VideoCapture(video_path)
    processed_frames = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video: {video_path}")
    print(f"Total frames: {total_frames}")
    print(f"Output will be saved to: {output_path}")
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
            continue  # 跳过部分帧
            
        # 优化3：使用高效推理（imgsz=800, conf=0.5）
        result = detect_single_frame(frame)
        
        # 保存处理后的帧
        out.write(result['frame'])
        
        # 打印进度
        if processed_frames % 10 == 0 or processed_frames == total_frames:
            print(f"Processed frame {processed_frames}/{total_frames} ({processed_frames/total_frames:.1%})")
        
        processed_frames += 1
    
    cap.release()
    out.release()
    print(f"\nProcessing completed. Results saved to: {output_path}")

if __name__ == "__main__":
    """
    视频处理入口
    
    使用方法:
    python video_process.py --video_path /path/to/input/video.mp4 --output_dir /path/to/output
    
    注意: 
    - 请将 --video_path 替换为实际视频路径
    - --output_dir 可选（默认路径已设置）
    - 通过 --skip_frames 跳过帧数（0表示不跳过）
    """
    parser = argparse.ArgumentParser(description='Process video for seatbelt detection')
    parser.add_argument('--video_path', type=str, required=True, 
                        help='Path to the input video file (e.g., /home/user/video.mp4)')
    parser.add_argument('--output_dir', type=str, default="/home/sutpc/sjh/project03/runs/track/car_inside_detection",
                        help='Directory to save output video (default: /home/sutpc/sjh/project03/runs/track/car_inside_detection)')
    parser.add_argument('--skip_frames', type=int, default=0,
                        help='Skip frames (e.g., 1 means process every 2 frames)')
    args = parser.parse_args()
    
    # 检查输入视频是否存在
    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"Input video not found at: {args.video_path}")
    
    # 执行视频处理
    process_video(
        video_path=args.video_path,
        output_dir=args.output_dir,
        skip_frames=args.skip_frames
    )