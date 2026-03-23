# seatbelt_detector.py
from ultralytics import RTDETR
import numpy as np
import cv2
import argparse
import os

# 加载模型
model = RTDETR("/home/ubuntu/graduation_design/seatbelt_det.pt")

# 类别ID映射
CLASS_MAP = {
    0: "person-noseatbelt",
    1: "person-seatbelt",
    2: "seatbelt",
    3: "windshield"
}
WINDOW_CLASS_ID = 3
PERSON_CLASS_IDS = [0, 1]

def is_inside_window(person_bbox, window_bbox, threshold=0.5):
    """判断人员检测框是否在车前窗区域内"""
    px1, py1, px2, py2 = person_bbox
    wx1, wy1, wx2, wy2 = window_bbox

    inter_x1 = max(px1, wx1)
    inter_y1 = max(py1, wy1)
    inter_x2 = min(px2, wx2)
    inter_y2 = min(py2, wy2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    person_area = (px2 - px1) * (py2 - py1)
    if person_area == 0:
        return False
    
    return (inter_area / person_area) >= threshold

def calculate_iou(box1, box2):
    """计算两个边界框的IoU"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def detect_single_frame(image):
    """
    对单帧图像进行安全带检测（优化版）
    
    Args:
        image: 输入的BGR图像
    
    Returns:
        dict: 包含检测结果的字典
            - 'frame': 处理后的图像
            - 'results': 检测框信息列表
    """
    # 关键优化：使用与训练一致的尺寸
    results = model.predict(
        image,
        imgsz=800,  # 与训练尺寸一致
        conf=0.6,   # 优化置信度阈值
        iou=0.8,    # 适当降低IOU阈值
        device=0,
        classes=PERSON_CLASS_IDS + [WINDOW_CLASS_ID]
    )
    
    result = results[0]
    boxes = result.boxes
    ids = boxes.id if boxes.id is not None else list(range(len(boxes)))
    
    # 优化1：只提取一次车窗区域，避免重复遍历
    window_bbox = None
    for box in boxes:
        if int(box.cls) == WINDOW_CLASS_ID:
            wx1, wy1, wx2, wy2 = box.xyxy[0].tolist()
            window_bbox = (wx1, wy1, wx2, wy2)
            break
    
    # 优化2：预先计算安全带框（用于修正逻辑）
    seatbelt_boxes = []
    for box in boxes:
        if int(box.cls) == 2:  # 安全带类别
            seatbelt_boxes.append(box.xyxy[0].tolist())
    
    # 优化3：只处理人员框，避免处理非人员类别
    person_boxes = []
    # 存储人员框及其所属类别，方便后续修正逻辑使用
    for box, obj_id in zip(boxes, ids):
        cls_id = int(box.cls)
        if cls_id in PERSON_CLASS_IDS:
            person_boxes.append((box.xyxy[0].tolist(), obj_id, cls_id))
    
    # 优化4：应用修正逻辑（仅处理未系安全带人员）
    revised_cls = {}
    # revision_count = 0  # 记录修正次数
    for person_bbox, obj_id, cls_id in person_boxes:
        if int(obj_id) in revised_cls:
            continue  # 已修正
            
        if cls_id == 0:  # 仅处理未系安全带
            for sb in seatbelt_boxes:
                iou = calculate_iou(person_bbox, sb)
                if iou > 0.8:
                    revised_cls[int(obj_id)] = 1  # 修正为已系安全带
                    # revision_count += 1
                    break
    
    # 处理检测结果（优化后的循环）
    frame = image.copy()
    for person_bbox, obj_id, cls_id in person_boxes:
        if int(obj_id) in revised_cls:
            cls_id = revised_cls[int(obj_id)]
        
        # 仅处理人员类别
        if cls_id not in PERSON_CLASS_IDS:
            continue
            
        x1, y1, x2, y2 = map(int, person_bbox)
        obj_id = int(obj_id)
        
        # 优化5：避免重复计算is_inside_window
        is_inside = window_bbox is not None and is_inside_window(person_bbox, window_bbox)
        
        if is_inside:
            if cls_id == 0:
                label = f"Unbelted"
                box_color = (0, 0, 255)
                text_color = (0, 0, 255)
            else:
                label = f"Belted"
                box_color = (0, 255, 0)
                text_color = (0, 255, 0)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        else:
            label = f"Outside"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # 绘制车前窗边界（如果存在）
    if window_bbox is not None:
        wx1, wy1, wx2, wy2 = map(int, window_bbox)
        cv2.rectangle(frame, (wx1, wy1), (wx2, wy2), (255, 0, 0), 2)
        cv2.putText(frame, "Windshield", (wx1, wy1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # # 输出修正逻辑执行次数
    # print(f"修正逻辑进行了 {revision_count} 次")

    return {
        'frame': frame,
        # 'revision_count': revision_count,
        'results': [{
            'bbox': box.xyxy[0].tolist(),
            'cls': int(box.cls),
            'id': int(obj_id),
            'is_inside': window_bbox is not None and is_inside_window(box.xyxy[0].tolist(), window_bbox)
        } for box, obj_id in zip(boxes, ids) if int(box.cls) in PERSON_CLASS_IDS]
    }

if __name__ == "__main__":
    """
    单张图片推理入口
    
    使用方法:
    python seatbelt_detector.py --image_path /path/to/input/image.jpg
    
    注意: 请将 --image_path 参数替换为实际图片路径
    """
    parser = argparse.ArgumentParser(description='Seatbelt Detection on Single Image')
    parser.add_argument('--image_path', type=str, required=True, 
                        help='Path to the input image file (e.g., /home/user/image.jpg)')
    args = parser.parse_args()
    
    # 检查输入图片是否存在
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Input image not found at: {args.image_path}")
    
    # 读取图片
    image = cv2.imread(args.image_path)
    if image is None:
        raise ValueError(f"Failed to read image from: {args.image_path}")
    
    # 执行检测
    result = detect_single_frame(image)
    
    # 保存结果（不再显示窗口）
    output_path = os.path.splitext(args.image_path)[0] + "_result.jpg"
    cv2.imwrite(output_path, result['frame'])
    print(f"Result saved to: {output_path}")