# # seatbelt_detector.py
# from ultralytics import RTDETR
# import numpy as np
# import cv2
# import argparse
# import os

# # 加载模型
# model = RTDETR("/home/ubuntu/graduation_design/seatbelt_det.pt")

# # 类别ID映射
# CLASS_MAP = {
#     0: "person-noseatbelt",
#     1: "person-seatbelt",
#     2: "seatbelt",
#     3: "windshield"
# }
# WINDOW_CLASS_ID = 3
# PERSON_CLASS_IDS = [0, 1]

# def is_inside_window(person_bbox, window_bbox, threshold=0.5):
#     """判断人员检测框是否在车前窗区域内"""
#     px1, py1, px2, py2 = person_bbox
#     wx1, wy1, wx2, wy2 = window_bbox

#     inter_x1 = max(px1, wx1)
#     inter_y1 = max(py1, wy1)
#     inter_x2 = min(px2, wx2)
#     inter_y2 = min(py2, wy2)
#     inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
#     person_area = (px2 - px1) * (py2 - py1)
#     if person_area == 0:
#         return False
    
#     return (inter_area / person_area) >= threshold

# def calculate_iou(box1, box2):
#     """计算两个边界框的IoU"""
#     x1_1, y1_1, x2_1, y2_1 = box1
#     x1_2, y1_2, x2_2, y2_2 = box2
    
#     inter_x1 = max(x1_1, x1_2)
#     inter_y1 = max(y1_1, y1_2)
#     inter_x2 = min(x2_1, x2_2)
#     inter_y2 = min(y2_1, y2_2)
    
#     inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
#     area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
#     area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
#     union_area = area1 + area2 - inter_area
    
#     return inter_area / union_area if union_area > 0 else 0.0

# def detect_single_frame(image):
#     """
#     对单帧图像进行安全带检测（优化版）
    
#     Args:
#         image: 输入的BGR图像
    
#     Returns:
#         dict: 包含检测结果的字典
#             - 'frame': 处理后的图像
#             - 'results': 检测框信息列表
#     """
#     # 关键优化：使用与训练一致的尺寸，conf设为0.4以保留中置信度框
#     results = model.predict(
#         image,
#         imgsz=800,  # 与训练尺寸一致
#         conf=0.38,   # 调整为0.4，保留低/中置信度框后续过滤
#         iou=0.8,    # 适当降低IOU阈值
#         device=0,
#         classes=PERSON_CLASS_IDS + [WINDOW_CLASS_ID]
#     )
    
#     result = results[0]
#     boxes = result.boxes
#     ids = boxes.id if boxes.id is not None else list(range(len(boxes)))
    
#     # 优化1：只提取一次车窗区域，避免重复遍历
#     window_bbox = None
#     for box in boxes:
#         if int(box.cls) == WINDOW_CLASS_ID:
#             wx1, wy1, wx2, wy2 = box.xyxy[0].tolist()
#             window_bbox = (wx1, wy1, wx2, wy2)
#             break
    
#     # 优化2：预先计算安全带框（用于修正逻辑）
#     seatbelt_boxes = []
#     for box in boxes:
#         if int(box.cls) == 2:  # 安全带类别
#             seatbelt_boxes.append(box.xyxy[0].tolist())
    
#     # 优化3：拆分人员框为高/中/低置信度，低置信度直接丢弃
#     high_conf_person_boxes = []  # >0.7 直接输出
#     mid_conf_person_boxes = []   # 0.38-0.7 二次验证
#     # 存储人员框及其所属类别、置信度，方便后续处理
#     for box, obj_id in zip(boxes, ids):
#         cls_id = int(box.cls)
#         conf = box.conf.item()  # 获取置信度值
#         if cls_id in PERSON_CLASS_IDS:
#             if conf > 0.7:
#                 high_conf_person_boxes.append((box.xyxy[0].tolist(), obj_id, cls_id, conf))
#             elif 0.38 <= conf <= 0.7:
#                 mid_conf_person_boxes.append((box.xyxy[0].tolist(), obj_id, cls_id, conf))
#             # <0.38 直接丢弃，不加入任何列表
    
#     # 优化4：仅对中置信度框应用修正逻辑（仅处理未系安全带人员）
#     revised_cls = {}
#     for person_bbox, obj_id, cls_id, conf in mid_conf_person_boxes:
#         if int(obj_id) in revised_cls:
#             continue  # 已修正
            
#         if cls_id == 0:  # 仅处理未系安全带
#             for sb in seatbelt_boxes:
#                 iou = calculate_iou(person_bbox, sb)
#                 if iou > 0.8:
#                     revised_cls[int(obj_id)] = 1  # 修正为已系安全带
#                     break
    
#     # 处理检测结果（区分高/中置信度逻辑）
#     frame = image.copy()
#     all_results = []  # 存储最终结果
    
#     # 处理高置信度人员框：直接输出，不校正、不判断车窗
#     for person_bbox, obj_id, cls_id, conf in high_conf_person_boxes:
#         x1, y1, x2, y2 = map(int, person_bbox)
#         obj_id = int(obj_id)
        
#         # 高置信度直接输出类别，不判断车窗、不校正
#         if cls_id == 0:
#             label = f"Unbelted"
#             box_color = (0, 0, 255)
#             text_color = (0, 0, 255)
#         else:
#             label = f"Belted"
#             box_color = (0, 255, 0)
#             text_color = (0, 255, 0)
        
#         cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
#         cv2.putText(frame, label, (x1, y1-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
#         # 记录高置信度结果（is_inside设为True，因未判断车窗）
#         all_results.append({
#             'bbox': person_bbox,
#             'cls': cls_id,
#             'id': obj_id,
#             'is_inside': True,
#             'conf': conf
#         })
    
#     # 处理中置信度人员框：执行原有车窗判断和类别校正
#     for person_bbox, obj_id, cls_id, conf in mid_conf_person_boxes:
#         if int(obj_id) in revised_cls:
#             cls_id = revised_cls[int(obj_id)]
        
#         # 仅处理人员类别
#         if cls_id not in PERSON_CLASS_IDS:
#             continue
            
#         x1, y1, x2, y2 = map(int, person_bbox)
#         obj_id = int(obj_id)
        
#         # 执行车窗判断
#         is_inside = window_bbox is not None and is_inside_window(person_bbox, window_bbox)
        
#         if is_inside:
#             if cls_id == 0:
#                 label = f"Unbelted"
#                 box_color = (0, 0, 255)
#                 text_color = (0, 0, 255)
#             else:
#                 label = f"Belted"
#                 box_color = (0, 255, 0)
#                 text_color = (0, 255, 0)
            
#             cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
#             cv2.putText(frame, label, (x1, y1-10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
#         else:
#             label = f"Outside"
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
#             cv2.putText(frame, label, (x1, y1-10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
#         # 记录中置信度结果
#         all_results.append({
#             'bbox': person_bbox,
#             'cls': cls_id,
#             'id': obj_id,
#             'is_inside': is_inside,
#             'conf': conf
#         })
    
#     # 绘制车前窗边界（如果存在）
#     if window_bbox is not None:
#         wx1, wy1, wx2, wy2 = map(int, window_bbox)
#         cv2.rectangle(frame, (wx1, wy1), (wx2, wy2), (255, 0, 0), 2)
#         cv2.putText(frame, "Windshield", (wx1, wy1-10), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

#     return {
#         'frame': frame,
#         'results': all_results
#     }

# if __name__ == "__main__":
#     """
#     单张图片推理入口
    
#     使用方法:
#     python seatbelt_detector.py --image_path /path/to/input/image.jpg
    
#     注意: 请将 --image_path 参数替换为实际图片路径
#     """
#     parser = argparse.ArgumentParser(description='Seatbelt Detection on Single Image')
#     parser.add_argument('--image_path', type=str, required=True, 
#                         help='Path to the input image file (e.g., /home/user/image.jpg)')
#     args = parser.parse_args()
    
#     # 检查输入图片是否存在
#     if not os.path.exists(args.image_path):
#         raise FileNotFoundError(f"Input image not found at: {args.image_path}")
    
#     # 读取图片
#     image = cv2.imread(args.image_path)
#     if image is None:
#         raise ValueError(f"Failed to read image from: {args.image_path}")
    
#     # 执行检测
#     result = detect_single_frame(image)
    
#     # 保存结果（不再显示窗口）
#     output_path = os.path.splitext(args.image_path)[0] + "_result.jpg"
#     cv2.imwrite(output_path, result['frame'])
#     print(f"Result saved to: {output_path}")

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
    # 关键优化：使用与训练一致的尺寸，conf设为0.4以保留中置信度框
    results = model.predict(
        image,
        imgsz=800,  # 与训练尺寸一致
        conf=0.38,   # 调整为0.4，保留低/中置信度框后续过滤
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
    
    # 优化3：拆分人员框为高/中/低置信度，低置信度直接丢弃
    high_conf_person_boxes = []  # >0.7 直接输出
    mid_conf_person_boxes = []   # 0.38-0.7 二次验证
    # 存储人员框及其所属类别、置信度，方便后续处理
    for box, obj_id in zip(boxes, ids):
        cls_id = int(box.cls)
        conf = box.conf.item()  # 获取置信度值
        if cls_id in PERSON_CLASS_IDS:
            if conf > 0.7:
                high_conf_person_boxes.append((box.xyxy[0].tolist(), obj_id, cls_id, conf))
            elif 0.38 <= conf <= 0.7:
                mid_conf_person_boxes.append((box.xyxy[0].tolist(), obj_id, cls_id, conf))
            # <0.38 直接丢弃，不加入任何列表
    
    # 优化4：仅对中置信度框应用修正逻辑（先判断是否在车窗内，再修正安全带状态）
    revised_cls = {}
    for person_bbox, obj_id, cls_id, conf in mid_conf_person_boxes:
        if int(obj_id) in revised_cls:
            continue  # 已修正
        
        # 第一步：先判断是否在车窗内，不在则跳过修正
        is_inside = window_bbox is not None and is_inside_window(person_bbox, window_bbox)
        if not is_inside:
            continue
        
        # 第二步：仅对车窗内的未系安全带人员框修正
        if cls_id == 0:  # 仅处理未系安全带
            for sb in seatbelt_boxes:
                iou = calculate_iou(person_bbox, sb)
                if iou > 0.8:
                    revised_cls[int(obj_id)] = 1  # 修正为已系安全带
                    break
    
    # 处理检测结果（区分高/中置信度逻辑）
    frame = image.copy()
    all_results = []  # 存储最终结果
    
    # 处理高置信度人员框：直接输出，不校正、不判断车窗
    for person_bbox, obj_id, cls_id, conf in high_conf_person_boxes:
        x1, y1, x2, y2 = map(int, person_bbox)
        obj_id = int(obj_id)
        
        # 高置信度直接输出类别，不判断车窗、不校正
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
        
        # 记录高置信度结果（is_inside设为True，因未判断车窗）
        all_results.append({
            'bbox': person_bbox,
            'cls': cls_id,
            'id': obj_id,
            'is_inside': True,
            'conf': conf
        })
    
    # 处理中置信度人员框：先判断车窗，再应用安全带状态校正
    for person_bbox, obj_id, cls_id, conf in mid_conf_person_boxes:
        # 【修复】添加坐标解包
        x1, y1, x2, y2 = map(int, person_bbox)
        obj_id = int(obj_id)
        
        # 先应用安全带状态修正（仅车窗内的框会被修正）
        if int(obj_id) in revised_cls:
            cls_id = revised_cls[int(obj_id)]
        
        # 执行车窗判断（用于绘制和结果记录）
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
        
        # 记录中置信度结果
        all_results.append({
            'bbox': person_bbox,
            'cls': cls_id,
            'id': obj_id,
            'is_inside': is_inside,
            'conf': conf
        })
    
    # 绘制车前窗边界（如果存在）
    if window_bbox is not None:
        wx1, wy1, wx2, wy2 = map(int, window_bbox)
        cv2.rectangle(frame, (wx1, wy1), (wx2, wy2), (255, 0, 0), 2)
        cv2.putText(frame, "Windshield", (wx1, wy1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return {
        'frame': frame,
        'results': all_results
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