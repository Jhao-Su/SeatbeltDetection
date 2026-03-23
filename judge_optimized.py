# 核心修改：增加类别一致性校验，仅匹配同类别标注
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安全带检测模型评测脚本（置信度分数版）
P/R/F1曲线横坐标为置信度分数（而非阈值）
展示不同置信度水平下的模型性能
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix
import json

# 导入原程序的检测函数和配置
from SeatBeltDetectionv3.seatbelt_detector import detect_single_frame, CLASS_MAP, PERSON_CLASS_IDS
# from SeatBeltDetectionv2.seatbelt_detector import detect_single_frame, CLASS_MAP, PERSON_CLASS_IDS

# # ==================== 配置路径 ====================
# TEST_IMAGES_DIR = "/home/ubuntu/graduation_design/test/images"
# TEST_LABELS_DIR = "/home/ubuntu/graduation_design/test/labels"
# OUTPUT_DIR = "/home/ubuntu/graduation_design/optimized_test_result"
# IMAGES_DIR = os.path.join(OUTPUT_DIR, 'images')
# INFOS_DIR = os.path.join(OUTPUT_DIR, 'infos')
# IOU_THRESHOLD = 0.5

# # ==================== 工具函数 ====================

# def parse_yolo_label(label_path, img_w, img_h):
#     """解析YOLO格式标注，返回 [cls, x1, y1, x2, y2]"""
#     annotations = []
#     if not os.path.exists(label_path):
#         return annotations
#     with open(label_path, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) < 5:
#                 continue
#             cls_id = int(parts[0])
#             if cls_id in PERSON_CLASS_IDS:
#                 x_c, y_c, w, h = map(float, parts[1:5])
#                 x1 = (x_c - w/2) * img_w
#                 y1 = (y_c - h/2) * img_h
#                 x2 = (x_c + w/2) * img_w
#                 y2 = (y_c + h/2) * img_h
#                 annotations.append([cls_id, x1, y1, x2, y2])
#     return annotations

# def calc_iou(box1, box2):
#     """计算两个边界框的IoU"""
#     x1_1, y1_1, x2_1, y2_1 = box1
#     x1_2, y1_2, x2_2, y2_2 = box2
#     inter_x1, inter_y1 = max(x1_1, x1_2), max(y1_1, y1_2)
#     inter_x2, inter_y2 = min(x2_1, x2_2), min(y2_1, y2_2)
#     inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
#     area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
#     area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
#     union_area = area1 + area2 - inter_area
#     return inter_area / union_area if union_area > 0 else 0.0

# def match_preds_to_gts(predictions, ground_truth, iou_thresh=0.5):
#     """
#     匹配预测与真实标注（增加类别一致性校验）
#     返回：TP数量, FP数量, FN数量, 以及详细匹配信息
#     """
#     matched_gt = set()
#     tp, fp = 0, 0
#     match_details = []
    
#     # 按置信度降序处理预测
#     sorted_preds = sorted(predictions, key=lambda x: x.get('conf', 0.5), reverse=True)
    
#     for pred in sorted_preds:
#         pred_cls = pred['cls']
#         pred_bbox = pred['bbox']
#         pred_conf = pred.get('conf', 0.5)
        
#         best_iou, best_gt_idx, best_gt_cls = 0, -1, -1
        
#         for gt_idx, gt in enumerate(ground_truth):
#             if gt_idx in matched_gt:
#                 continue
#             gt_cls = gt[0]
#             # 核心修改：增加类别一致性校验，仅匹配同类别标注
#             if gt_cls != pred_cls:
#                 continue
#             gt_bbox = gt[1:]
#             iou = calc_iou(pred_bbox, gt_bbox)
#             if iou > best_iou:
#                 best_iou, best_gt_idx, best_gt_cls = iou, gt_idx, gt_cls
        
#         if best_iou >= iou_thresh:
#             tp += 1
#             matched_gt.add(best_gt_idx)
#             match_details.append({
#                 'pred_cls': pred_cls,
#                 'gt_cls': best_gt_cls,
#                 'conf': pred_conf,
#                 'iou': best_iou,
#                 'is_tp': True
#             })
#         else:
#             fp += 1
#             match_details.append({
#                 'pred_cls': pred_cls,
#                 'gt_cls': -1,
#                 'conf': pred_conf,
#                 'iou': best_iou,
#                 'is_tp': False
#             })
    
#     fn = len(ground_truth) - len(matched_gt)
    
#     return tp, fp, fn, match_details

# def calculate_metrics_vs_confidence(all_predictions, all_ground_truth, iou_thresh=0.5, n_bins=50):
#     """
#     计算不同置信度分数下的P/R/F1指标
#     横坐标：置信度分数（0-1）
#     纵坐标：该置信度水平下的累积Precision/Recall/F1
#     """
#     # 收集所有预测及其匹配结果
#     all_pred_items = []
#     total_gts = 0
    
#     for preds, gts in zip(all_predictions, all_ground_truth):
#         total_gts += len(gts)
#         _, _, _, match_details = match_preds_to_gts(preds, gts, iou_thresh)
#         for item in match_details:
#             all_pred_items.append(item)
    
#     if len(all_pred_items) == 0:
#         return [], [], [], []
    
#     # 按置信度排序
#     all_pred_items.sort(key=lambda x: x['conf'])
    
#     # 生成置信度分数点（从0到1）
#     confidence_scores = np.linspace(0, 1, n_bins)
    
#     precisions, recalls, f1_scores = [], [], []
    
#     for conf_score in confidence_scores:
#         # 统计置信度 <= conf_score 的预测
#         tp_count = sum(1 for item in all_pred_items if item['conf'] <= conf_score and item['is_tp'])
#         fp_count = sum(1 for item in all_pred_items if item['conf'] <= conf_score and not item['is_tp'])
        
#         # Recall 基于总GT数
#         fn_count = total_gts - tp_count
        
#         p = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
#         r = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
#         f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        
#         precisions.append(p)
#         recalls.append(r)
#         f1_scores.append(f1)
    
#     return confidence_scores, precisions, recalls, f1_scores

# def calculate_pr_curve(all_predictions, all_ground_truth, iou_thresh=0.5):
#     """
#     计算PR曲线（按置信度阈值）
#     返回：每个类别的precision, recall, ap
#     """
#     # 收集所有预测及其GT信息
#     all_pred_items = []
#     total_gts = {cls_id: 0 for cls_id in PERSON_CLASS_IDS}
    
#     for preds, gts in zip(all_predictions, all_ground_truth):
#         # 统计GT数量
#         for gt in gts:
#             if gt[0] in total_gts:
#                 total_gts[gt[0]] += 1
        
#         # 匹配预测
#         _, _, _, match_details = match_preds_to_gts(preds, gts, iou_thresh)
#         for item in match_details:
#             all_pred_items.append(item)
    
#     # 按置信度降序排序
#     all_pred_items.sort(key=lambda x: x['conf'], reverse=True)
    
#     # 计算每类的PR曲线
#     pr_data = {}
    
#     for cls_id in PERSON_CLASS_IDS:
#         tp_cumsum = 0
#         fp_cumsum = 0
        
#         precisions = []
#         recalls = []
#         confidences = []
        
#         for item in all_pred_items:
#             if item['pred_cls'] == cls_id:
#                 confidences.append(item['conf'])
#                 if item['is_tp']:
#                     tp_cumsum += 1
#                 else:
#                     fp_cumsum += 1
                
#                 precision = tp_cumsum / (tp_cumsum + fp_cumsum)
#                 recall = tp_cumsum / total_gts[cls_id] if total_gts[cls_id] > 0 else 0
                
#                 precisions.append(precision)
#                 recalls.append(recall)
        
#         # 计算AP（使用积分法）
#         if len(precisions) > 0 and len(recalls) > 0:
#             # 确保recall单调递增，precision取后续最大值
#             precisions_interp = []
#             for i in range(len(precisions)):
#                 precisions_interp.append(max(precisions[i:]))
#             ap = np.trapz(precisions_interp, recalls)
#         else:
#             ap = 0.0
#             precisions = [0]
#             recalls = [0]
        
#         pr_data[cls_id] = {
#             'precision': precisions,
#             'recall': recalls,
#             'confidence': confidences,
#             'ap': ap
#         }
    
#     return pr_data

# # ==================== 主评测函数 ====================

# def evaluate():
#     """主评测函数：调用原程序进行推理并评测"""
    
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     os.makedirs(IMAGES_DIR, exist_ok=True)
#     os.makedirs(INFOS_DIR, exist_ok=True)
    
#     # 获取测试图片列表
#     image_files = sorted([
#         f for f in Path(TEST_IMAGES_DIR).iterdir() 
#         if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
#     ])
    
#     print(f"找到 {len(image_files)} 张测试图片")
    
#     all_predictions, all_ground_truth = [], []
    
#     # ========== 调用原程序进行推理 ==========
#     for idx, img_path in enumerate(image_files):
#         print(f"处理 [{idx+1}/{len(image_files)}]: {img_path.name}")
        
#         image = cv2.imread(str(img_path))
#         if image is None:
#             continue
        
#         h, w = image.shape[:2]
        
#         # 直接调用原程序的检测函数，获取最终推理结果
#         result = detect_single_frame(image)
        
#         # 保存推理结果图片到 test_result/images
#         cv2.imwrite(os.path.join(IMAGES_DIR, f"pred_{img_path.name}"), result['frame'])
        
#         # 直接使用原程序返回的检测结果
#         predictions = result['results']
        
#         # 读取真实标注
#         label_path = os.path.join(TEST_LABELS_DIR, img_path.stem + ".txt")
#         ground_truth = parse_yolo_label(label_path, w, h)
        
#         all_predictions.append(predictions)
#         all_ground_truth.append(ground_truth)
        
#         # 保存预测结果为 JSON 到 test_result/infos
#         with open(os.path.join(INFOS_DIR, f"pred_{img_path.stem}.json"), 'w') as f:
#             json.dump(predictions, f, indent=2)
    
#     # ========== 计算评测指标（IoU=0.5, 默认置信度） ==========
#     print("\n计算评测指标...")
    
#     total_tp, total_fp, total_fn = 0, 0, 0
#     for preds, gts in zip(all_predictions, all_ground_truth):
#         tp, fp, fn, _ = match_preds_to_gts(preds, gts, IOU_THRESHOLD)
#         total_tp += tp
#         total_fp += fp
#         total_fn += fn
    
#     precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
#     recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
#     f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
#     print(f"\n总体评测结果 (IoU={IOU_THRESHOLD}):")
#     print(f"  TP={total_tp}, FP={total_fp}, FN={total_fn}")
#     print(f"  Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    
#     # ========== 计算P/R/F1 vs 置信度分数（累积方式） ==========
#     print("\n计算置信度分数曲线（累积方式）...")
#     conf_scores, precisions, recalls, f1_scores = calculate_metrics_vs_confidence(
#         all_predictions, all_ground_truth, IOU_THRESHOLD, n_bins=50
#     )
    
#     # ========== 计算PR曲线 ==========
#     print("计算PR曲线...")
#     pr_data = calculate_pr_curve(all_predictions, all_ground_truth, IOU_THRESHOLD)
    
#     # ========== 绘制曲线 ==========
#     print("\n绘制评测曲线...")
#     class_names = ['person-noseatbelt', 'person-seatbelt']
    
#     # 【P曲线】Precision vs Confidence Score（累积方式）
#     plt.figure(figsize=(10, 6))
#     plt.plot(conf_scores, precisions, 'b-o', linewidth=2, markersize=3, label='Precision')
#     plt.xlabel('Confidence Score')
#     plt.ylabel('Precision')
#     plt.title('Precision-Confidence Curve (Cumulative)')
#     plt.grid(True, alpha=0.3)
#     plt.xlim(0, 1.0)
#     plt.ylim(0, 1.0)
#     plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Default(0.5)')
#     plt.legend()
#     plt.savefig(os.path.join(OUTPUT_DIR, 'P_curve.png'), dpi=150, bbox_inches='tight')
#     plt.close()
    
#     # 【R曲线】Recall vs Confidence Score（累积方式）
#     plt.figure(figsize=(10, 6))
#     plt.plot(conf_scores, recalls, 'g-s', linewidth=2, markersize=3, label='Recall')
#     plt.xlabel('Confidence Score')
#     plt.ylabel('Recall')
#     plt.title('Recall-Confidence Curve (Cumulative)')
#     plt.grid(True, alpha=0.3)
#     plt.xlim(0, 1.0)
#     plt.ylim(0, 1.0)
#     plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Default(0.5)')
#     plt.legend()
#     plt.savefig(os.path.join(OUTPUT_DIR, 'R_curve.png'), dpi=150, bbox_inches='tight')
#     plt.close()
    
#     # 【F1曲线】F1 Score vs Confidence Score（累积方式）
#     plt.figure(figsize=(10, 6))
#     plt.plot(conf_scores, f1_scores, 'm-^', linewidth=2, markersize=3, label='F1 Score')
#     plt.xlabel('Confidence Score')
#     plt.ylabel('F1 Score')
#     plt.title('F1 Score-Confidence Curve (Cumulative)')
#     plt.grid(True, alpha=0.3)
#     plt.xlim(0, 1.0)
#     plt.ylim(0, 1.0)
    
#     # 标注最佳F1点
#     if len(f1_scores) > 0:
#         max_idx = np.argmax(f1_scores)
#         best_conf = conf_scores[max_idx]
#         best_f1 = f1_scores[max_idx]
#         plt.axvline(x=best_conf, color='red', linestyle='--', alpha=0.7, 
#                    label=f'Best F1={best_f1:.4f} @ conf={best_conf:.2f}')
#         plt.plot(best_conf, best_f1, 'ro', markersize=8)
#         plt.legend()
    
#     plt.savefig(os.path.join(OUTPUT_DIR, 'F1_curve.png'), dpi=150, bbox_inches='tight')
#     plt.close()
    
#     # 【PR曲线】Precision-Recall Curve
#     plt.figure(figsize=(10, 6))
#     colors = ['r', 'b']
#     ap_values = []
    
#     for i, cls_id in enumerate(PERSON_CLASS_IDS):
#         if cls_id in pr_data:
#             recalls_cls = pr_data[cls_id]['recall']
#             precisions_cls = pr_data[cls_id]['precision']
#             ap = pr_data[cls_id]['ap']
#             ap_values.append(ap)
            
#             if len(recalls_cls) > 0 and len(precisions_cls) > 0:
#                 plt.plot(recalls_cls, precisions_cls, colors[i], linewidth=2, 
#                         label=f'{class_names[i]} (AP={ap:.4f})')
    
#     # 计算 mAP
#     map_value = np.mean(ap_values) if ap_values else 0
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title(f'PR Curve (mAP={map_value:.4f})')
#     plt.grid(True, alpha=0.3)
#     plt.xlim(0, 1.0)
#     plt.ylim(0, 1.0)
#     plt.legend()
#     plt.savefig(os.path.join(OUTPUT_DIR, 'PR_curve.png'), dpi=150, bbox_inches='tight')
#     plt.close()
    
#     # ========== 混淆矩阵 ==========
#     print("计算混淆矩阵...")
    
#     y_true, y_pred = [], []
#     for preds, gts in zip(all_predictions, all_ground_truth):
#         _, _, _, match_details = match_preds_to_gts(preds, gts, IOU_THRESHOLD)
#         matched_gt = set()
        
#         for item in match_details:
#             if item['is_tp']:
#                 y_true.append(item['gt_cls'])
#                 y_pred.append(item['pred_cls'])
#             else:
#                 y_true.append(-1)
#                 y_pred.append(item['pred_cls'])
        
#         for gt_idx, gt in enumerate(gts):
#             if gt_idx not in matched_gt:
#                 y_true.append(gt[0])
#                 y_pred.append(-1)
    
#     cm = confusion_matrix(y_true, y_pred, labels=PERSON_CLASS_IDS)
    
#     # 混淆矩阵
#     fig, ax = plt.subplots(figsize=(8, 6))
#     im = ax.imshow(cm, cmap=plt.cm.Blues)
#     ax.figure.colorbar(im, ax=ax)
#     ax.set(xticks=np.arange(len(PERSON_CLASS_IDS)),
#            yticks=np.arange(len(PERSON_CLASS_IDS)),
#            xticklabels=class_names, yticklabels=class_names,
#            title='Confusion Matrix', ylabel='True', xlabel='Predicted')
#     plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
#     thresh = cm.max() / 2.
#     for i in range(len(PERSON_CLASS_IDS)):
#         for j in range(len(PERSON_CLASS_IDS)):
#             ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
#                     color="white" if cm[i, j] > thresh else "black")
#     fig.tight_layout()
#     plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
#     plt.close()
    
#     # 归一化混淆矩阵
#     with np.errstate(divide='ignore', invalid='ignore'):
#         cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         cm_norm = np.nan_to_num(cm_norm)
    
#     fig, ax = plt.subplots(figsize=(8, 6))
#     im = ax.imshow(cm_norm, cmap=plt.cm.Blues)
#     ax.figure.colorbar(im, ax=ax)
#     ax.set(xticks=np.arange(len(PERSON_CLASS_IDS)),
#            yticks=np.arange(len(PERSON_CLASS_IDS)),
#            xticklabels=class_names, yticklabels=class_names,
#            title='Normalized Confusion Matrix', ylabel='True', xlabel='Predicted')
#     plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
#     thresh = cm_norm.max() / 2.
#     for i in range(len(PERSON_CLASS_IDS)):
#         for j in range(len(PERSON_CLASS_IDS)):
#             ax.text(j, i, format(cm_norm[i, j], '.2f'), ha="center", va="center",
#                     color="white" if cm_norm[i, j] > thresh else "black")
#     fig.tight_layout()
#     plt.savefig(os.path.join(OUTPUT_DIR, 'normalized_confusion_matrix.png'), dpi=150, bbox_inches='tight')
#     plt.close()
    
#     # ========== 保存评测报告 ==========
#     report_path = os.path.join(OUTPUT_DIR, 'evaluation_report.txt')
    
#     # 找到最佳 F1 置信度
#     best_conf_score = conf_scores[np.argmax(f1_scores)] if len(f1_scores) > 0 else 0.5
#     best_f1_score = max(f1_scores) if len(f1_scores) > 0 else 0
    
#     with open(report_path, 'w') as f:
#         f.write("=" * 60 + "\n")
#         f.write("安全带检测模型评测报告（附带校准修正逻辑）\n")
#         f.write("=" * 60 + "\n\n")
#         f.write(f"测试图片数量：{len(image_files)}\n")
#         f.write(f"IoU 阈值：{IOU_THRESHOLD}\n\n")
#         f.write("总体评测指标:\n")
#         f.write(f"  TP={total_tp}, FP={total_fp}, FN={total_fn}\n")
#         f.write(f"  Precision={precision:.4f}\n")
#         f.write(f"  Recall={recall:.4f}\n")
#         f.write(f"  F1 Score={f1:.4f}\n\n")
#         f.write("各类别 AP:\n")
#         for i, cls_id in enumerate(PERSON_CLASS_IDS):
#             if cls_id in pr_data:
#                 f.write(f"  {class_names[i]}: AP={pr_data[cls_id]['ap']:.4f}\n")
#         f.write(f"  mAP={map_value:.4f}\n\n")
#         f.write("最佳置信度分数:\n")
#         f.write(f"  Best Confidence Score={best_conf_score:.2f}\n")
#         f.write(f"  Best F1 Score={best_f1_score:.4f}\n\n")
#         f.write("输出文件:\n")
#         f.write("  - P_curve.png (Precision vs Confidence)\n")
#         f.write("  - R_curve.png (Recall vs Confidence)\n")
#         f.write("  - PR_curve.png (Precision-Recall Curve)\n")
#         f.write("  - F1_curve.png (F1 Score vs Confidence)\n")
#         f.write("  - confusion_matrix.png\n")
#         f.write("  - normalized_confusion_matrix.png\n")
#         f.write("  - images/pred_*.jpg (推理结果图片)\n")
#         f.write("  - infos/pred_*.json (推理结果数据)\n")
    
#     print(f"\n评测完成！结果已保存到：{OUTPUT_DIR}")
#     print(f"最佳置信度分数：{best_conf_score:.2f} (F1={best_f1_score:.4f})")

# if __name__ == "__main__":
#     evaluate()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==================== 配置路径 ====================
TEST_IMAGES_DIR = "/home/ubuntu/graduation_design/test/images"
TEST_LABELS_DIR = "/home/ubuntu/graduation_design/test/labels"
OUTPUT_DIR = "/home/ubuntu/graduation_design/test_result_v2"
IMAGES_DIR = os.path.join(OUTPUT_DIR, 'images')
INFOS_DIR = os.path.join(OUTPUT_DIR, 'infos')
IOU_THRESHOLD = 0.5

# ==================== 类别配置（与 seatbelt_detector.py 一致） ====================
# 类别映射：0=person-noseatbelt, 1=person-seatbelt
CLASS_MAP = {0: 'person-noseatbelt', 1: 'person-seatbelt'}
PERSON_CLASS_IDS = [0, 1]

# ==================== 工具函数 ====================
def parse_yolo_label(label_path, img_w, img_h):
    """解析 YOLO 格式标注，返回 [cls, x1, y1, x2, y2]"""
    annotations = []
    if not os.path.exists(label_path):
        return annotations
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            if cls_id in PERSON_CLASS_IDS:
                x_c, y_c, w, h = map(float, parts[1:5])
                x1 = (x_c - w/2) * img_w
                y1 = (y_c - h/2) * img_h
                x2 = (x_c + w/2) * img_w
                y2 = (y_c + h/2) * img_h
                annotations.append([cls_id, x1, y1, x2, y2])
    return annotations

def calc_iou(box1, box2):
    """计算两个边界框的 IoU"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    inter_x1, inter_y1 = max(x1_1, x1_2), max(y1_1, y1_2)
    inter_x2, inter_y2 = min(x2_1, x2_2), min(y2_1, y2_2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def match_preds_to_gts(predictions, ground_truth, iou_thresh=0.5):
    """
    匹配预测与真实标注（增加类别一致性校验）
    返回：TP 数量，FP 数量，FN 数量，以及详细匹配信息
    """
    matched_gt = set()
    tp, fp = 0, 0
    match_details = []
    
    # 按置信度降序处理预测
    sorted_preds = sorted(predictions, key=lambda x: x.get('conf', 0.5), reverse=True)
    
    for pred in sorted_preds:
        pred_cls = pred['cls']
        pred_bbox = pred['bbox']
        pred_conf = pred.get('conf', 0.5)
        
        best_iou, best_gt_idx, best_gt_cls = 0, -1, -1
        
        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in matched_gt:
                continue
            gt_cls = gt[0]
            # 核心修改：先校验类别是否一致，不一致则跳过
            if gt_cls != pred_cls:
                continue
            gt_bbox = gt[1:]
            iou = calc_iou(pred_bbox, gt_bbox)
            if iou > best_iou:
                best_iou, best_gt_idx, best_gt_cls = iou, gt_idx, gt_cls
        
        if best_iou >= iou_thresh:
            tp += 1
            matched_gt.add(best_gt_idx)
            match_details.append({
                'pred_cls': pred_cls,
                'gt_cls': best_gt_cls,
                'conf': pred_conf,
                'iou': best_iou,
                'is_tp': True
            })
        else:
            fp += 1
            match_details.append({
                'pred_cls': pred_cls,
                'gt_cls': -1,
                'conf': pred_conf,
                'iou': best_iou,
                'is_tp': False
            })
    
    fn = len(ground_truth) - len(matched_gt)
    
    return tp, fp, fn, match_details

def calculate_metrics_vs_confidence(all_predictions, all_ground_truth, iou_thresh=0.5, n_bins=50):
    """
    计算不同置信度分数下的 P/R/F1 指标
    横坐标：置信度分数（0-1）
    纵坐标：该置信度水平下的累积 Precision/Recall/F1
    """
    # 收集所有预测及其匹配结果
    all_pred_items = []
    total_gts = 0
    
    for preds, gts in zip(all_predictions, all_ground_truth):
        total_gts += len(gts)
        _, _, _, match_details = match_preds_to_gts(preds, gts, iou_thresh)
        for item in match_details:
            all_pred_items.append(item)
    
    if len(all_pred_items) == 0:
        return [], [], [], []
    
    # 按置信度排序
    all_pred_items.sort(key=lambda x: x['conf'])
    
    # 生成置信度分数点（从 0 到 1）
    confidence_scores = np.linspace(0, 1, n_bins)
    
    precisions, recalls, f1_scores = [], [], []
    
    for conf_score in confidence_scores:
        # 统计置信度 <= conf_score 的预测
        tp_count = sum(1 for item in all_pred_items if item['conf'] <= conf_score and item['is_tp'])
        fp_count = sum(1 for item in all_pred_items if item['conf'] <= conf_score and not item['is_tp'])
        
        # Recall 基于总 GT 数
        fn_count = total_gts - tp_count
        
        p = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
        r = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        
        precisions.append(p)
        recalls.append(r)
        f1_scores.append(f1)
    
    return confidence_scores, precisions, recalls, f1_scores

def calculate_pr_curve(all_predictions, all_ground_truth, iou_thresh=0.5):
    """
    计算 PR 曲线（按置信度阈值）
    返回：每个类别的 precision, recall, ap
    """
    # 收集所有预测及其 GT 信息
    all_pred_items = []
    total_gts = {cls_id: 0 for cls_id in PERSON_CLASS_IDS}
    
    for preds, gts in zip(all_predictions, all_ground_truth):
        # 统计 GT 数量
        for gt in gts:
            if gt[0] in total_gts:
                total_gts[gt[0]] += 1
        
        # 匹配预测
        _, _, _, match_details = match_preds_to_gts(preds, gts, iou_thresh)
        for item in match_details:
            all_pred_items.append(item)
    
    # 按置信度降序排序
    all_pred_items.sort(key=lambda x: x['conf'], reverse=True)
    
    # 计算每类的 PR 曲线
    pr_data = {}
    
    for cls_id in PERSON_CLASS_IDS:
        tp_cumsum = 0
        fp_cumsum = 0
        
        precisions = []
        recalls = []
        confidences = []
        
        for item in all_pred_items:
            if item['pred_cls'] == cls_id:
                confidences.append(item['conf'])
                if item['is_tp']:
                    tp_cumsum += 1
                else:
                    fp_cumsum += 1
                
                precision = tp_cumsum / (tp_cumsum + fp_cumsum)
                recall = tp_cumsum / total_gts[cls_id] if total_gts[cls_id] > 0 else 0
                
                precisions.append(precision)
                recalls.append(recall)
        
        # 计算 AP（使用积分法）
        if len(precisions) > 0 and len(recalls) > 0:
            # 确保 recall 单调递增，precision 取后续最大值
            precisions_interp = []
            for i in range(len(precisions)):
                precisions_interp.append(max(precisions[i:]))
            ap = np.trapz(precisions_interp, recalls)
        else:
            ap = 0.0
            precisions = [0]
            recalls = [0]
        
        pr_data[cls_id] = {
            'precision': precisions,
            'recall': recalls,
            'confidence': confidences,
            'ap': ap
        }
    
    return pr_data

# ==================== 主评测函数 ====================
def evaluate():
    """主评测函数：使用 seatbelt_detector.py 进行推理并评测"""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(INFOS_DIR, exist_ok=True)
    
    # 获取测试图片列表
    image_files = sorted([
        f for f in Path(TEST_IMAGES_DIR).iterdir() 
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
    ])
    
    print(f"找到 {len(image_files)} 张测试图片")
    
    all_predictions, all_ground_truth = [], []
    
    # ========== 使用 seatbelt_detector.py 进行推理 ==========
    import time
    total_inference_time = 0.0
    
    for idx, img_path in enumerate(image_files):
        print(f"处理 [{idx+1}/{len(image_files)}]: {img_path.name}")
        
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        h, w = image.shape[:2]
        
        # 记录推理开始时间
        start_time = time.time()
        
        # 使用 seatbelt_detector.py 进行检测
        result = detect_single_frame(image)
        
        # 记录推理结束时间
        end_time = time.time()
        inference_time = end_time - start_time
        total_inference_time += inference_time
        
        # 保存推理结果图片到 test_result_v3/images
        cv2.imwrite(os.path.join(IMAGES_DIR, f"pred_{img_path.name}"), result['frame'])
        
        # 使用 seatbelt_detector.py 返回的检测结果
        predictions = result['results']
        
        # 读取真实标注
        label_path = os.path.join(TEST_LABELS_DIR, img_path.stem + ".txt")
        ground_truth = parse_yolo_label(label_path, w, h)
        
        all_predictions.append(predictions)
        all_ground_truth.append(ground_truth)
        
        # 保存预测结果为 JSON 到 test_result_v3/infos
        with open(os.path.join(INFOS_DIR, f"pred_{img_path.stem}.json"), 'w') as f:
            json.dump(predictions, f, indent=2)
    
    avg_inference_time = total_inference_time / len(image_files) if image_files else 0
    
    # ========== 计算评测指标（IoU=0.5, 默认置信度） ==========
    print("\n计算评测指标...")
    
    total_tp, total_fp, total_fn = 0, 0, 0
    for preds, gts in zip(all_predictions, all_ground_truth):
        tp, fp, fn, _ = match_preds_to_gts(preds, gts, IOU_THRESHOLD)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n总体评测结果 (IoU={IOU_THRESHOLD}):")
    print(f"  TP={total_tp}, FP={total_fp}, FN={total_fn}")
    print(f"  Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    
    # ========== 计算 P/R/F1 vs 置信度分数（累积方式） ==========
    print("\n计算置信度分数曲线（累积方式）...")
    conf_scores, precisions, recalls, f1_scores = calculate_metrics_vs_confidence(
        all_predictions, all_ground_truth, IOU_THRESHOLD, n_bins=50
    )
    
    # ========== 计算 PR 曲线 ==========
    print("计算 PR 曲线...")
    pr_data = calculate_pr_curve(all_predictions, all_ground_truth, IOU_THRESHOLD)
    
    # ========== 绘制曲线 ==========
    print("\n绘制评测曲线...")
    class_names = ['person-noseatbelt', 'person-seatbelt']
    
    # 【P 曲线】Precision vs Confidence Score（累积方式）
    plt.figure(figsize=(10, 6))
    plt.plot(conf_scores, precisions, 'b-o', linewidth=2, markersize=3, label='Precision')
    plt.xlabel('Confidence Score')
    plt.ylabel('Precision')
    plt.title('Precision-Confidence Curve (Cumulative)')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Default(0.5)')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'P_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 【R 曲线】Recall vs Confidence Score（累积方式）
    plt.figure(figsize=(10, 6))
    plt.plot(conf_scores, recalls, 'g-s', linewidth=2, markersize=3, label='Recall')
    plt.xlabel('Confidence Score')
    plt.ylabel('Recall')
    plt.title('Recall-Confidence Curve (Cumulative)')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Default(0.5)')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'R_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 【F1 曲线】F1 Score vs Confidence Score（累积方式）
    plt.figure(figsize=(10, 6))
    plt.plot(conf_scores, f1_scores, 'm-^', linewidth=2, markersize=3, label='F1 Score')
    plt.xlabel('Confidence Score')
    plt.ylabel('F1 Score')
    plt.title('F1 Score-Confidence Curve (Cumulative)')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    
    # 标注最佳 F1 点
    if len(f1_scores) > 0:
        max_idx = np.argmax(f1_scores)
        best_conf = conf_scores[max_idx]
        best_f1 = f1_scores[max_idx]
        plt.axvline(x=best_conf, color='red', linestyle='--', alpha=0.7, 
                   label=f'Best F1={best_f1:.4f} @ conf={best_conf:.2f}')
        plt.plot(best_conf, best_f1, 'ro', markersize=8)
        plt.legend()
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'F1_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 【PR 曲线】Precision-Recall Curve
    plt.figure(figsize=(10, 6))
    colors = ['r', 'b']
    ap_values = []
    
    for i, cls_id in enumerate(PERSON_CLASS_IDS):
        if cls_id in pr_data:
            recalls_cls = pr_data[cls_id]['recall']
            precisions_cls = pr_data[cls_id]['precision']
            ap = pr_data[cls_id]['ap']
            ap_values.append(ap)
            
            if len(recalls_cls) > 0 and len(precisions_cls) > 0:
                plt.plot(recalls_cls, precisions_cls, colors[i], linewidth=2, 
                        label=f'{class_names[i]} (AP={ap:.4f})')
    
    # 计算 mAP
    map_value = np.mean(ap_values) if ap_values else 0
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PR Curve (mAP={map_value:.4f})')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'PR_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # ========== 混淆矩阵（修改部分） ==========
    print("计算混淆矩阵...")
    
    y_true, y_pred = [], []
    matched_gt_global = set()  # 全局记录已匹配的 GT
    
    for preds, gts in zip(all_predictions, all_ground_truth):
        # 重新进行匹配，但不校验类别，只校验 IoU
        sorted_preds = sorted(preds, key=lambda x: x.get('conf', 0.5), reverse=True)
        matched_gt_local = set()
        
        for pred in sorted_preds:
            pred_cls = pred['cls']
            pred_bbox = pred['bbox']
            
            best_iou, best_gt_idx, best_gt_cls = 0, -1, -1
            
            for gt_idx, gt in enumerate(gts):
                if gt_idx in matched_gt_local:
                    continue
                gt_cls = gt[0]
                gt_bbox = gt[1:]
                iou = calc_iou(pred_bbox, gt_bbox)
                if iou > best_iou:
                    best_iou, best_gt_idx, best_gt_cls = iou, gt_idx, gt_cls
            
            # 所有 IoU≥阈值的预测都纳入混淆矩阵（无论类别是否正确）
            if best_iou >= IOU_THRESHOLD:
                matched_gt_local.add(best_gt_idx)
                matched_gt_global.add((id(gts), best_gt_idx))  # 记录已匹配的 GT
                y_true.append(best_gt_cls)  # 真实类别
                y_pred.append(pred_cls)     # 预测类别（可能错误）
            else:
                # IoU<阈值：FP（误检），真实类别为 -1（背景）
                y_true.append(-1)
                y_pred.append(pred_cls)
        
        # FN（漏检）：有 GT 但无预测匹配
        for gt_idx, gt in enumerate(gts):
            if gt_idx not in matched_gt_local:
                y_true.append(gt[0])  # 真实类别
                y_pred.append(-1)     # 预测为 -1（漏检）
    
    # 修改：labels 包含 -1，使 FP 和 FN 也能在矩阵中体现
    cm_labels = PERSON_CLASS_IDS + [-1]
    cm_labels_names = class_names + ['Background']
    
    cm = confusion_matrix(y_true, y_pred, labels=cm_labels)
    
    # 混淆矩阵
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(cm_labels)),
           yticks=np.arange(len(cm_labels)),
           xticklabels=cm_labels_names, yticklabels=cm_labels_names,
           title='Confusion Matrix (Including FP & FN)', ylabel='True', xlabel='Predicted')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2.
    for i in range(len(cm_labels)):
        for j in range(len(cm_labels)):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 归一化混淆矩阵
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(cm_labels)),
           yticks=np.arange(len(cm_labels)),
           xticklabels=cm_labels_names, yticklabels=cm_labels_names,
           title='Normalized Confusion Matrix (Including FP & FN)', ylabel='True', xlabel='Predicted')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm_norm.max() / 2.
    for i in range(len(cm_labels)):
        for j in range(len(cm_labels)):
            ax.text(j, i, format(cm_norm[i, j], '.2f'), ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'normalized_confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # ========== 保存评测报告 ==========
    report_path = os.path.join(OUTPUT_DIR, 'evaluation_report.txt')
    
    # 找到最佳 F1 置信度
    best_conf_score = conf_scores[np.argmax(f1_scores)] if len(f1_scores) > 0 else 0.5
    best_f1_score = max(f1_scores) if len(f1_scores) > 0 else 0
    
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("安全带检测模型评测报告（优化检测）\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"测试图片数量：{len(image_files)}\n")
        f.write(f"IoU 阈值：{IOU_THRESHOLD}\n")
        f.write(f"平均每张图片推理耗时：{avg_inference_time:.4f}秒\n\n")
        f.write("总体评测指标:\n")
        f.write(f"  TP={total_tp}, FP={total_fp}, FN={total_fn}\n")
        f.write(f"  Precision={precision:.4f}\n")
        f.write(f"  Recall={recall:.4f}\n")
        f.write(f"  F1 Score={f1:.4f}\n\n")
        f.write("各类别 AP:\n")
        for i, cls_id in enumerate(PERSON_CLASS_IDS):
            if cls_id in pr_data:
                f.write(f"  {class_names[i]}: AP={pr_data[cls_id]['ap']:.4f}\n")
        f.write(f"  mAP={map_value:.4f}\n\n")
        f.write("最佳置信度分数:\n")
        f.write(f"  Best Confidence Score={best_conf_score:.2f}\n")
        f.write(f"  Best F1 Score={best_f1_score:.4f}\n\n")
        f.write("输出文件:\n")
        f.write("  - P_curve.png (Precision vs Confidence)\n")
        f.write("  - R_curve.png (Recall vs Confidence)\n")
        f.write("  - PR_curve.png (Precision-Recall Curve)\n")
        f.write("  - F1_curve.png (F1 Score vs Confidence)\n")
        f.write("  - confusion_matrix.png\n")
        f.write("  - normalized_confusion_matrix.png\n")
        f.write("  - images/pred_*.jpg (推理结果图片)\n")
        f.write("  - infos/pred_*.json (推理结果数据)\n")
    
    print(f"\n评测完成！结果已保存到：{OUTPUT_DIR}")
    print(f"最佳置信度分数：{best_conf_score:.2f} (F1={best_f1_score:.4f})")
    print(f"平均每张图片推理耗时：{avg_inference_time:.4f}秒")

if __name__ == "__main__":
    evaluate()