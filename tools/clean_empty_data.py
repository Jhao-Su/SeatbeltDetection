#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""筛选有效图片 - 标注对"""

from pathlib import Path
import shutil

IMAGES_SRC_DIR = Path("/home/ubuntu/graduation_design/coco_val/valid/images")
LABELS_SRC_DIR = Path("/home/ubuntu/graduation_design/coco_val/valid/labels_converted")
IMAGES_DST_DIR = Path("/home/ubuntu/graduation_design/coco_val/valid/images_fixed")
LABELS_DST_DIR = Path("/home/ubuntu/graduation_design/coco_val/valid/labels_fixed")

print("=" * 70)
print("📁 筛选有效图片 - 标注对")
print("=" * 70)

if IMAGES_DST_DIR.exists():
    shutil.rmtree(IMAGES_DST_DIR)
if LABELS_DST_DIR.exists():
    shutil.rmtree(LABELS_DST_DIR)

IMAGES_DST_DIR.mkdir(parents=True, exist_ok=True)
LABELS_DST_DIR.mkdir(parents=True, exist_ok=True)

image_files = sorted(IMAGES_SRC_DIR.glob("*.jpg"))
valid_pairs = 0
missing_labels = 0
empty_labels = 0

for img_file in image_files:
    label_file = LABELS_SRC_DIR / (img_file.stem + ".txt")
    
    if not label_file.exists():
        missing_labels += 1
        continue
    
    with open(label_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    if not content:
        empty_labels += 1
        continue
    
    shutil.copy2(img_file, IMAGES_DST_DIR / img_file.name)
    shutil.copy2(label_file, LABELS_DST_DIR / label_file.name)
    valid_pairs += 1

print(f"原始图片数：{len(image_files)}")
print(f"有效图片 - 标注对：{valid_pairs}")
print(f"缺失标注文件：{missing_labels}")
print(f"标注文件为空：{empty_labels}")
print(f"\nimages_fixed: {len(list(IMAGES_DST_DIR.glob('*.jpg')))}")
print(f"labels_fixed: {len(list(LABELS_DST_DIR.glob('*.txt')))}")
print("\n✅ 完成！")
print("=" * 70)