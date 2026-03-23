#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查看 RTDETR 模型类别列表
输出格式：0: 类别1\n1: 类别2\n...
"""

import torch
from pathlib import Path

# 配置路径
MODEL_PATH = Path("/home/ubuntu/graduation_design/rtdetr-l.pt")
OUTPUT_PATH = Path("/home/ubuntu/graduation_design/model_classes_map.txt")

# 加载模型权重
print(f"正在加载模型: {MODEL_PATH}")
checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)

# 获取类别名称
class_names = None

if isinstance(checkpoint, dict):
    if 'model' in checkpoint and hasattr(checkpoint['model'], 'names'):
        class_names = checkpoint['model'].names
    elif 'names' in checkpoint:
        class_names = checkpoint['names']

# 如果是列表，保持；如果是字典，按 key 排序
if isinstance(class_names, dict):
    class_names = [class_names[i] for i in sorted(class_names.keys())]

# 输出到文件
print(f"共找到 {len(class_names)} 个类别")
print(f"正在写入: {OUTPUT_PATH}")

with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    for i, name in enumerate(class_names):
        f.write(f"{i}: {name}\n")

print("✅ 完成！")
print(f"\n类别列表预览:")
for i, name in enumerate(class_names[:10]):
    print(f"  {i}: {name}")
if len(class_names) > 10:
    print(f"  ... 共 {len(class_names)} 个类别")