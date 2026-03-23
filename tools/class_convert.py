#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
转换标注文件中的类别 ID
从 data.yaml（字母顺序）转换为 model_classes_map.txt
"""

from pathlib import Path
import yaml
import shutil
from datetime import datetime

# ============ 路径配置 ============
DATA_YAML_PATH = Path("/home/ubuntu/graduation_design/coco_val/data.yaml")
MODEL_CLASSES_PATH = Path("/home/ubuntu/graduation_design/model_classes_map.txt")
LABELS_SRC_DIR = Path("/home/ubuntu/graduation_design/coco_val/valid/labels")
LABELS_DST_DIR = Path("/home/ubuntu/graduation_design/coco_val/valid/labels_converted")

# ============ 读取类别映射 ============
print("=" * 70)
print("🔄 YOLO 标注类别 ID 转换脚本")
print("=" * 70)

# 1. 读取 data.yaml（字母顺序）
print("\n1️⃣ 读取 data.yaml（原始字母顺序）...")
with open(DATA_YAML_PATH, 'r', encoding='utf-8') as f:
    data_yaml = yaml.safe_load(f)

alpha_names = data_yaml.get('names', [])
if isinstance(alpha_names, dict):
    alpha_names = [alpha_names[i] for i in sorted(alpha_names.keys())]
print(f"   类别数：{len(alpha_names)}")
print(f"   前 5 个：{alpha_names[:5]}")

# 2. 读取 model_classes_map.txt（COCO 标准顺序）
print("\n2️⃣ 读取 model_classes_map.txt（COCO 标准顺序）...")
coco_names = []
with open(MODEL_CLASSES_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            parts = line.split(': ', 1)
            if len(parts) == 2:
                coco_names.append(parts[1])
print(f"   类别数：{len(coco_names)}")
print(f"   前 5 个：{coco_names[:5]}")

# 3. 建立映射：字母顺序 ID → COCO 标准 ID
print("\n3️⃣ 建立类别 ID 映射...")
name_to_coco_id = {name: i for i, name in enumerate(coco_names)}
alpha_to_coco_map = {}

print(f"   {'字母序ID':<10} {'类别名':<18} {'→':<3} {'COCO标准ID':<10}")
print(f"   {'-'*10} {'-'*18} {'-'*3} {'-'*10}")

mismatch_count = 0
for alpha_id, name in enumerate(alpha_names):
    if name in name_to_coco_id:
        coco_id = name_to_coco_id[name]
        alpha_to_coco_map[alpha_id] = coco_id
        if alpha_id != coco_id:
            mismatch_count += 1
        # 只显示不匹配的
        if alpha_id != coco_id:
            print(f"   {alpha_id:<10} {name:<18} → {coco_id:<10}")
    else:
        print(f"   ⚠️  警告：类别 '{name}' 在 model_classes_map 中未找到！")
        alpha_to_coco_map[alpha_id] = alpha_id  # 保持原 ID

print(f"\n   需要转换的类别数：{mismatch_count} / {len(alpha_names)}")

# ============ 转换标注文件 ============
print("\n4️⃣ 开始转换标注文件...")

# 备份原目录
if LABELS_DST_DIR.exists():
    print(f"   清理旧目录：{LABELS_DST_DIR}")
    shutil.rmtree(LABELS_DST_DIR)

LABELS_DST_DIR.mkdir(parents=True, exist_ok=True)

# 统计
total_files = 0
converted_files = 0
total_boxes = 0
converted_boxes = 0

# 获取所有标注文件
label_files = list(LABELS_SRC_DIR.glob("*.txt"))
print(f"   找到标注文件：{len(label_files)} 个")

for txt_file in label_files:
    total_files += 1
    
    # 读取原文件
    with open(txt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 转换类别 ID
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            old_id = int(parts[0])
            new_id = alpha_to_coco_map.get(old_id, old_id)
            
            if old_id != new_id:
                converted_boxes += 1
            
            parts[0] = str(new_id)
            new_lines.append(' '.join(parts) + '\n')
            total_boxes += 1
        elif line.strip():
            new_lines.append(line)
    
    # 保存新文件（保持相同文件名）
    dst_file = LABELS_DST_DIR / txt_file.name
    with open(dst_file, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    converted_files += 1

print(f"\n   ✅ 转换完成！")
print(f"   - 文件总数：{total_files}")
print(f"   - 标注框总数：{total_boxes}")
print(f"   - 转换的标注框数：{converted_boxes}")

# ============ 验证转换结果 ============
print("\n5️⃣ 验证转换结果...")

# 随机检查几个文件
import random
sample_files = random.sample(list(LABELS_DST_DIR.glob("*.txt")), min(3, len(list(LABELS_DST_DIR.glob("*.txt")))))

for sf in sample_files:
    print(f"\n   文件：{sf.name}")
    with open(sf, 'r') as f:
        for i, line in enumerate(f.readlines()[:5]):
            parts = line.strip().split()
            if len(parts) >= 1:
                class_id = int(parts[0])
                class_name = coco_names[class_id] if class_id < len(coco_names) else "N/A"
                print(f"      行{i+1}: ID={class_id} → 类别={class_name}")

# ============ 更新 data.yaml ============
print("\n6️⃣ 生成新的 data.yaml（使用 COCO 标准顺序）...")

new_data_yaml = {
    'path': data_yaml.get('path', '/home/ubuntu/graduation_design/coco_val'),
    'train': data_yaml.get('train', 'valid/images'),
    'val': data_yaml.get('val', 'valid/images'),
    'names': coco_names
}

output_yaml_path = LABELS_SRC_DIR.parent / "data_converted.yaml"
with open(output_yaml_path, 'w', encoding='utf-8') as f:
    f.write(f"# ============================================================\n")
    f.write(f"# COCO 数据集配置文件 (类别顺序已修正为 COCO 标准)\n")
    f.write(f"# 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"# 标注文件位置：{LABELS_DST_DIR}\n")
    f.write(f"# ============================================================\n\n")
    yaml.dump(new_data_yaml, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

print(f"   ✅ 已保存：{output_yaml_path}")

# ============ 完成 ============
print("\n" + "=" * 70)
print("✅ 转换完成！")
print("=" * 70)
print(f"""
📁 转换后的标注文件位置：{LABELS_DST_DIR}
📄 新的配置文件位置：{output_yaml_path}

🔧 下一步操作：
1. 更新 data.yaml 中的路径指向转换后的标注
2. 或删除 runs/detect/val 缓存后重新验证

🧪 验证命令：
from ultralytics import RTDETR
model = RTDETR("/home/ubuntu/graduation_design/rtdetr-l.pt")
results = model.val(data="{output_yaml_path}", plots=True)
""")
print("=" * 70)