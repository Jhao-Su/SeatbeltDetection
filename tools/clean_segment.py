import os
import glob

def clean_labels(label_dir):
    txt_files = glob.glob(os.path.join(label_dir, "*.txt"))
    cleaned_count = 0
    
    for txt_file in txt_files:
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        modified = False
        
        for line in lines:
            parts = list(map(float, line.strip().split()))
            # 标准检测格式应该是 5 个数: class x y w h
            # 如果超过 5 个，说明后面跟着分割点，需要截断
            if len(parts) > 5:
                # 保留前5个
                new_line = " ".join(map(str, parts[:5]))
                new_lines.append(new_line + "\n")
                modified = True
            else:
                new_lines.append(line)
        
        if modified:
            with open(txt_file, 'w') as f:
                f.writelines(new_lines)
            cleaned_count += 1
            
    print(f"清洗完成：处理了 {len(txt_files)} 个文件，修正了 {cleaned_count} 个包含多余分割数据的文件。")

# 分别清洗训练集和验证集
train_label_dir = "/home/ubuntu/graduation_design/seatbelt_detection/project03/train/labels"
val_label_dir = "/home/ubuntu/graduation_design/seatbelt_detection/project03/valid/labels"

if os.path.exists(train_label_dir):
    clean_labels(train_label_dir)
if os.path.exists(val_label_dir):
    clean_labels(val_label_dir)