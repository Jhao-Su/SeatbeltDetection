from ultralytics import RTDETR
import os
import shutil

# 配置路径
DATA_YAML = "/home/ubuntu/graduation_design/seatbelt_detection/project03/data.yaml"
PRETRAINED_WEIGHTS = "/home/ubuntu/graduation_design/rtdetr-l.pt"
PROJECT_NAME = "seatbelt_detection_train"

# 1. 【重要】清理旧的损坏缓存，强制重新扫描
cache_files = [
    "/home/ubuntu/graduation_design/seatbelt_detection/project03/train/labels.cache",
    "/home/ubuntu/graduation_design/seatbelt_detection/project03/valid/labels.cache"
]
for f in cache_files:
    if os.path.exists(f):
        os.remove(f)
        print(f"Removed old cache: {f}")

# 2. 加载模型
model = RTDETR(PRETRAINED_WEIGHTS)

# 3. 执行训练 (针对上述问题的修正版)
results = model.train(
    data=DATA_YAML,
    
    # --- 基础设置 ---
    epochs=100,
    imgsz=800,
    batch=8,              # 【修正】显存受限，手动设为8以保证稳定，不再依赖自动降级
    device=0,
    workers=8,
    pretrained=True,
    patience=20,          # 增加耐心，防止因初期震荡过早停止
    seed=0,
    deterministic=True,
    
    # --- 优化器设置 (关键修正) ---
    optimizer='AdamW',    # 【修正】显式指定，防止 'auto' 覆盖学习率
    lr0=0.00005,          # 【修正】强制使用小学习率微调
    lrf=0.005,            # 最终学习率比例
    momentum=0.937,       # 【修正】强制指定动量
    weight_decay=0.0005,
    warmup_epochs=5.0,    # 增加预热轮数，适应小Batch
    warmup_momentum=0.8,  # 预热动量
    warmup_bias_lr=0.1,   # 预热偏置学习率
    
    # --- 损失函数权重 (针对安全带小目标) ---
    cls=0.7,              # 提高分类权重
    dfl=1.8,              # 提高分布焦点损失
    box=7.5,              # 提高边界框损失权重
    
    # --- 数据增强 (保守策略，保护小目标特征) ---
    hsv_h=0.01,           # 减小色相增强范围
    hsv_s=0.3,            # 减小饱和度增强范围
    hsv_v=0.2,            # 减小明度增强范围
    degrees=5.0,          # 减小旋转角度范围
    translate=0.05,       # 减小平移比例
    scale=0.3,            # 减小缩放范围
    shear=0.0,            # 不使用剪切
    perspective=0.0,      # 减小Mosaic和Mixup增强比例，保护小目标特征
    flipud=0.0,           # 不使用垂直翻转，避免不自然的场景
    fliplr=0.3,           # 降低翻转概率
    mosaic=0.8,           # 降低Mosaic
    mixup=0.15,
    copy_paste=0.1,       # 降低Copy-Paste增强比例
    auto_augment='randaugment',          # 使用RandAugment增强，提供多样化但受控的增强策略
    erasing=0.4,          # 增加随机擦除，提升模型鲁棒性
    
    # --- 训练策略 ---
    cos_lr=True,
    close_mosaic=15,      # 最后15轮关闭Mosaic，精细化收敛
    amp=True,             # 混合精度加速
    cache='disk',         # 使用磁盘缓存避免内存溢出，且重新生成
    rect=False,           # 不使用矩形训练，保持输入尺寸一致，保护小目标特征
    
    # --- 输出设置 ---
    project="/home/ubuntu/graduation_design/seatbelt_detection/project03/runs/detect",
    name=PROJECT_NAME,
    exist_ok=False,
    save=True,
    val=True,
    plots=True,
    verbose=True,
    
    # --- 关键：关闭自动优化器选择 ---
    # 在ultralytics中，只要显式写了 optimizer='AdamW' 和 lr0，通常不会触发auto
    # 但为了保险，我们确保不传 'optimizer=auto' 字符串
)

print("\n" + "="*50)
print("训练完成！请检查以下事项：")
print("1. 确认日志中不再有 'optimizer=auto' 提示。")
print("2. 确认 Batch size 稳定在 8，无 OOM 重试。")
print("3. 确认不再有 'Box and segment counts' 警告（若仍有，需清洗标签）。")
print("="*50)