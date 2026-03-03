from ultralytics import RTDETR
import os

# 加载本地预训练模型（使用更大的预训练模型）
model = RTDETR("/home/ubuntu/graduation_design/rtdetr-l.pt")  

# 配置训练参数并开始训练
results = model.train(
    data="/home/ubuntu/graduation_design/seatbelt_detection/project03/data.yaml",  # 数据集配置文件
    epochs=100,        # 训练轮数（保持100轮，足够收敛）
    imgsz=800,         # 输入图片尺寸（提升至800，平衡精度与计算效率）
    batch=16,          # 批次大小（基于GPU显存优化）
    device=0,          # 使用GPU训练
    val=True,          # 启用验证集评估
    save=True,         # 保存模型权重
    name="seatbelt_detection_train",  # 训练任务名称
    pretrained=True,   # 使用预训练权重
    patience=15,       # 早停耐心值（连续15轮验证指标不提升后停止）
    optimizer="AdamW", # 使用AdamW优化器（比SGD收敛更快）
    lr0=0.0001,        # 初始学习率（0.0001，更精细的调整）
    lrf=0.01,          # 最终学习率比例（0.01）
    momentum=0.937,    # 动量（提升收敛稳定性）
    weight_decay=0.0005, # 权重衰减（防止过拟合）
    hsv_h=0.015,       # 色相增强范围（更精细的色彩调整）
    hsv_s=0.7,         # 饱和度增强范围
    hsv_v=0.4,         # 明度增强范围
    degrees=10.0,      # 旋转角度范围
    translate=0.1,     # 平移比例
    scale=0.5,         # 缩放范围
    shear=0.0,         # 剪切角度
    perspective=0.0,   # 透视变换
    mixup=0.25,        # Mixup增强比例
    copy_paste=0.2,    # Copy-Paste增强比例
    mosaic=1.0,        # Mosaic增强比例
    auto_augment="randaugment",  # 使用RandAugment增强
    cos_lr=True,       # 启用余弦退火学习率
    close_mosaic=10,   # 最后10轮关闭Mosaic增强
    augment=False,     # 保持默认增强（由上述参数控制）
    profile=False,     # 关闭性能分析
    plots=True,        # 生成训练曲线图
    save_period=5,     # 每5轮保存一次权重
    workers=8,         # 数据加载工作进程数
    cache = "ram"      # 将数据集缓存到RAM中（加速数据加载，需足够内存支持
)

# 训练完成后可自动生成训练日志、权重文件（保存在runs/detect/seatbelt_detection_train目录）
print("\nTraining completed successfully!")
print("Best model saved at: runs/detect/seatbelt_detection_train/weights/best.pt")
print("Training logs available at: runs/detect/seatbelt_detection_train")