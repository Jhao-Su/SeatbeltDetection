from ultralytics import RTDETR

# 加载本地预训练模型
model = RTDETR("/home/ubuntu/graduation_design/rtdetr-l.pt")  

# 配置训练参数并开始训练
# 核心参数说明：
# - data: 数据集配置文件data.yaml的路径（需正确关联train/valid的图片和标签路径）
# - epochs: 训练轮数（根据需求调整，如50、100等）
# - imgsz: 输入图片尺寸（默认640，可根据硬件调整）
# - batch: 批次大小（根据GPU显存调整，如8、16）
# - device: 训练设备（0表示第1块GPU，cpu表示CPU）
results = model.train(
    data="/home/ubuntu/graduation_design/seatbelt_detection/project03/data.yaml",  # 数据集配置文件
    epochs=100,        # 训练轮数
    imgsz=640,         # 输入图片尺寸
    batch=16,          # 批次大小
    device=0,          # 使用GPU训练（若无GPU可改为"cpu"）
    val = True,  # 启用验证集评估
    save=True,  # 保存模型权重
    name="seatbelt_detection_train",  # 训练任务名称（用于保存日志和权重）
    pretrained=True,    # 使用预训练权重（默认True，此处显式指定）
    patience=15  # 早停耐心值（连续多少轮验证指标不提升后停止训练）
)

# 训练完成后可自动生成训练日志、权重文件（保存在runs/detect/seatbelt_detection_train目录）