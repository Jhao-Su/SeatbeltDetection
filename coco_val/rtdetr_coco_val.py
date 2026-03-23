from ultralytics.data.utils import download

from ultralytics import RTDETR

model = RTDETR("/home/ubuntu/graduation_design/rtdetr-l.pt")

model.val(
    data="/home/ubuntu/graduation_design/coco_val/valid/data_converted.yaml", 
    imgsz=640, 
    device=0,
    batch=16,
    conf=0.001, 
)