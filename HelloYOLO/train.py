from ultralytics import YOLO

# 加载YOLOv8模型
model = YOLO('yolov8n.pt')

if __name__ == '__main__':
    # 模型训练
    results = model.train(
        data=r'C:\Users\penguin\Desktop\code\Practice1.v1-roboflow-instant-1--eval-.yolov8\data.yaml',
        epochs=50,
        batch=16,
        imgsz=640,
        device=0,
        project='runs/train',
        name='exp',
        pretrained=True,
        optimizer='Adam',
        lr0=0.01,
        warmup_epochs=3,
        task='detect'  # 明确指定为检测任务
    )

    # 评估模型性能
    metrics = model.val()