from ultralytics import YOLO
import os

# model = YOLO(r'C:\Users\penguin\Desktop\code\runs\train\exp7\weights\best.pt')
model = YOLO(r'C:\Users\penguin\Desktop\code\yolov8n.pt')

source = r"C:\Users\penguin\Desktop\code\test\6.mp4"

results = model.predict(
    source=source, 
    conf=0.25, 
    save=True,  # 保存检测后的视频
    save_txt=True,  # 保存检测结果到txt文件
    iou=0.45, 
    device=0, 
    imgsz=640,
    # 添加视频检测专用参数
    stream=True,  # 流式处理，减少内存占用
    show=False,  # 如果需要实时显示视频窗口可设为True
    save_dir=r'C:\Users\penguin\Desktop\code\runs\detect',  # 检测结果保存目录
    vid_stride=2  # 视频采样间隔，2表示每2帧检测一次，提高速度
)

# 遍历视频检测结果（按帧）
for frame_idx, result in enumerate(results):
    print(f"\n帧序号: {frame_idx}")
    print("视频路径：", result.path)
    print("检测结果：", result.boxes)  # 打印检测结果
    for box in result.boxes:
        cls_id = int(box.cls)  # 获取类别ID
        conf = float(box.conf)  # 获取置信度
        print(f"类别ID: {cls_id}, 置信度: {conf}, 名称: {model.names[cls_id]}")