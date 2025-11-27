from ultralytics import YOLO

model = YOLO("runs/detect/eye_detector/weights/best.pt")
metrics = model.val()
print(metrics)
