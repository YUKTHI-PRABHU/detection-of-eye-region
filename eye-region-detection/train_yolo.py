from ultralytics import YOLO

# load pretrained model
model = YOLO("yolov8n.pt")

# train
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="eye_detector"
)

# save best model
model.export(format="onnx")  # optional
