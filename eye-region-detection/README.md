# 👁️ Eye Region Detection & Feature Analysis using YOLO

A deep learning-based system to detect eye regions from face images and extract visual eye features such as brightness, openness, symmetry, and shape using YOLOv8.

---

## 🚀 Features
- Eye region detection using custom-trained YOLOv8
- Extracts:
  - Eye Openness
  - Symmetry Ratio
  - Brightness Level
  - Shape Properties
- Bounding box visualization for detected eyes
- Performance evaluation using mAP/IoU

---

## 🧠 Model Details
| Component | Description |
|----------|-------------|
| Model | YOLOv8n |
| Task | Eye Detection |
| Framework | Ultralytics (PyTorch) |
| Dataset | Custom annotated dataset |

---

## 📁 Folder Structure

Eye-Region-Detection/
│
├─ data.yaml
├─ train_yolo.py
├─ eye_analysis.py
├─ evaluate.py
│
├─ dataset/
│ ├─ images/
│ │ ├─ train/
│ │ ├─ val/
│ ├─ labels/
│ ├─ train/
│ ├─ val/
│
├─ runs/ → trained model weights saved here
└─ results/ → saved prediction outputs


---

## ⚙️ Installation

```bash
python -m venv venv


Activate environment:

Windows:

venv\Scripts\activate


Install dependencies:

pip install ultralytics opencv-python numpy matplotlib

🎯 Training the YOLO Model
python train_yolo.py


After training, best model will be saved at:

runs/detect/eye_detector*/weights/best.pt

🔍 Eye Feature Analysis

Add a test image in the project folder:

test.jpg  (or test.jpeg)


Then run:

python eye_analysis.py


📌 Results saved inside:

results/output.jpg