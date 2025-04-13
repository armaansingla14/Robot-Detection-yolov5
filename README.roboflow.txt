# Robot-Detection-yolov5

A computer vision project that uses YOLOv5 to detect armor plates on robots for real-time tracking and classification. The dataset was collected from RoboMaster matches and augmented using Roboflow to enhance training robustness. This project includes dataset preprocessing, training scripts, and an inference pipeline.

---

## 📁 Project Structure

```
├── datasets/robot_detection/   # Roboflow-exported dataset in YOLOv5 format
├── runs/train/                 # YOLOv5 training output (metrics, weights, logs)
├── data_preparation.py         # Script to adjust annotations, structure folders
├── detection_model.py          # Inference logic using trained model
├── model_pipeline.py           # End-to-end prediction pipeline
├── yolov5s.pt                  # Final trained YOLOv5 model
├── README.dataset.txt          # Notes on dataset structure
├── README.roboflow.txt         # Roboflow export details
└── .cometml-runs/              # Experiment tracking artifacts
```

---

## 📊 Dataset Details

- **Exported via Roboflow** on May 29, 2022.
- **Total Images:** 1200  
- **Classes:** `armor`
- **Format:** YOLOv5 annotations
- **Augmentations:**  
  - 3 versions per image
  - 90-degree rotation: none, clockwise, counterclockwise

---

## 🧠 Model

- **Architecture:** YOLOv5s
- **Framework:** PyTorch
- **Training Platform:** Roboflow + Local Training

---

## 🚀 How to Run

### 1. Setup
```bash
pip install -r requirements.txt
```

### 2. Train the model (if retraining)
```bash
# Inside cloned YOLOv5 repo or compatible trainer
python train.py --img 640 --batch 16 --epochs 50 --data dataset.yaml --weights yolov5s.pt --name armor-detection
```

### 3. Run Inference
```bash
python detection_model.py
```

---

## 📌 Output

The trained YOLOv5 model returns bounding boxes for detected `armor` components on input images, suitable for real-time use in competitive robotics like RoboMaster.

---

## 👤 Author

Armaan Singla  
Computer Engineering @ Queen's University  
[GitHub Profile](https://github.com/armaansingla14)
