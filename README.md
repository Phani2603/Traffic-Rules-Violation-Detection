# Automated Traffic Rule Violation Detection using YOLOv8

## Overview

This project implements an automated traffic rule violation detection system using the YOLOv8 deep learning model. The system is designed to detect helmet-related traffic violations among motorcycle riders and identify corresponding vehicle number plates. It processes video footage to classify helmet usage and optionally capture evidence snapshots of violations.

The goal of this project is to assist in promoting road safety through automated and scalable AI-based surveillance.

---

## Features

### Helmet Violation Detection
- Detects motorcycle riders in traffic footage.
- Classifies helmet usage into three categories:
  - `faceWithGoodHelmet` – compliant
  - `faceWithBadHelmet` – violation
  - `faceWithNoHelmet` – violation

### Number Plate Detection
- Detects vehicle number plates associated with riders for identification.

### Violation Evidence Capture (Optional)
- Saves image snapshots of detected violations, including cropped number plates.

### Video Annotation
- Generates annotated output videos with bounding boxes and class labels indicating detected objects and violation types.

---

## Dataset

**Source:** [Indian Helmet Detection Dataset – Roboflow Universe](https://universe.roboflow.com/yolo-ftygl/aryan_1/dataset/1)

| Attribute | Details |
|------------|----------|
| Total Images | ~942 |
| Classes | numberPlate, faceWithNoHelmet, faceWithGoodHelmet, faceWithBadHelmet, rider |
| Annotation Format | YOLOv8 |
| Split | Train: 800 images, Validation: 142 images |
| License | CC BY 4.0 |

---

## Technology Stack

| Category | Technology |
|-----------|-------------|
| Programming Language | Python 3.x |
| Deep Learning Framework | PyTorch |
| Object Detection Model | YOLOv8 (yolov8n, yolov8s variants) |
| Computer Vision | OpenCV |
| Training Environment | Google Colab (GPU-enabled) |
| Development Environment | PyCharm, VS Code |

---

## Project Structure

```
traffic_violation_project/
├── .gitignore
├── README.md
├── requirements.txt
│
├── dataset/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   └── valid/
│       ├── images/
│       └── labels/
│
├── models/
│   └── best.pt
│
├── input_data/
│   └── traffic1.mp4
│
├── output/
│   ├── annotated_videos/
│   └── violation_snapshots/
│
└── scripts/
    ├── data.yaml
    ├── train.py
    └── detect_video.py
```

---

## Setup and Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd traffic_violation_project
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
# Activate (Windows)
.\venv\Scripts\activate
# or (macOS/Linux)
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*Note: PyTorch is installed automatically with the Ultralytics package.*

### 4. Download and Prepare the Dataset
Download the dataset and place the `train` and `valid` folders (with `images` and `labels` subfolders) inside the `dataset/` directory.

### 5. (Optional) Sync with Google Drive for Colab Training
If you plan to train the model on Google Colab, store the entire project directory inside your Google Drive for easy access.

---

## Usage

### 1. Training (Google Colab)

1. Open a new Colab notebook.
2. Enable GPU:
   - Runtime → Change runtime type → Select GPU (T4 recommended).
3. Mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. Install Ultralytics:
   ```bash
   !pip install ultralytics
   ```
5. Navigate to the scripts directory:
   ```bash
   %cd /content/drive/MyDrive/traffic_violation_project/scripts/
   ```
6. Run the training script:
   ```bash
   !python train.py
   ```
7. The best-trained model weights (`best.pt`) will be saved under:
   ```
   scripts/training_results/<run_name>/weights/
   ```
   Move this file to the `models/` directory.

---

### 2. Detection / Inference (Local Execution)

1. Ensure the trained model `best.pt` is located in the `models/` folder.
2. Place the input video (e.g., `traffic1.mp4`) inside the `input_data/` folder.
3. Navigate to the scripts directory:
   ```bash
   cd path/to/traffic_violation_project/scripts/
   ```
4. Run detection:
   ```bash
   python detect_video.py
   ```
5. The output files will be generated as follows:
   - Annotated video: `output/annotated_videos/`
   - Violation snapshots: `output/violation_snapshots/` (if enabled)

---

## Results

### Example Detection
Annotated frame showing detection of a "No Helmet" violation.

| Metric | YOLOv8s (50 Epochs) |
|---------|----------------------|
| mAP50 | 0.688 |
| mAP50-95 | 0.318 |

| Class | mAP50 | mAP50-95 |
|--------|-------|----------|
| numberPlate | 0.695 | 0.250 |
| faceWithNoHelmet | 0.719 | 0.288 |
| faceWithGoodHelmet | 0.765 | 0.356 |
| faceWithBadHelmet | 0.356 | 0.175 |
| rider | 0.907 | 0.520 |

*(Additional figures such as confusion matrices, precision-recall curves, and training logs can be included here when available.)*

---

## Future Enhancements

- Train larger models (e.g., YOLOv8m, YOLOv8l) for improved accuracy.
- Increase the number of training epochs for better generalization.
- Expand the dataset with additional images under varied lighting and environments.
- Integrate Optical Character Recognition (OCR) tools such as Tesseract or EasyOCR to extract text from number plates.
- Implement object tracking using SORT or DeepSORT for multi-frame rider tracking.
- Add database integration to store violation details, timestamps, and evidence.
- Adapt the detection pipeline for real-time video stream processing.

---

## Acknowledgements

- Dataset: Indian Helmet Detection Dataset from Roboflow Universe.
- Model: YOLOv8 architecture by Ultralytics.
- Compute Resources: Google Colab GPU Environment.

---

## License

This project is licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license. You are free to use, modify, and distribute this project with proper attribution.

---

## Authors

Developed by [Your Organization or Team Name]  
For research and development in intelligent traffic monitoring and road safety systems.
