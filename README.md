🚦 Traffic Sign Detection using YOLOv8m

An end-to-end Traffic Sign Detection system built using YOLOv8m that covers the complete machine learning pipeline — from dataset preparation to training, evaluation, and inference — in a single executable script.
This project detects and localizes traffic signs in images using bounding boxes and class labels, making it suitable for Intelligent Transportation Systems (ITS) and road safety applications.

📌 Project Highlights

🔁 One-click CSV → YOLO format conversion
📊 Automatic train–validation split
🧠 YOLOv8m (Medium) pretrained model
🚀 GPU acceleration (if available)
📈 Model evaluation using YOLO metrics
🔍 Inference on unseen test images
📂 Organized YOLO directory structure

🧠 Tech Stack

Python 3.8+
Ultralytics YOLOv8
PyTorch
OpenCV
Pandas
Pillow (PIL)
Scikit-learn
TQDM
PyYAML

📂 Dataset Structure (Input)
TrafficSignDataset/
├── train.csv
├── test.csv
└── images/
    ├── img1.jpg
    ├── img2.jpg
    └── ...

CSV Annotation Format

Each CSV file should contain:
Column Name	Description
Path	Image file path
Roi.X1	Top-left X coordinate
Roi.Y1	Top-left Y coordinate
Roi.X2	Bottom-right X coordinate
Roi.Y2	Bottom-right Y coordinate
ClassId	Traffic sign class label

📁 YOLO Dataset Structure
TrafficSignYOLO/
├── images/
│   ├── train
│   ├── val
│   └── test
├── labels/
│   ├── train
│   ├── val
│   └── test
└── data.yaml

⚙️ Installation
1️⃣ Create & Activate Virtual Environment (Recommended)
conda create -n traffic-sign-yolo python=3.9 -y
conda activate traffic-sign-yolo

2️⃣ Install Dependencies
pip install ultralytics opencv-python pandas tqdm pillow pyyaml scikit-learn

▶️ How to Run the Project

Simply run the script:
python traffic_sign_detection_yolov8.py
That’s it 🚀
The script automatically performs:

Dataset validation
CSV → YOLO conversion
Train/validation split
Model training
Model evaluation
Inference on test images

🧪 Training Configuration
Parameter	Value
Model	YOLOv8m
Image Size	640
Batch Size	16
Epochs	3
Train Split	85%
Validation Split	15%
Device	GPU (if available) / CPU
📊 Model Evaluation

Uses YOLOv8 built-in validation

Reports:

Precision
Recall
mAP@0.5
mAP@0.5:0.95

🔍 Inference Results

Inference is run on test images
Output images with bounding boxes are saved automatically
runs/traffic_sign_yolov8m/
├── weights/
│   ├── best.pt
│   └── last.pt
├── val/
└── predict/

🚀 Use Cases

Intelligent Traffic Systems
Autonomous Vehicles
Smart City Surveillance
Road Safety Analysis
Government & Highway Authority Projects (e.g., NHAI)

📈 Future Improvements

Add class name mapping instead of numeric labels
Integrate real-time video detection
MLflow experiment tracking
