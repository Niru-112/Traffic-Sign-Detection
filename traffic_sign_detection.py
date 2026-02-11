"""
============================================================
🚦 TRAFFIC SIGN DETECTION USING YOLOv8m (ALL-IN-ONE SCRIPT)
============================================================

PROJECT DESCRIPTION:
This project implements an end-to-end Traffic Sign Detection system
using the YOLOv8m deep learning model. The pipeline includes:

1. Dataset preparation from CSV annotations
2. Conversion to YOLO format
3. Training a YOLOv8m object detection model
4. Model evaluation
5. Inference on test images

The model detects and localizes traffic signs in images using
bounding boxes and class labels.

Frameworks Used:
- Ultralytics YOLOv8
- PyTorch
- OpenCV
- Pandas
- PIL

============================================================
"""

# ===================== INSTALLATION =====================
# Run this once if required (not needed inside script execution)
# pip install ultralytics opencv-python pandas tqdm pillow pyyaml

# ===================== IMPORTS =====================
import os              # For file & directory handling
import shutil          # For copying image files
import yaml            # For writing YOLO configuration file
import torch           # PyTorch backend (CPU/GPU support)
import pandas as pd    # CSV file processing
from tqdm import tqdm  # Progress bar
from PIL import Image  # Image size extraction
from ultralytics import YOLO  # YOLOv8 framework
from sklearn.model_selection import train_test_split  # Dataset split

# ===================== CONFIGURATION =====================
# Root directory of original dataset
DATASET_ROOT = "TrafficSignDataset"

# Root directory for YOLO-formatted dataset
YOLO_ROOT = "TrafficSignYOLO"

# CSV annotation files
TRAIN_CSV = os.path.join(DATASET_ROOT, "train.csv")
TEST_CSV  = os.path.join(DATASET_ROOT, "test.csv")

# Training hyperparameters
TRAIN_RATIO = 0.85      # 85% training, 15% validation
EPOCHS = 3              # Number of training epochs
IMG_SIZE = 640          # Input image resolution
BATCH = 16              # Batch size
MODEL_NAME = "yolov8m.pt"  # Pretrained YOLOv8 Medium model
# ========================================================

# ===================== DEVICE SELECTION =====================
# Automatically selects GPU if available, else CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n🧠 Device selected: {DEVICE}")

if DEVICE == "cuda":
    print(f"🚀 GPU in use: {torch.cuda.get_device_name(0)}")

# ===================== DIRECTORY CREATION =====================
def make_dirs():
    """
    Creates YOLO directory structure:
    YOLO_ROOT/
        ├── images/
        │   ├── train
        │   ├── val
        │   └── test
        └── labels/
            ├── train
            ├── val
            └── test
    """
    for split in ["train", "val", "test"]:
        os.makedirs(f"{YOLO_ROOT}/images/{split}", exist_ok=True)
        os.makedirs(f"{YOLO_ROOT}/labels/{split}", exist_ok=True)

# ===================== DATASET CHECK =====================
def dataset_ready():
    """
    Checks whether YOLO dataset is already prepared.
    Prevents re-processing every time the script runs.
    """
    required_paths = [
        f"{YOLO_ROOT}/images/train",
        f"{YOLO_ROOT}/images/val",
        f"{YOLO_ROOT}/images/test",
        f"{YOLO_ROOT}/labels/train",
        f"{YOLO_ROOT}/labels/val",
        f"{YOLO_ROOT}/labels/test",
        f"{YOLO_ROOT}/data.yaml"
    ]

    # Verify existence of all required folders/files
    for path in required_paths:
        if not os.path.exists(path):
            return False

    # Ensure images exist (dataset not empty)
    return len(os.listdir(f"{YOLO_ROOT}/images/train")) > 0

# ===================== CSV → YOLO FORMAT =====================
def convert_csv_to_yolo(df, split):
    """
    Converts bounding box annotations from CSV format
    into YOLO format (.txt files).

    YOLO Format:
    class_id x_center y_center width height
    (All values normalized between 0 and 1)
    """
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Converting {split}"):

        # Original image path
        img_path = os.path.join(DATASET_ROOT, row["Path"])
        if not os.path.exists(img_path):
            continue

        # Load image to get width & height
        img = Image.open(img_path)
        w, h = img.size

        # Convert bounding box to YOLO format
        x_center = ((row["Roi.X1"] + row["Roi.X2"]) / 2) / w
        y_center = ((row["Roi.Y1"] + row["Roi.Y2"]) / 2) / h
        bw = (row["Roi.X2"] - row["Roi.X1"]) / w
        bh = (row["Roi.Y2"] - row["Roi.Y1"]) / h

        class_id = int(row["ClassId"])
        img_name = os.path.basename(row["Path"])

        # Copy image into YOLO directory
        shutil.copy(img_path, f"{YOLO_ROOT}/images/{split}/{img_name}")

        # Create label file
        label_path = f"{YOLO_ROOT}/labels/{split}/{os.path.splitext(img_name)[0]}.txt"
        with open(label_path, "w") as f:
            f.write(f"{class_id} {x_center} {y_center} {bw} {bh}")

# ===================== DATA PREPARATION =====================
if not dataset_ready():
    print("📦 Preparing YOLO dataset (one-time process)...")

    make_dirs()

    # Load CSV annotation files
    df_train = pd.read_csv(TRAIN_CSV)
    df_test  = pd.read_csv(TEST_CSV)

    # Stratified split to preserve class distribution
    train_df, val_df = train_test_split(
        df_train,
        test_size=1 - TRAIN_RATIO,
        stratify=df_train["ClassId"],
        random_state=42
    )

    # Convert CSV annotations to YOLO format
    convert_csv_to_yolo(train_df, "train")
    convert_csv_to_yolo(val_df, "val")
    convert_csv_to_yolo(df_test, "test")

    # ===================== YOLO data.yaml =====================
    num_classes = df_train["ClassId"].nunique()

    data_yaml = {
        "path": os.path.abspath(YOLO_ROOT),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": num_classes,
        "names": [str(i) for i in range(num_classes)]
    }

    with open(f"{YOLO_ROOT}/data.yaml", "w") as f:
        yaml.dump(data_yaml, f)

    print("✅ data.yaml created successfully")

else:
    print("✅ YOLO dataset already exists — skipping preparation")

# ===================== MODEL TRAINING =====================
model = YOLO(MODEL_NAME)

model.train(
    data=f"{YOLO_ROOT}/data.yaml",
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH,
    device=0 if DEVICE == "cuda" else "cpu",
    workers=0,
    project="runs",
    name="traffic_sign_yolov8m"
)

# ===================== MODEL EVALUATION =====================
print("\n📊 Evaluating trained model...")
model.val(data=f"{YOLO_ROOT}/data.yaml")

# ===================== INFERENCE =====================
print("\n🔍 Running inference on test images...")

# Load best trained weights
model = YOLO("runs/traffic_sign_yolov8m/weights/best.pt")

model.predict(
    source=f"{YOLO_ROOT}/images/test",
    conf=0.25,
    save=True
)

print("\n✅ ALL DONE SUCCESSFULLY!")
print("📁 Results saved inside: runs/traffic_sign_yolov8m/")
