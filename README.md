# Aeroponics-Vegetable-Monitoring---Method-3-Image-Preprocessing

Image preprocessing pipeline for Task 2 (Variety Identification) using perspective transformation to convert Cam5 top-down views into angled views matching Cam6 perspective.

---

## 📋 Project Overview

This preprocessing system prepares image data for training a vegetable variety identification model in an Aeroponics monitoring system. The project focuses on **Method 3** approach, which transforms Cam5 images to match Cam6 viewing angles, creating additional training data with diverse perspectives.

### Main Objectives

1. Transform Cam5 (top-down view) images to angled views matching Cam6
2. Label transformed images with vegetable varieties
3. Combine with existing Cam1-5 dataset
4. Train improved variety identification model

---

## 🎯 Task 2: Variety Identification - Method 3

### What is Method 3?

**Method 3** uses perspective transformation to convert Cam5's top-down view into side-angled views that match Cam6's perspective. This approach:
- **Increases training data** by creating additional images from Cam5
- **Adds viewing diversity** by combining multiple camera angles
- **Improves model generalization** with varied perspectives
- **Maintains compatibility** with existing Cam1-5 labeled dataset

### Variety Classes

| Index | Variety Name | Visual Characteristics |
|-------|-------------|----------------------|
| 0 | Italian | Long narrow leaves, dark green |
| 1 | Deer Tongue | Round leaves with pointed tips |
| 2 | Green Lollo Rossa | Very curly leaves, green |
| 3 | Red Coral | Red-purple color, curly leaves |
| 4 | Caramel Romaine | Long leaves, green-yellow |
| 5 | empty / no sponge | No plant or empty sponge |

### Dataset Information

- **Date Range**: May 17 - June 12, 2025 (27 days)
- **Capture Frequency**: Every hour (24 images/day)
- **Total Images**: ~648 images per camera
- **File Format**: `YYYYMMDD_hhmmss.jpg` (e.g., `20250517_060001.jpg`)
- **Main Camera**: Cam5 (used as left side view)
- **Secondary Camera**: Cam1/2/3/4/5 (used as right side view)

---

## 🗂️ Project Structure

```
aeroponics_preprocessing/
├── data/
│   ├── cam5_24H/              # Original Cam5 images
│   │   ├── 20250517/
│   │   ├── 20250518/
│   │   └── ... (27 folders)
│   └── cam1-4_24H/            # Secondary camera images
├── result/
│   ├── cam5_bent_dual_24H/    # Transformed images (Method 3)
│   │   └── [date]/
│   │       ├── left_bend/     # Left-angled view
│   │       └── right_bend/    # Right-angled view
│   ├── cam5_panorama_24H/     # Optional panorama output
│   │   └── [date]/
│   └── cam5_transformed/      # Alternative 3-section output
│       └── [date]/
│           ├── left/
│           ├── middle/
│           └── right/
├── scripts/
│   ├── cam5_transform.py      # Main transformation script ⭐
│   ├── main_cam5.py           # Alternative 3-section transform
│   ├── image_panorama.py      # Create panorama
│   ├── flatten_cam5.py        # Organize images for labeling
│   └── combi_image.py         # Combine images from folders
└── README.md
```

---

## 🔧 Installation & Requirements

### Prerequisites

```bash
Python 3.8+
opencv-python>=4.8.0
numpy>=1.24.0
pillow>=10.0.0
```

### Installation

```bash
# Install required packages
pip install opencv-python numpy pillow

# Clone or download this repository
# Organize your data according to the project structure above
```

---

## 🚀 Method 3 Workflow

### Complete Pipeline Overview

```
1. Original Cam5 Images (Top-Down View)
   ↓
2. Perspective Transformation (cam5_transform.py)
   ↓
3. Generate Left & Right Angled Views
   ↓
4. Organize Images (flatten_cam5.py)
   ↓
5. Upload to Labeling Platform (CVAT/Roboflow)
   ↓
6. Label with Variety Classes
   ↓
7. Export as YOLO Format
   ↓
8. Combine with Existing Dataset
   ↓
9. Split Train/Val/Test (70/20/10)
   ↓
10. Ready for Model Training
```

---

## 📝 Detailed Step-by-Step Guide

### Step 1: Prepare Source Data

Organize your Cam5 images in the following structure:

```
data/cam5_24H/
├── 20250517/
│   ├── 20250517_000001.jpg
│   ├── 20250517_010001.jpg
│   ├── ...
│   └── 20250517_230001.jpg
├── 20250518/
└── ... (continue for all 27 days)
```

**Verification Checklist**:
- ✓ All 27 days present (05-17 to 06-12)
- ✓ 24 images per day (hourly captures)
- ✓ Filenames follow `YYYYMMDD_hhmmss` format
- ✓ Images are clear and not corrupted

---

### Step 2: Transform Images with Perspective Transformation

**Run the main transformation script**:

```bash
python cam5_transform.py
```

#### Interactive Transformation Process

The script will guide you through an interactive point selection process:

1. **View Sample Image**: First image from each date folder is displayed

2. **Click 4 Corner Points** in this exact order:
   - Point 1: **Top-Left** corner
   - Point 2: **Top-Right** corner
   - Point 3: **Bottom-Right** corner
   - Point 4: **Bottom-Left** corner

3. **Preview Results**: Press `[p]` to see transformation preview
   - Left bend view
   - Right bend view

4. **Confirm or Adjust**:
   - Press `[y]` to confirm and process all images in folder
   - Press `[c]` to clear points and start over
   - Press `[q]` to skip this folder

5. **Batch Processing**: Script processes all images in the folder using the same transformation points

#### Keyboard Controls

| Key | Action |
|-----|--------|
| `[p]` | Preview transformation result |
| `[y]` | Confirm and process all images |
| `[c]` | Clear points and restart |
| `[q]` | Skip current folder |

#### Output Structure

```
result/cam5_bent_dual_24H/
├── 20250517/
│   ├── left_bend/
│   │   ├── 20250517_000001_left_bend.jpg
│   │   ├── 20250517_010001_left_bend.jpg
│   │   └── ... (24 images)
│   └── right_bend/
│       ├── 20250517_000001_right_bend.jpg
│       ├── 20250517_010001_right_bend.jpg
│       └── ... (24 images)
├── 20250518/
└── ... (all 27 days)
```

**Result**: ~1,296 transformed images (27 days × 24 images/day × 2 views)

---

### Step 3: Organize Images for Labeling

**Flatten folder structure for easy upload**:

```bash
python flatten_cam5.py "result/cam5_bent_dual_24H" "ready_for_labeling/method3"
```

This script:
- Collects all images from nested date folders
- Renames files to include parent folder name
- Places all images in a single flat folder
- Handles filename conflicts automatically

**Output**:
```
ready_for_labeling/method3/
├── 20250517_000001_left_bend.jpg
├── 20250517_000001_right_bend.jpg
├── 20250517_010001_left_bend.jpg
├── 20250517_010001_right_bend.jpg
└── ... (all ~1,296 images)
```

---

### Step 4: Upload to Labeling Platform

#### Recommended Platform: CVAT

**Why CVAT?**
- Handles large image volumes efficiently
- Advanced keyboard shortcuts for fast labeling
- Team collaboration support
- AI-assisted auto-annotation
- Clean YOLO export

#### Setup CVAT

**Option A: Docker (Recommended)**
```bash
docker run -d -p 8080:8080 --name cvat cvat/cvat
# Access at http://localhost:8080
```

**Option B: Cloud**
- Visit [app.cvat.ai](https://app.cvat.ai)
- Free tier available

#### Create Labeling Project

1. **Create New Project**: `Task2_Method3_Variety_Identification`

2. **Add Labels** (6 classes):
   - Italian (index: 0)
   - Deer Tongue (index: 1)
   - Green Lollo Rossa (index: 2)
   - Red Coral (index: 3)
   - Caramel Romaine (index: 4)
   - empty (index: 5)

3. **Create Task**: Upload images from `ready_for_labeling/method3/`

4. **Configure Settings**:
   - Annotation type: Bounding Box
   - Overlap size: 0
   - Image quality: 95

#### Alternative Platform: Roboflow

```bash
# Visit roboflow.com
# Create project → Upload images → Start labeling
# Export as YOLO format when done
```

---

### Step 5: Label Images

#### Labeling Guidelines

**Bounding Box Rules**:
- Draw tight boxes around each vegetable plant
- Include all visible leaves of the plant
- Don't include neighboring plants in the same box
- Box edges should align with outermost leaves

**Class Selection**:
- Identify variety based on visual characteristics (see table above)
- When uncertain, refer to Index Reference sheet
- Mark empty sponges as class "empty"
- Label every plant in the image

#### CVAT Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `N` | Create bounding box |
| `1-6` | Select class (Italian=1, Deer Tongue=2, etc.) |
| `F` | Next image |
| `D` | Previous image |
| `Ctrl+S` | Save |
| `Ctrl+Z` | Undo |

#### Quality Control

- **Review frequency**: Every 50-100 images
- **Consistency check**: Ensure similar plants are labeled the same
- **Coverage check**: Verify all plants are labeled
- **Accuracy check**: Verify correct variety classes

**Estimated Time**: ~2-3 hours per 100 images (experienced labeler)

---

### Step 6: Export Labeled Data

#### From CVAT

1. Go to Project → Export Dataset
2. Select Format: **YOLO 1.1**
3. Download ZIP file
4. Extract to working directory

**Output Structure**:
```
cvat_export/
├── obj_train_data/
│   ├── 20250517_000001_left_bend.jpg
│   ├── 20250517_000001_left_bend.txt
│   ├── 20250517_000001_right_bend.jpg
│   ├── 20250517_000001_right_bend.txt
│   └── ...
└── obj.names
```

#### From Roboflow

1. Generate → Download
2. Select Format: **YOLO v8**
3. Download and extract

---

### Step 7: Combine with Existing Dataset

#### Download Existing Dataset

```bash
# Download from Google Drive
# Link: https://drive.google.com/drive/folders/1RBzdOaFsPkJNI3835MWPOyPq5oQHoMnC
# Extract to: existing_dataset/
```

#### Merge Datasets

Combine Method 3 labels with existing Cam1-5 dataset:

```
combined_dataset/
├── images/
│   ├── cam1_img_001.jpg         # From existing dataset
│   ├── cam2_img_002.jpg         # From existing dataset
│   ├── 20250517_000001_left.jpg # From Method 3
│   ├── 20250517_000001_right.jpg # From Method 3
│   └── ...
└── labels/
    ├── cam1_img_001.txt
    ├── cam2_img_002.txt
    ├── 20250517_000001_left.txt
    ├── 20250517_000001_right.txt
    └── ...
```

---

### Step 8: Split Train/Val/Test Sets

**Recommended Split**:
- **Train**: 70% (for model training)
- **Validation**: 20% (for hyperparameter tuning)
- **Test**: 10% (for final evaluation)

**Example Python Script**:

```python
import os
import shutil
import random
from pathlib import Path

def split_dataset(images_dir, labels_dir, output_dir, train_ratio=0.7, val_ratio=0.2):
    """Split dataset into train/val/test sets"""
    
    # Get all image files
    images = list(Path(images_dir).glob('*.jpg'))
    random.shuffle(images)
    
    # Calculate split sizes
    total = len(images)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    # Split images
    train_imgs = images[:train_size]
    val_imgs = images[train_size:train_size + val_size]
    test_imgs = images[train_size + val_size:]
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (Path(output_dir) / 'images' / split).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Copy files
    for split_name, img_list in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
        for img_path in img_list:
            # Copy image
            img_dest = Path(output_dir) / 'images' / split_name / img_path.name
            shutil.copy2(img_path, img_dest)
            
            # Copy corresponding label
            label_path = Path(labels_dir) / img_path.with_suffix('.txt').name
            if label_path.exists():
                label_dest = Path(output_dir) / 'labels' / split_name / label_path.name
                shutil.copy2(label_path, label_dest)
    
    print(f"Dataset split complete:")
    print(f"  Train: {len(train_imgs)} images")
    print(f"  Val: {len(val_imgs)} images")
    print(f"  Test: {len(test_imgs)} images")

# Usage
split_dataset('combined_dataset/images', 
              'combined_dataset/labels', 
              'final_dataset')
```

#### Create data.yaml

```yaml
# final_dataset/data.yaml

path: ./final_dataset
train: images/train
val: images/val
test: images/test

nc: 6
names: ['Italian', 'Deer Tongue', 'Green Lollo Rossa', 'Red Coral', 'Caramel Romaine', 'empty']
```

---

### Step 9: Upload to Google Drive

#### Folder Naming Convention

**Format**: `Task2_Method3_Cam5Angles_[StartDate]-[EndDate]_v[Version]`

**Example**: `Task2_Method3_Cam5Angles_20250517-20250612_v1`

#### Recommended Google Drive Structure

```
Task2_Method3_Cam5Angles_20250517-20250612_v1/
├── 01_Preprocessed_Images/
│   ├── left_bend/              # All left-angled views
│   ├── right_bend/             # All right-angled views
│   └── statistics.txt          # Image count, sizes
│
├── 02_Labeled_Data/
│   ├── cvat_export/            # Raw export from CVAT
│   └── yolo_format/            # Converted YOLO format
│       ├── images/
│       ├── labels/
│       └── data.yaml
│
├── 03_Combined_Dataset/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── labels/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── data.yaml
│   └── statistics.txt          # Class distribution, split sizes
│
├── 04_Documentation/
│   ├── labeling_report.md      # Labeling process and statistics
│   ├── preprocessing_log.txt   # Transformation parameters
│   └── sample_images/          # Example labeled images
│
└── README.md                    # Project summary
```

#### Create Project README

Include in your Google Drive folder:

```markdown
# Task 2 Method 3: Cam5 to Cam6 Angles Transformation

## Project Information
- **Task**: Variety Identification - Method 3
- **Date Range**: May 17 - June 12, 2025
- **Main Camera**: Cam5 (transformed to angled views)
- **Total Images Processed**: [XXX] images
- **Labeling Platform**: CVAT
- **Labeled By**: [Your Name]
- **Completion Date**: [Date]

## Dataset Statistics

### Transformed Images
- Left bend views: [XXX] images
- Right bend views: [XXX] images
- Total: [XXX] images

### Combined Dataset
- Existing Cam1-5: [XXX] images
- Method 3: [XXX] images
- Combined Total: [XXX] images

### Data Split
- Train: [XXX] images (70%)
- Validation: [XXX] images (20%)
- Test: [XXX] images (10%)

### Class Distribution
| Class | Count | Percentage |
|-------|-------|-----------|
| Italian | XXX | XX% |
| Deer Tongue | XXX | XX% |
| Green Lollo Rossa | XXX | XX% |
| Red Coral | XXX | XX% |
| Caramel Romaine | XXX | XX% |
| empty | XXX | XX% |

## Transformation Method
- Script: `cam5_transform.py`
- Technique: Perspective transformation with dual-angle output
- Parameters: 4-point corner selection, 25% bend factor

## Notes
[Add any observations or issues encountered]
```

---

## 🎓 Best Practices & Tips

### Image Transformation

1. **Point Selection Accuracy**
   - Click precisely at corners, not approximate locations
   - Use zoom if needed for pixel-perfect placement
   - Maintain consistent point selection across date folders

2. **Preview Before Processing**
   - Always preview with `[p]` before confirming
   - Check both left and right bend outputs
   - Verify no excessive distortion

3. **Batch Consistency**
   - Use same transformation points for images taken on same day
   - Document point coordinates for reproducibility
   - Keep backup of original images

### Labeling Quality

1. **Consistency is Key**
   - Define clear criteria for each variety before starting
   - Use reference images for uncertain cases
   - Take breaks to maintain concentration

2. **Accuracy Over Speed**
   - Quality labels are more valuable than quantity
   - Review difficult cases multiple times
   - Ask team members when uncertain

3. **Regular Reviews**
   - Review every 50-100 images
   - Check for labeling drift
   - Maintain consistent box sizes

### File Organization

1. **Clear Naming**
   - Use descriptive, structured folder names
   - Include dates and version numbers
   - Document any naming conventions

2. **Backup Everything**
   - Keep original images separate
   - Backup to multiple locations
   - Version control for dataset iterations

---

## 🐛 Troubleshooting

### Problem: UnicodeEncodeError on Windows

**Solution**: Run with UTF-8 encoding
```bash
chcp 65001
python cam5_transform.py
```

### Problem: Distorted transformation results

**Symptoms**: Plants appear stretched or compressed

**Solutions**:
- Reselect transformation points more carefully
- Ensure points follow clockwise order starting from top-left
- Verify sample image quality before transformation

### Problem: Inconsistent point selection

**Symptoms**: Results vary across date folders

**Solutions**:
- Document exact pixel coordinates of points
- Use same reference features (e.g., growing area corners)
- Create template overlay for consistent placement

### Problem: Out of memory during processing

**Symptoms**: Script crashes with memory error

**Solutions**:
- Process fewer date folders at once
- Reduce JPEG quality parameter (currently 95)
- Close other applications

---

## 📊 Expected Results

### Dataset Size Estimation

**Method 3 Contribution**:
- Original Cam5 images: ~648 (27 days × 24 hours)
- Transformed images: ~1,296 (648 × 2 views)
- Combined with existing: ~[existing_count] + 1,296 images

**Labeling Time Estimate**:
- Average: 2-3 hours per 100 images
- Total Method 3: ~26-39 hours
- With team: Divide by number of labelers

### Model Performance Expectations

Method 3 should provide:
- **Better generalization** from diverse viewing angles
- **Improved accuracy** with increased training data
- **Robust detection** across different camera positions
- **Enhanced performance** on Cam6 images

---

## 📚 Additional Resources

### Reference Links

- **CVAT Documentation**: [docs.cvat.ai](https://docs.cvat.ai)
- **YOLO Documentation**: [docs.ultralytics.com](https://docs.ultralytics.com)

### Related Tasks

This project is part of a larger Aeroponics monitoring system:

- **Task 1**: Dead Leaf Detection
- **Task 2**: Variety Identification (this project - Method 3)
- **Task 3**: Coverage Percentage Calculation
