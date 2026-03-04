
# Aerial GCP Detection

## Overview

This project addresses the task of automatically detecting **Ground Control Point (GCP) markers** in aerial drone imagery.  
The goal is to predict:

1. The **pixel coordinates (x, y)** of the center of the GCP marker.
2. The **shape of the marker** (Cross or Square).

The solution uses a **deep learning approach with a convolutional neural network** trained to jointly perform **coordinate regression and shape classification**.


# Dataset

The dataset consists of aerial images organized in the following nested structure:

```

project_name / survey_name / gcp_id / image.JPG

````

Each training image has an associated annotation stored in:

```

curated\_gcp\_marks.json

````

Example annotation format:

```json
{
    "project1/survey1/2/DJI_0431.JPG": {
        "mark": {
            "x": 1024.5,
            "y": 850.2
        },
        "verified_shape": "Square"
    }
}
```

---

# Exploratory Data Analysis (EDA)

During EDA several important dataset characteristics were discovered:

* **Image resolution:** 4096 × 2730 pixels
* **Markers per image:** 1
* **Marker size:** approximately 20–40 pixels
* **Marker locations:** distributed across the entire image
* **Class distribution:**

| Shape  | Count |
| ------ | ----- |
| Square | 892   |
| Cross  | 105   |

The dataset is therefore **highly imbalanced (~8.5:1)**.

Additionally, the assignment description mentioned a third class **L-Shaped**, but no such samples were present in the provided training dataset.

---

# Approach

Because the GCP marker occupies a **very small portion of the full-resolution image**, training directly on the entire image would be inefficient.

Instead, the training pipeline performs:

1. **Cropping around the annotated marker location**
2. **Resizing the crop for CNN input**
3. **Predicting the marker center relative to the crop**

The model uses a **multi-task learning architecture**:

```
Input Image Crop
        ↓
ResNet18 Backbone
        ↓
Shared Feature Representation
       / \
      /   \
Coordinate Head   Classification Head
(x, y) regression  marker shape
```

The coordinate head predicts the **normalized position of the marker within the crop**, while the classification head predicts the **marker shape**.

---

# Training Strategy

### Model

* Backbone: **ResNet18 (ImageNet pretrained)**
* Two prediction heads:

  * Coordinate regression
  * Shape classification

---

### Data Preprocessing

* Cropping around ground-truth marker location
* Resize to **256 × 256**
* Convert to tensor

---

### Data Augmentation

To improve generalization the following augmentations were used:

* Random horizontal flip
* Random vertical flip
* Color jitter

---

### Loss Function

The model is trained using a **combined loss**:

```
Total Loss = MSE Loss (coordinates) + CrossEntropy Loss (shape)
```

Where:

* **MSE Loss** supervises the predicted marker coordinates.
* **CrossEntropy Loss** supervises the shape classification.

---

### Training Configuration

| Parameter     | Value     |
| ------------- | --------- |
| Optimizer     | Adam      |
| Learning Rate | 1e-4      |
| Batch Size    | 16        |
| Epochs        | 10        |
| Input Size    | 256 × 256 |

Training was performed using **PyTorch with GPU acceleration (CUDA)**.

---

# Challenges and Solutions

### 1. Extremely Small Marker Size

**Problem**

The GCP markers occupy only a tiny region of the full-resolution image.

**Solution**

Images were cropped around the marker location during training, allowing the model to focus on the relevant region.

---

### 2. Class Imbalance

**Problem**

Square markers significantly outnumber Cross markers.

**Solution**

A stratified train-validation split and data augmentation were used to improve generalization.

---

### 3. High Resolution Images

**Problem**

Full-resolution images (4096×2730) are too large for direct training.

**Solution**

The pipeline uses **localized crops and resizing**, which greatly reduces memory requirements and training time.

---

# Model Weights

The trained model weights can be downloaded here:

**Google Drive Link**

[https://drive.google.com/file/d/1EZJt69z64lH4JdRprvds7Jf2p4qNV1op/view?usp=drive_link](https://drive.google.com/file/d/1EZJt69z64lH4JdRprvds7Jf2p4qNV1op/view?usp=drive_link)

After downloading, place the weights file in:

```
weights/gcp_model.pth
```

---

# Generating predictions.json

To reproduce the predictions for the test dataset:

```
python inference.py --weights weights/gcp_model.pth --test_dir test_dataset
```

This script will generate:

```
predictions.json
```

The output file follows the same format as the training annotations:

```json
{
  "project/survey/gcp/image.JPG": {
    "mark": {
      "x": 1234.5,
      "y": 567.8
    },
    "verified_shape": "Square"
  }
}
```

---

# Repository Structure

```
gcp-detection
│
├── notebooks
│   └── eda.ipynb
│
├── src
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── inference.py
│
├── weights
│   └── gcp_model.pth
│
├── predictions.json
├── requirements.txt
└── README.md
```

---

# Requirements

The project depends on the following libraries:

```
torch
torchvision
opencv-python
numpy
pandas
matplotlib
scikit-learn
tqdm
```

Install dependencies using:

```
pip install -r requirements.txt
```

---

# Conclusion

This project demonstrates a practical deep learning approach for **automated GCP marker localization and classification in aerial imagery**. By combining **image cropping, multi-task learning, and convolutional feature extraction**, the system is able to accurately predict marker positions and shapes from high-resolution drone images.

