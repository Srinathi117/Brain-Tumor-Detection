# 🧠 Brain Tumor Detection using Optimization with Genetic and Heuristic Algorithms

Deep learning-based Brain MRI tumor classification system using **Transfer Learning with EfficientNetB3**.  
The model classifies MRI scans into four categories:

- Glioma  
- Meningioma  
- Pituitary  
- No Tumor  

Achieves **~98–99% accuracy** with strong sensitivity and specificity.

---

## 📌 Project Overview

Manual analysis of Brain MRI scans is:

- Time-consuming  
- Dependent on radiologist expertise  
- Prone to human error  

This project proposes an automated deep learning pipeline that:

- Enhances MRI images using preprocessing techniques  
- Uses EfficientNetB3 for feature extraction  
- Applies two-phase transfer learning  
- Provides real-time prediction via a Colab interface  

The system is lightweight, accurate, and suitable for clinical assistance.

---

## 🏗️ Project Workflow

```
MRI Image
   ↓
Preprocessing (Auto-crop + Histogram Equalization + Normalization)
   ↓
EfficientNetB3 (Feature Extraction)
   ↓
Fully Connected Layers
   ↓
Softmax Classification (4 Classes)
   ↓
Performance Evaluation + Visualization
```

---

## 📂 Dataset

- Publicly available Brain MRI dataset  
- 4 Classes:
  - Glioma
  - Meningioma
  - Pituitary
  - No Tumor  

### Dataset Split
- 80% Training  
- 20% Validation  

---

## 🧪 Preprocessing Pipeline

### 1️⃣ Auto-Cropping
- Detects largest brain contour  
- Removes unnecessary background  

### 2️⃣ Resizing
- Resized to **300 × 300 pixels**

### 3️⃣ Histogram Equalization
- Applied to luminance channel (YUV)  
- Enhances tumor contrast  

### 4️⃣ Normalization
- Pixel values scaled to `[0,1]`

### 5️⃣ TensorFlow Dataset Pipeline
- `image_dataset_from_directory`
- Caching & Prefetching for performance  

---

## 🧠 Model Architecture

### 🔹 Backbone: EfficientNetB3

- Pretrained on ImageNet  
- `include_top=False`  
- Transfer Learning enabled  

### 🔹 Custom Classification Head

- GlobalAveragePooling2D  
- Dense (256 neurons, ReLU)  
- Dropout (0.4)  
- Dense (4 neurons, Softmax)  

---

## 🔄 Training Strategy

### 🥇 Phase 1 – Frozen Base (Feature Extraction)

- Base model frozen  
- Only custom layers trained  
- Learning rate: `1e-4`  
- Validation accuracy reached ~97%  

### 🥈 Phase 2 – Fine-Tuning

- Top 30 layers unfrozen  
- Learning rate reduced to `1e-5`  
- Final validation accuracy ~98–99%  

**Optimizer:** Adam  
**Loss Function:** Sparse Categorical Crossentropy  

---

## 📊 Performance Metrics

| Metric | Score |
|--------|--------|
| Accuracy | 95.7% – 99% |
| Precision (Macro) | 0.954 |
| Recall (Macro) | 0.957 |
| F1-Score | 0.955 |
| Mean Sensitivity | 0.949 |
| Mean Specificity | 0.974 |

---

## 📈 Confusion Matrix

| Actual \ Predicted | Glioma | Meningioma | No Tumor | Pituitary |
|-------------------|--------|------------|----------|------------|
| Glioma | 78 | 2 | 1 | 0 |
| Meningioma | 2 | 72 | 3 | 2 |
| No Tumor | 0 | 1 | 65 | 1 |
| Pituitary | 3 | 0 | 2 | 79 |

✔ Strong diagonal dominance  
✔ Very few misclassifications  

---

## 🔍 Class-wise Sensitivity & Specificity

| Class | Sensitivity | Specificity |
|--------|-------------|-------------|
| Glioma | 0.963 | 0.972 |
| Meningioma | 0.945 | 0.978 |
| No Tumor | 0.955 | 0.987 |
| Pituitary | 0.967 | 0.959 |

---

## 📊 Visualizations Included

- 📈 Training & Validation Accuracy Curve  
- 📉 Loss Curve  
- 🔥 Confusion Matrix Heatmap  
- 📊 Sensitivity & Specificity Bar Chart  
- 📊 Precision–Recall–F1 Visualization  
- 🖼 Sample Prediction Output  

---

## 💻 Prediction Interface

A lightweight Google Colab runtime interface allows:

- Uploading MRI image  
- Automatic preprocessing  
- Real-time prediction  
- Confidence score display  

### Example Output

```
Prediction: Glioma
Confidence: 98.3%
```

---

## 🚀 Why EfficientNetB3?

EfficientNet uses compound scaling:

- Scales depth  
- Scales width  
- Scales resolution  

This ensures:

- High accuracy  
- Low computational cost  
- Faster training  
- Better generalization  

Perfect for clinical deployment.

---

## 🔬 Key Contributions

✔ Two-phase transfer learning strategy  
✔ Strong preprocessing pipeline  
✔ High sensitivity & specificity  
✔ Lightweight deployment  
✔ Near state-of-the-art accuracy  

---

## 🛠️ Technologies Used

- Python  
- TensorFlow / Keras  
- EfficientNetB3  
- OpenCV  
- NumPy  
- Matplotlib  
- Scikit-learn  
- Google Colab  

---

## 📌 Future Improvements

- Add Grad-CAM visualization  
- Deploy as web application (Streamlit/Flask)  
- Integrate multi-modal MRI  
- Expand dataset for better robustness  

---

## 📜 Conclusion

This project demonstrates that transfer learning using EfficientNetB3 can achieve highly accurate, reliable, and computationally efficient brain tumor classification from MRI scans.

The system shows strong potential for real-world clinical assistance and AI-powered diagnostic support.
