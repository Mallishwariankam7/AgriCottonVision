# 🌱 CottonVision-AI  
### Cotton Crop Growth Stage & Health Classification using Computer Vision

## 📌 Project Overview
CottonVision-AI is a Computer Vision–based system that analyzes images of cotton crops to determine:

- 🌿 Growth Stage of the cotton boll  
- ❤️ Health Status (Healthy / Diseased)  
- 📊 Health Score (0–100%)  
- 🔥 Grad-CAM heatmap for model explainability  

This system helps farmers decide the **optimal harvest time** and **detect crop health issues early**.

---

## 🎯 Problem Statement
Manual inspection of cotton crops is:
- Time-consuming  
- Error-prone  
- Not scalable  

This project automates crop maturity and health assessment using **Deep Learning and Transfer Learning**.

---

## 🧠 Key Features

### 1️⃣ Growth Stage Classification
Classifies cotton crops into **four phases**:
- Phase 1: Vegetative / Budding  
- Phase 2: Flowering  
- Phase 3: Bursting (Ripped)  
- Phase 4: Harvest Ready  

### 2️⃣ Health Assessment
- Detects **Healthy vs Diseased** cotton bolls  
- Outputs a **Health Score (0–100%)** based on prediction confidence  

### 3️⃣ Data Augmentation
To handle real-world agricultural conditions:
- Rotation  
- Zoom  
- Brightness variation  
- Horizontal flip  
- Shift transformations  

### 4️⃣ Explainable AI (Grad-CAM)
- Visual heatmaps showing **where the model focuses**
- Improves trust and interpretability

### 5️⃣ End-to-End Inference Pipeline
Single image input → JSON output:
```json
{
  "image": "sample.jpg",
  "growth_stage": "Phase 4: Harvest Ready",
  "is_ripped": true,
  "health_status": "healthy",
  "health_score": 97.72
}
