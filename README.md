# Advanced Deep Learning: Time Series & Computer Vision Comparative Analysis

This repository contains two comprehensive deep learning projects developed as part of the **M.Tech in AI/ML program at BITS Pilani**. Each project focuses on evaluating different architectural approaches to solve domain-specific problems: **Time Series Forecasting** and **Medical Image Classification**.

---

## Project 1: RNN vs. Transformer for Time Series Forecasting
**File:** `RNN_vs_Transformer.ipynb`

### **Overview**
This project compares the efficiency and accuracy of traditional **Recurrent Neural Networks (LSTMs)** against modern **Transformer-based architectures** for univariate time series forecasting.

* **Dataset:** Hourly Energy Consumption (AEP) from the PJM Interconnection (Kaggle).
* **Objective:** Predict the next hour's energy consumption based on a 24-hour historical window.
* **Primary Metric:** **RMSE (Root Mean Square Error)**, chosen to penalize large forecasting errors which are critical for grid stability.

### **Key Findings**
* **LSTM Performance:** Achieved a highly efficient 1-step prediction with **49,985 parameters** and a training time of **~1,146 seconds**.
* **Transformer Performance:** Utilized multi-head attention to weigh 24 hours simultaneously, but required significantly more parameters (**172,801**) and a longer training time (**3,090 seconds**) to align positional encodings.
* **Efficiency:** For this specific 24-step window, the **LSTM proved more computationally efficient and stable**, reaching a lower loss faster than the Transformer.

---

## Project 2: Custom CNN vs. Transfer Learning for Medical Diagnosis
**File:** `Custom_CNN_vs_Transfer_Learning.ipynb`

### **Overview**
This project evaluates a lightweight **custom Convolutional Neural Network** against a pre-trained **VGG16 model (Transfer Learning)** for detecting Pneumonia in chest X-ray images.

* **Dataset:** Chest X-Ray (Pneumonia) dataset by Paul Mooney (Kaggle).
* **Objective:** Binary classification of X-ray images into **'Normal'** or **'Pneumonia'**.
* **Primary Metric:** **Recall**, because missing a positive case (False Negative) is much more costly in medical diagnostics.

### **Key Findings**
* **Custom CNN:** A highly efficient implementation (**~19.5k parameters**) that focuses on global spatial features using **Global Average Pooling**. It achieved an accuracy of **82%** with a training time of **1,804 seconds**.
* **Transfer Learning (VGG16):** Leveraged robust feature extraction from ImageNet weights, providing **higher sensitivity (Recall)** compared to the custom model trained from scratch.
* **Takeaway:** While Custom CNNs are ideal for lightweight or niche applications, **Transfer Learning is superior for medical datasets** where data is limited and high Recall is mandatory.

---

## Technical Stack
* **Frameworks:** TensorFlow, Keras, Scikit-learn
* **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Statsmodels
* **Data Sourcing:** Kaggle API (opendatasets)

---
### **How to Use**
1. Clone the repository.
2. Open the `.ipynb` files in **Google Colab** or Jupyter Notebooks.
3. Ensure you have your `kaggle.json` credentials for dataset downloading via the `opendatasets` library.
