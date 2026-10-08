##Brain Tumor Detection Using Deep Learning (CNN)

#Overview

This project uses a Convolutional Neural Network (CNN) to detect and classify brain tumors from MRI images. The model is trained on brain MRI scans and can classify images into four categories: Glioma, Meningioma, Pituitary Tumor, and No Tumor.

The system automatically learns features from MRI images and assists in identifying tumor types with high accuracy.

##Objectives

Detect brain tumors from MRI images.
Classify tumors into different categories.
Automate medical image analysis using Deep Learning.
Support faster and more accurate diagnosis.

##Technologies Used

Python
TensorFlow
Keras
OpenCV
NumPy
Matplotlib


##Dataset Structure

Training/
├── glioma
├── meningioma
├── pituitary
└── notumor

Testing/
├── glioma
├── meningioma
├── pituitary
└── notumor

##CNN Architecture

Input Image (128x128x3)
        ↓
Rescaling
        ↓
Conv2D (32 Filters)
        ↓
MaxPooling2D
        ↓
Conv2D (64 Filters)
        ↓
MaxPooling2D
        ↓
Flatten
        ↓
Dense (128)
        ↓
Dense (4)
        ↓
Output Class

Classes Predicted

The model predicts one of the following classes:

Glioma
Meningioma
Pituitary
No Tumor

##Results

Training Images: 5600
Testing Images: 1600
Validation Accuracy: 86.94%
Trained Model: brain_tumor_model.h5
## Trained Model

The trained model file (`brain_tumor_model.h5`) is not included due to GitHub file size limitations. The model can be recreated by running `brain_tumor.py`. Because its an computer readable binary format.


##Project Workflow

MRI Image
    ↓
Image Preprocessing
    ↓
CNN Feature Extraction
    ↓
Classification
    ↓
Tumor Prediction

##Applications

Brain Tumor Detection
Medical Image Analysis
Healthcare Decision Support Systems
AI-Assisted Diagnosis
Future Improvements
Web Application Deployment
Real-Time MRI Analysis
Transfer Learning for Higher Accuracy
Multi-Disease Brain MRI Classification  

##Author

ARUN S R
