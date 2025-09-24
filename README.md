# 🧠 Steel Crack Detection using Vision Transformers (ViT)
<p align="center"> <img src="confusion_matrix.png" width="600" alt="Confusion Matrix"/> </p>

## 📖 Introduction

Steel surfaces in industries like shipbuilding, automotive, and infrastructure often develop cracks from fatigue, stress, or corrosion.
This project leverages Vision Transformers (ViT) for automated crack detection, achieving higher precision, recall, and generalization compared to CNN-based methods.

## 🎯 Problem Statement

Manual inspection is time-consuming & error-prone

Traditional methods fail for fine cracks or variable lighting

Aim: build a robust ViT model to classify steel patches into Crack / No Crack

## 📂 Dataset
[Open GitHub](https://github.com)


Patch Size: 512×512 px

Training Samples: ~20,000

Testing Samples: ~6,000

Class Distribution:

Crack: ~3,540

No Crack: ~51,397 (imbalanced)

##⚙️ Methodology

Preprocessing: Resize (384×384), normalization

Augmentation: Flips, rotations, color jitter, affine & perspective transforms

Imbalance Handling: WeightedRandomSampler + Focal Loss (α=0.5, γ=1.5)

Model: ViT-B/16 (ImageNet-1K pretrained) + binary classifier head

Optimizer: Adam (lr=1e-4) + ReduceLROnPlateau scheduler

## 🏋️ Training Setup

Epochs: 15

Batch Size: 32

Device: CUDA GPU

Metrics: Precision, Recall, F1-score, Accuracy, Confusion Matrix

## **Key Features of `STEP 3 UPGRADED(training).py`**

This script is designed for the **training** phase and incorporates several advanced techniques to create a more effective and reliable model.

* **Advanced Data Augmentation**: It uses a rich set of transformations to diversify the training data, including `RandomHorizontalFlip`, `RandomRotation`, `ColorJitter`, `RandomAffine`, and `RandomPerspective`. This helps the model generalize better to new, unseen images and reduces overfitting.  
* **Focal Loss Implementation**: The script uses a custom `FocalLoss` function, which is particularly effective for highly imbalanced datasets like crack detection, where "no crack" samples far outnumber "crack" samples. This loss function down-weights the contribution of easy-to-classify examples, forcing the model to focus on the more difficult ones.  
* **Class Imbalance Handling**: To directly address the class imbalance, it uses a `WeightedRandomSampler` to ensure that both crack and no-crack images are represented more equally in each training batch. This is a more proactive approach than simply adjusting the loss weight.  
* **Dynamic Learning Rate**: The script includes a `ReduceLROnPlateau` scheduler, which automatically decreases the learning rate when the training loss stops improving. This helps the model converge more effectively to a better solution.  
* **Comprehensive Evaluation**: After training, it evaluates the model's performance by generating a classification report and a confusion matrix heatmap, providing a clear visual summary of its accuracy, precision, recall, and F1-score.


## **Key Features of `step 4(evaluation and visualisation ).py`**

This script is dedicated to the **evaluation** and **visualization** of a trained model. It's used after `STEP 3` to gain a deeper understanding of the model's performance.

* **Model Loading and Evaluation**: It loads a pre-trained model and evaluates it on the test dataset. The script calculates and prints the overall test accuracy, providing a quick performance metric.  
* **Detailed Performance Metrics**: It generates and prints a comprehensive **classification report**, which includes precision, recall, and F1-score for both "No Crack" and "Crack" classes. This is crucial for understanding how the model performs on each class individually, especially given the dataset's imbalance.  
* **Confusion Matrix Visualization**: The script creates and displays a **confusion matrix** using `ConfusionMatrixDisplay` from `sklearn`. This visual representation breaks down the model's predictions into true positives, true negatives, false positives, and false negatives, which is essential for diagnosing specific types of errors.  
* **Sample Prediction Visualization**: It visualizes a small number of sample images from the test set, displaying the true label and the model's predicted label. This provides an intuitive and qualitative look at how the model performs on individual examples, showing both correct and incorrect predictions.


## 📊 Results

Classification Report

No Crack → Precision: 0.99 | Recall: 0.98 | F1: 0.985
Crack    → Precision: 0.92 | Recall: 0.94 | F1: 0.93


✅ Crack recall: 94%
✅ False negatives reduced vs CNN baseline

<p align="center"> <img src="confusion_matrix.png" width="400"/> </p>

## 💡 Key Insights

ViT achieves higher recall → fewer missed cracks

Stronger generalization vs CNNs

Trade-off: longer training & higher GPU usage

## 🚀 Future Scope

Real-time crack detection in video streams

Integration with robotic arms / UAVs

Multi-defect detection (beyond cracks)

Hybrid CNN–ViT for faster inference

## 🛠️ Tech Stack

Python, PyTorch, TorchVision, timm

Vision Transformer (ViT-B/16)

scikit-learn, matplotlib, seaborn

CUDA for training acceleration

## 🏆 Resume Highlight

“Developed a ViT-based crack detection pipeline with focal loss, balanced sampling, and augmentation — achieving 94% recall on cracks while reducing false negatives compared to CNNs.”
