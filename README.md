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


| Feature | Technical Detail | Human Interviewer Explanation |
| :--- | :--- | :--- |
| Model Architecture | The model is a pre-trained Vision Transformer (ViT-B/16) from torchvision. The classification head is a linear layer with num_classes=1. The input image size is explicitly set to 384x384.  | The model is a Vision Transformer, a cutting-edge AI architecture that analyzes images more comprehensively than older models. We use a version pre-trained on a massive dataset, giving it a strong head start before fine-tuning it for our specific crack detection task.  |
| Data Augmentation | Uses transforms.Compose with geometric transformations like RandomHorizontalFlip(), RandomRotation(10), and RandomPerspective(), as well as photometric ColorJitter().  | This is our way of creating a more diverse training set from our limited data. By randomly flipping, rotating, and altering colors, we ensure the model learns to identify cracks under different angles and lighting conditions, which makes it more resilient in the real world. |
| Loss Function | A custom Focal Loss module is used. This is a modified BCEWithLogitsLoss that down-weights the loss from easy-to-classify examples using alpha and gamma parameters. | Our dataset is imbalanced, with far more \"no crack\" images than \"crack\" images. Focal Loss addresses this by making the model focus its learning on the difficult cases—the actual cracks—ensuring it doesn't get lazy on the easy examples.  |
| Optimizer & Scheduler | The model uses the Adam optimizer with a learning rate of 1e-4. It also includes a ReduceLROnPlateau scheduler that automatically reduces the learning rate if the training loss plateaus.  | We use the Adam optimizer to efficiently adjust the model’s internal parameters.The scheduler is like a smart assistant that lowers the learning rate when progress slows, helping the model find the best possible solution without overshooting it. |
| Class Balancing | A WeightedRandomSampler is used during training.It assigns higher sampling weights to the minority class (\"Crack\") to ensure a balanced mix of classes in each training batch. | This is a direct approach to solving the imbalanced dataset problem. Instead of just adjusting the loss, we actively ensure the model sees a balanced number of both crack and no-crack images during training, which forces it to learn from both equally. |


## **Key Features of `step 4(evaluation and visualisation ).py`**

This script is dedicated to the **evaluation** and **visualization** of a trained model. It's used after `STEP 3` to gain a deeper understanding of the model's performance.

| Feature | Technical Detail | Human Interviewer Explanation |
| :--- | :--- | :--- |
| Model Loading | The script loads the model and its saved state dictionary (vit_crack_detection.pth) using torch.load(). | We separate training and evaluation for a cleaner workflow. This allows us to load our pre-trained model and test it on new data without having to run the entire training process again.  |
| Performance Metrics | It calculates accuracy and generates a classification_report from sklearn.metrics, providing per-class metrics such as precision, recall, and F1-score. | We get more than just a simple accuracy score. This report tells us how reliable our model is at finding cracks (precision) and how many cracks it actually finds (recall), which is vital for knowing if it's fit for the job.  |
| Confusion Matrix | It uses sklearn.metrics.confusion_matrix and ConfusionMatrixDisplay to visualize the results.  | This is our most important tool for understanding the model's performance. The chart provides a clear breakdown of correct and incorrect predictions, showing us exactly how many times the model correctly identified a crack versus how many times it missed one.  |
| Sample Visualization | Displays PIL images from the test set and uses matplotlib.pyplot to show the true label and predicted label. [cite: 1] | This provides a quick visual check of the model’s behavior on a few random examples. It helps us see firsthand whether it's making the right predictions and can provide insight into the types of images it struggles with.  |


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
