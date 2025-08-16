import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from timm import create_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Constants
IMG_SIZE = 224
BATCH_SIZE = 16
MODEL_PATH = "F:/New folder/vit_crack_detection.pth"
TEST_DIR = "F:/python robotics/CSB_dataset/patch datasets/512_512/full/test"

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset class
class CrackDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.image_filenames = [f for f in os.listdir(img_dir) if f.endswith(".jpg") and "mask" not in f]
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.img_dir, img_name.replace(".jpg", "mask.jpg"))

        image = Image.open(img_path).convert("RGB")

        # If mask doesn't exist, use black mask (no crack)
        if not os.path.exists(mask_path):
            mask = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)
        else:
            mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        label = 1 if np.any(np.asarray(mask) > 0) else 0
        return image.to(torch.float32), torch.tensor(label, dtype=torch.float32), img_name

# Transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# Load test data
test_dataset = CrackDataset(TEST_DIR, transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load model
model = create_model("vit_base_patch16_224", pretrained=True, num_classes=1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=True)
model.to(device)
model.eval()

# Evaluation
y_true, y_pred, file_names = [], [], []
sample_images, sample_preds, sample_labels = [], [], []

with torch.no_grad():
    for images, labels, names in test_loader:
        images = images.to(device)
        outputs = model(images).squeeze(1)
        preds = torch.sigmoid(outputs) > 0.5

        y_true.extend(labels.tolist())
        y_pred.extend(preds.cpu().tolist())
        file_names.extend(names)

        if len(sample_images) < 5:  # Save 5 samples for visualization
            sample_images.extend(images.cpu())
            sample_preds.extend(preds.cpu())
            sample_labels.extend(labels)

# Accuracy
accuracy = np.mean(np.array(y_true) == np.array(y_pred)) * 100
print(f"✅ Test Accuracy: {accuracy:.2f}%")

# Classification Report
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=["No Crack", "Crack"]))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Crack", "Crack"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Sample Visualization
print("\n🖼️ Sample Predictions:")
for i in range(min(5, len(sample_images))):
    img = sample_images[i].permute(1, 2, 0).numpy()
    true_label = "Crack" if sample_labels[i] == 1 else "No Crack"
    pred_label = "Crack" if sample_preds[i] == 1 else "No Crack"

    plt.imshow(img)
    plt.title(f"True: {true_label} | Pred: {pred_label}")
    plt.axis("off")
    plt.show()
