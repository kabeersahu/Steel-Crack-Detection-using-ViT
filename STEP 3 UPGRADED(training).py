import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------- Dataset -------------------------
class CrackDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg") and "mask" not in f])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_file = self.images[idx]
        img_path = os.path.join(self.img_dir, img_file)
        mask_path = os.path.join(self.mask_dir, img_file.replace(".jpg", "mask.jpg"))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") if os.path.exists(mask_path) else Image.new("L", image.size)

        if self.transform:
            image = self.transform(image)

        label = 1 if np.array(mask).sum() > 0 else 0
        return image, label

# ------------------------- Focal Loss -------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=1.5, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.float().view(-1)
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

# ------------------------- ViT Loader -------------------------
def get_vit_model(num_classes):
    from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights)
    model.image_size = 384  # ✅ Add this line to match resized input
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model

# ------------------------- Train -------------------------
def train():
    train_img_dir = r"F:\python robotics\CSB_dataset\patch datasets\512_512\full\train"
    test_img_dir = r"F:\python robotics\CSB_dataset\patch datasets\512_512\full\test"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((384, 384), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
    ])

    train_dataset = CrackDataset(train_img_dir, train_img_dir, transform=train_transform)
    test_dataset = CrackDataset(test_img_dir, test_img_dir, transform=test_transform)

    # Auto class balancing
    targets = [label for _, label in train_dataset]
    crack_count = sum(targets)
    no_crack_count = len(targets) - crack_count
    print(f"✅ Detected Counts -> Crack: {crack_count}, No Crack: {no_crack_count}")

    weights = 1. / torch.tensor([no_crack_count, crack_count], dtype=torch.float)
    sample_weights = [weights[t] for t in targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    model = get_vit_model(num_classes=1).to(device)
    criterion = FocalLoss(alpha=0.5, gamma=1.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)

    # Training
    model.train()
    for epoch in range(15):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"📈 Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        scheduler.step(avg_loss)

    # Evaluation
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = torch.sigmoid(model(images).squeeze())
            preds = (outputs > 0.5).int().cpu().numpy()
            y_true.extend(labels.numpy())
            y_pred.extend(preds)

    print("\n📊 Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=["No Crack", "Crack"]))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No Crack", "Crack"], yticklabels=["No Crack", "Crack"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("✅ Confusion matrix saved as confusion_matrix.png")

    # Save model
    torch.save(model.state_dict(), "vit_crack_detection.pth")
    print("✅ Model saved at vit_crack_detection.pth")

# ------------------------- Execute -------------------------
if __name__ == '__main__':
    train()
