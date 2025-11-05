import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from tqdm import tqdm
import kagglehub
import warnings

# Suppress warnings from torchvision
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# === CONFIGURATION ===
# =============================================================================

# Set this to True for a quick test run, False for the full dataset
USE_SMALL_SUBSET = False
SUBSET_TRAIN_SIZE = 5000
SUBSET_VAL_SIZE = 1000

# ‼️ === IMPORTANT === ‼️
# Change this to a path on your local computer
YOUR_IMAGE_PATH_TO_TEST = "C:/Users/kaifm/Downloads/PythonProject/download.jpeg"  # 👈 REPLA'CE THIS

# Model parameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
# Note: 3 epochs is very low for ResNet-18. You will get bad results.
# I strongly recommend 10-15 epochs for a real model.
NUM_EPOCHS_RESNET50 = 5
NUM_EPOCHS_RESNET18 = 3
OOD_THRESHOLD = 25.0


# =============================================================================
# === 1. DATASET CLASS DEFINITION ===
# =============================================================================
class CelebASpoofDataset(Dataset):
    """Custom PyTorch Dataset for the CelebA-Spoof data."""

    def __init__(self, df, base_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.base_dir = base_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_dir, self.df.loc[idx, 'filename'])
        try:
            image = Image.open(img_path).convert("RGB")
            label = int(self.df.loc[idx, 'label'])
            if self.transform:
                image = self.transform(image)
            return image, label
        except FileNotFoundError:
            print(f"Warning: File not found {img_path}. Skipping.")
            # Return a placeholder (or handle differently)
            return self.transform(Image.new('RGB', (224, 224))), -1
        except Exception as e:
            print(f"Error loading {img_path}: {e}. Skipping.")
            return self.transform(Image.new('RGB', (224, 224))), -1


# =============================================================================
# === 2. OOD/MAHALANOBIS FUNCTIONS ===
# =============================================================================

def extract_embeddings(loader, model, device):
    """Extracts deep features from the model for all data in the loader."""
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="Extracting embeddings"):
            imgs = imgs.to(device)
            # Manually pass through layers up to avgpool
            feats = model.conv1(imgs)
            feats = model.bn1(feats)
            feats = model.relu(feats)
            feats = model.maxpool(feats)
            feats = model.layer1(feats)
            feats = model.layer2(feats)
            feats = model.layer3(feats)
            feats = model.layer4(feats)
            feats = model.avgpool(feats)

            feats = torch.flatten(feats, 1).cpu().numpy()
            features.append(feats)
            labels.extend(lbls.numpy())

    return np.vstack(features), np.array(labels)


def mahalanobis_ood_score(feature, class_means, class_covs):
    """Calculates the minimum Mahalanobis distance to any known class."""
    dists = []
    for cls in class_means:
        try:
            inv_cov = np.linalg.inv(class_covs[cls])
            dist = distance.mahalanobis(feature, class_means[cls], inv_cov)
            dists.append(dist)
        except np.linalg.LinAlgError:
            print(f"Warning: Covariance matrix for class {cls} is singular. Skipping.")
            dists.append(float('inf'))

    return min(dists) if dists else float('inf')


def predict_with_ood(image_path, model, device, transform, class_names, class_means, class_covs, threshold):
    """Classifies an image and detects if it's an 'Unknown Attack'."""
    model.eval()
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred_class = torch.max(outputs, 1)

        # Manually pass through layers up to avgpool
        feats = model.conv1(input_tensor)
        feats = model.bn1(feats)
        feats = model.relu(feats)
        feats = model.maxpool(feats)
        feats = model.layer1(feats)
        feats = model.layer2(feats)
        feats = model.layer3(feats)
        feats = model.layer4(feats)
        feats = model.avgpool(feats)
        feats = torch.flatten(feats, 1).cpu().numpy()[0]

    # OOD distance
    score = mahalanobis_ood_score(feats, class_means, class_covs)

    if score > threshold:
        prediction = "🧠 Unknown / Unseen Attack Detected"
        color = "orange"
    else:
        pred_label_name = class_names[pred_class.item()]
        prediction = f"Classified as: {pred_label_name}"
        # Assuming class 0 is 'real'
        color = "green" if pred_label_name == 0 else "red"

    plt.imshow(img)
    plt.axis("off")
    plt.title(f"{prediction}\n(Mahalanobis Distance = {score:.2f})", color=color)
    plt.show()

    return prediction, score


# =============================================================================
# === 3. MAIN EXECUTION ===
# =============================================================================
def main():
    """Main function to run the entire pipeline."""

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Dataset Loading ---
    print("Downloading dataset (this may take a while)...")
    path = kagglehub.dataset_download("nlt2k2/celebaspoof-retinaface-extracted")
    print(f"Dataset path: {path}")

    train_txt = os.path.join(path, "train.txt")
    test_txt = os.path.join(path, "test.txt")
    val_txt = os.path.join(path, "val.txt")

    train_df = pd.read_csv(train_txt, sep=' ', header=None, names=['filename', 'label'])
    test_df = pd.read_csv(test_txt, sep=' ', header=None, names=['filename', 'label'])
    val_df = pd.read_csv(val_txt, sep=' ', header=None, names=['filename', 'label'])

    print(f"✅ Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    base_image_dir = os.path.join(path, "Dataset", "CelebA_Spoof_RetinaFaceExtracted")
    print(f"✅ Using base image directory: {base_image_dir}")

    # --- 2. Transforms and Dataloaders ---
    print("Setting up Dataloaders...")
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = CelebASpoofDataset(train_df, base_image_dir, transform=train_transform)
    val_dataset = CelebASpoofDataset(val_df, base_image_dir, transform=test_transform)
    test_dataset = CelebASpoofDataset(test_df, base_image_dir, transform=test_transform)

    if USE_SMALL_SUBSET:
        print(f"⚡ Using small subset: {SUBSET_TRAIN_SIZE} train, {SUBSET_VAL_SIZE} val")
        train_dataset = Subset(train_dataset, range(SUBSET_TRAIN_SIZE))
        val_dataset = Subset(val_dataset, range(SUBSET_VAL_SIZE))

    # Note: num_workers > 0 can cause issues on Windows if not in main()
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    print("✅ Dataloaders ready.")

    # --- 3. Model 1: ResNet-50 (Binary) ---
    print("\n--- Training Model 1: ResNet-50 (Binary) ---")
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    for param in resnet.parameters():
        param.requires_grad = False

    num_features = resnet.fc.in_features
    resnet.fc = nn.Linear(num_features, 2)
    resnet = resnet.to(device)
    print("✅ ResNet-50 model ready.")

    criterion_res50 = nn.CrossEntropyLoss()
    optimizer_res50 = optim.Adam(resnet.fc.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS_RESNET50):
        resnet.train()
        running_loss, correct, total = 0, 0, 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{NUM_EPOCHS_RESNET50}]")
        for images, labels in loop:
            # Skip bad batches
            if -1 in labels:
                continue

            images, labels = images.to(device), labels.to(device)

            optimizer_res50.zero_grad()
            outputs = resnet(images)
            loss = criterion_res50(outputs, labels)
            loss.backward()
            optimizer_res50.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            loop.set_postfix(loss=running_loss / (total / BATCH_SIZE), acc=100. * correct / total)

        # Validation
        resnet.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                if -1 in labels:
                    continue
                images, labels = images.to(device), labels.to(device)
                outputs = resnet(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total
        print(f"✅ Epoch {epoch + 1}: Train Acc: {100. * correct / total:.2f}%, Val Acc: {val_acc:.2f}%")

    print("🎯 ResNet-50 training completed.")
    torch.save(resnet.state_dict(), "resnet50_face_spoof.pth")
    print("✅ ResNet-50 model saved as resnet50_face_spoof.pth")

    # --- 4. Test ResNet-50 (Local Path) ---
    print("\n--- Testing ResNet-50 on local image ---")
    if os.path.exists(YOUR_IMAGE_PATH_TO_TEST):
        try:
            image = Image.open(YOUR_IMAGE_PATH_TO_TEST).convert("RGB")
            input_tensor = test_transform(image).unsqueeze(0).to(device)

            resnet.eval()
            with torch.no_grad():
                output = resnet(input_tensor)
                pred = torch.argmax(output, dim=1).item()

            label_map = {0: "Spoof Face", 1: "Real Face"}  # Adjust if 0/1 are swapped
            pred_label = label_map.get(pred, "Unknown")
            color = 'green' if pred_label == "Real Face" else 'red'

            plt.imshow(image)
            plt.axis('off')
            plt.title(f"ResNet-50 Prediction: {pred_label}", color=color, fontsize=14)
            plt.show()
            print(f"Prediction for {YOUR_IMAGE_PATH_TO_TEST}: {pred_label}")
        except Exception as e:
            print(f"❌ Error testing ResNet-50: {e}")
    else:
        print(f"⚠️ Warning: Test image not found at {YOUR_IMAGE_PATH_TO_TEST}. Skipping ResNet-50 test.")

    # --- 5. Model 2: ResNet-18 (Multi-Class + OOD) ---
    print("\n--- Training Model 2: ResNet-18 (Multi-Class + OOD) ---")

    # Note: Your dataset has 2 classes, so this is still binary,
    # but it's a different model (ResNet-18)
    num_classes = len(train_df['label'].unique())
    print(f"Detected {num_classes} classes.")

    model = torch.hub.load('pytorch/vision', 'resnet18', weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    print("✅ ResNet-18 model ready.")

    criterion_res18 = nn.CrossEntropyLoss()
    optimizer_res18 = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Training ResNet-18 for {NUM_EPOCHS_RESNET18} epochs. (Consider increasing for better accuracy)")

    for epoch in range(NUM_EPOCHS_RESNET18):
        model.train()
        total_loss, correct, total = 0, 0, 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS_RESNET18}")
        for imgs, labels in loop:
            if -1 in labels:
                continue
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer_res18.zero_grad()
            outputs = model(imgs)
            loss = criterion_res18(outputs, labels)
            loss.backward()
            optimizer_res18.step()
            total_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            loop.set_postfix(loss=total_loss / (total / BATCH_SIZE), acc=100. * correct / total)

        print(
            f"✅ Epoch {epoch + 1} | Loss: {total_loss / len(train_loader):.4f} | Train Acc: {100 * correct / total:.2f}%")

    torch.save(model.state_dict(), "multiclass_face_spoof.pth")
    print("🎯 ResNet-18 classifier trained and saved.")

    # --- 6. Evaluate ResNet-18 ---
    print("\n--- Evaluating ResNet-18 ---")
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating ResNet-18"):
            if -1 in labels:
                continue
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted')
    rec = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"✅ Test Accuracy: {acc * 100:.2f}%")
    print(f"🔹 Precision: {prec:.3f} | Recall: {rec:.3f} | F1 Score: {f1:.3f}")

    class_names = sorted(train_df['label'].unique())
    print("\n🎯 Generating confusion matrices...")

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Overall Confusion Matrix - ResNet-18")
    plt.show()

    for i, cls in enumerate(class_names):
        binary_true = (np.array(all_labels) == i).astype(int)
        binary_pred = (np.array(all_preds) == i).astype(int)

        cm = confusion_matrix(binary_true, binary_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f"Not {cls}", str(cls)])
        disp.plot(cmap='Reds')
        plt.title(f"Confusion Matrix for class '{cls}'")
        plt.show()

    # --- 7. Build OOD Feature Space ---
    print("\n--- Building Mahalanobis OOD Feature Space ---")
    # Must use the full train_loader, not the subset, for a good distribution
    full_train_loader = DataLoader(
        CelebASpoofDataset(train_df, base_image_dir, transform=train_transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )

    train_feats, train_labels = extract_embeddings(full_train_loader, model, device)

    class_means, class_covs = {}, {}
    for cls in np.unique(train_labels):
        cls_feats = train_feats[train_labels == cls]
        class_means[cls] = np.mean(cls_feats, axis=0)
        class_covs[cls] = np.cov(cls_feats, rowvar=False) + np.eye(cls_feats.shape[1]) * 1e-6
    print("✅ Mahalanobis feature space built.")

    # --- 8. Test OOD Predictor ---
    print("\n--- Testing OOD Predictor on local image ---")
    if os.path.exists(YOUR_IMAGE_PATH_TO_TEST):
        try:
            prediction, score = predict_with_ood(
                YOUR_IMAGE_PATH_TO_TEST, model, device, test_transform,
                class_names, class_means, class_covs, OOD_THRESHOLD
            )
            print(f"🎯 OOD Result: {prediction}")
            print(f"📏 Mahalanobis Distance: {score:.2f}")
        except Exception as e:
            print(f"❌ Error testing OOD: {e}")
    else:
        print(f"⚠️ Warning: Test image not found at {YOUR_IMAGE_PATH_TO_TEST}. Skipping OOD test.")

    # --- 9. Save Final OOD Model ---
    print("\n--- Saving final model with OOD parameters ---")
    torch.save({
        "model_state": model.state_dict(),
        "class_means": class_means,
        "class_covs": class_covs,
        "classes": class_names
    }, "multiclass_facepad_with_ood.pth")

    print("✅ Model and OOD parameters saved successfully.")
    print("\n🎉 --- Script Finished --- 🎉")


# This is the standard entry point for Python scripts
if __name__ == "__main__":
    # We must put the main code block here to support multiprocessing
    # on Windows for the DataLoader
    main()