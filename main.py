import os
import random
import time
import subprocess
import shutil
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib
# Use non-interactive backend if no display is available
if not os.environ.get('DISPLAY', ''):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ==========================================
# CONFIGURATION & HYPERPARAMETERS
# ==========================================
class Config:
    SEED = 6220
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0001  # Lower learning rate for stable VGG-16 training from scratch
    EPOCHS = 15
    NUM_WORKERS = min(4, multiprocessing.cpu_count())
    DATA_PATH = './data/'
    SAVING_PATH = './Saving_Path'
    TESTDATA_PATH = './data/bangla/Testing'
    TRAINDATA_PATH = './data/bangla/Training'
    LABEL_MAPPING = {"1": 0, "2": 1, "5": 2, "10": 3, "20": 4, "50": 5, "100": 6, "500": 7, "1000": 8}
    CLASS_NAMES = ('1', '10', '100', '1000', '2', '20', '5', '50', '500')
    NUM_CLASSES = 9

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# ==========================================
# DATASET UTILITIES
# ==========================================
def download_dataset(data_path):
    dataset_dir = os.path.join(data_path, "bangla")
    if not os.path.exists(data_path) or not os.path.exists(dataset_dir):
        print("Dataset not found. Downloading Bangla-Money-Dataset from GitHub...")
        os.makedirs(data_path, exist_ok=True)
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/nsojib/Bangla-Money-Dataset.git",
            dataset_dir
        ], check=True)
        # Remove git tracking to avoid nested git repository issues
        git_dir = os.path.join(dataset_dir, ".git")
        if os.path.exists(git_dir):
            shutil.rmtree(git_dir)
        print("Dataset downloaded successfully!")
    else:
        print("Dataset already exists.")

def get_img_info(data_dir, label_mapping):
    imgpath = []
    imglabel = []
    for root, dirs, _ in os.walk(data_dir):
        for sub_dir in dirs:
            if sub_dir in label_mapping:
                sub_dir_path = os.path.join(root, sub_dir)
                img_names = os.listdir(sub_dir_path)
                img_names = [f for f in img_names if f.endswith('.jpg')]
                for img_name in img_names:
                    imgpath.append(os.path.join(sub_dir_path, img_name))
                    imglabel.append(label_mapping[sub_dir])
    return imgpath, imglabel

class CustomDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index]).convert('RGB')
        label = self.labels[index]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.img_paths)

class CustomTestDataset(Dataset):
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.img_paths)

def denormalize_image(tensor):
    # Denormalize image using standard ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.cpu().numpy().transpose((1, 2, 0))
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

# ==========================================
# MODEL DEFINITION (VGG-16)
# ==========================================
class VGG16(nn.Module):
    def __init__(self, num_classes=Config.NUM_CLASSES):
        super(VGG16, self).__init__()

        self.conv_block = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.feat_classifier = nn.Sequential(
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = torch.flatten(x, 1)
        x = self.feat_classifier(x)
        return x

# ==========================================
# TRAINING & VALIDATION PIPELINE
# ==========================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data, target in tqdm(loader, desc="Training"):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = (correct / total) * 100
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in tqdm(loader, desc="Validation"):
            target = target.long()
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            running_loss += loss.item() * data.size(0)
            total += target.size(0)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = (correct / total) * 100
    return epoch_loss, epoch_acc

# ==========================================
# PLOTTING UTILITIES
# ==========================================
def save_and_show_plots(train_losses, valid_losses, train_accs, valid_accs, saving_path):
    os.makedirs(saving_path, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Loss Curve
    ax1.plot(epochs, train_losses, label='Train Loss', color='blue', marker='o')
    ax1.plot(epochs, valid_losses, label='Valid Loss', color='red', marker='x')
    ax1.set_title('Training & Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy Curve
    ax2.plot(epochs, train_accs, label='Train Acc', color='blue', marker='o')
    ax2.plot(epochs, valid_accs, label='Valid Acc', color='red', marker='x')
    ax2.set_title('Training & Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plot_file = os.path.join(saving_path, 'loss_accuracy_curves.png')
    plt.savefig(plot_file, dpi=300)
    print(f"Saved training curves to {plot_file}")
    plt.show()

def trueclass(num):
    # Map index to denomination
    mapping = {0: 1, 1: 2, 2: 5, 3: 10, 4: 20, 5: 50, 6: 100, 7: 500, 8: 1000}
    return mapping.get(num, 0)

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    # 1. Setup
    set_seed(Config.SEED)
    os.makedirs(Config.SAVING_PATH, exist_ok=True)

    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 2. Download and Load Data
    download_dataset(Config.DATA_PATH)
    
    # Define Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load image paths and labels
    img_paths, labels = get_img_info(Config.TRAINDATA_PATH, Config.LABEL_MAPPING)
    train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(
        img_paths, labels, test_size=0.2, random_state=42
    )

    # Build Datasets and DataLoaders
    train_set = CustomDataset(train_imgs, train_lbls, transform=train_transform)
    valid_set = CustomDataset(val_imgs, val_lbls, transform=valid_transform)

    print(f"Total training samples: {len(train_set)}")
    print(f"Total validation samples: {len(valid_set)}")

    train_loader = DataLoader(
        train_set, batch_size=Config.BATCH_SIZE, shuffle=True,
        num_workers=Config.NUM_WORKERS, pin_memory=True
    )
    valid_loader = DataLoader(
        valid_set, batch_size=Config.BATCH_SIZE, shuffle=False,
        num_workers=Config.NUM_WORKERS, pin_memory=True
    )

    # 3. Model, Loss, Optimizer & Scheduler
    model = VGG16(num_classes=Config.NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Cosine annealing lr scheduler for standard training
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # 4. Training Loop
    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []
    valid_loss_min = np.inf

    print("\n--- Starting Training ---")
    for epoch in range(1, Config.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{Config.EPOCHS} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_acc = validate(model, valid_loader, criterion, device)
        
        scheduler.step()

        print(f"Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.2f}%")
        print(f"Valid Loss: {valid_loss:.6f} | Valid Acc: {valid_acc:.2f}%")

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        valid_loss_list.append(valid_loss)
        valid_acc_list.append(valid_acc)

        # Save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            best_model_path = os.path.join(Config.SAVING_PATH, 'model_weight.pth')
            print(f"Validation loss decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}). Saving model weights to {best_model_path}")
            torch.save(model.state_dict(), best_model_path)
            valid_loss_min = valid_loss

    print("\nFinished Training!")

    # 5. Plot Loss & Accuracy Curves
    save_and_show_plots(train_loss_list, valid_loss_list, train_acc_list, valid_acc_list, Config.SAVING_PATH)

    # 6. Inference on Test Set
    print("\n--- Starting Inference ---")
    
    # Read test images
    for data in os.walk(Config.TESTDATA_PATH):
        test_filenames = [f for f in data[2] if f.endswith('.jpg')]
    
    test_paths = [os.path.join(Config.TESTDATA_PATH, f) for f in test_filenames]
    
    # Load Best Model Weights
    best_model_path = os.path.join(Config.SAVING_PATH, 'model_weight.pth')
    if os.path.exists(best_model_path):
        print(f"Loading best weights from {best_model_path} for test inference...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    
    # Setup test dataset and loader
    test_set = CustomTestDataset(test_paths, transform=valid_transform)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=Config.NUM_WORKERS)

    # Run predictions
    model.eval()
    prediction_classes = []
    ground_truth_classes = []
    
    with torch.no_grad():
        for i, images in enumerate(tqdm(test_loader, desc="Predicting")):
            images = images.to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)
            
            pred_val = trueclass(predicted.item())
            prediction_classes.append(pred_val)
            
            # Parse ground truth from filename
            filename = test_filenames[i]
            true_val = int(filename.split('_')[0])
            ground_truth_classes.append(true_val)

    # Calculate and display test accuracy
    correct = sum(1 for p, g in zip(prediction_classes, ground_truth_classes) if p == g)
    total = len(test_filenames)
    test_accuracy = (correct / total) * 100
    print(f"\nTest Accuracy: {correct}/{total} ({test_accuracy:.2f}%)")

    # Export to CSV
    results_df = pd.DataFrame({
        'image': test_filenames,
        'class': prediction_classes
    })
    
    csv_output_path = './data/example.csv'
    results_df.to_csv(csv_output_path, index=False)
    print(f"\nSaved test predictions to {csv_output_path}")
    print(results_df.head(10))

if __name__ == '__main__':
    # PyTorch multiprocess setup (fork is only supported on Unix-like systems)
    if os.name != 'nt':
        try:
            multiprocessing.set_start_method("fork", force=True)
        except RuntimeError:
            pass
    main()