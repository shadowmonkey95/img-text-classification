import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from transformers import AutoImageProcessor, ResNetForImageClassification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

class TrademarkDataset(Dataset):
    def __init__(self, df, target_column, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.target_column = target_column
        
        # Encode target labels
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(df[target_column].str.split(',').str[0])
        
        # Calculate class weights
        self.class_weights = self._calculate_class_weights()
    
    def _calculate_class_weights(self):
        counts = Counter(self.labels)
        total = len(self.labels)
        weights = {cls: total / (len(counts) * count) for cls, count in counts.items()}
        return weights
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        try:
            img_path = os.path.join(self.img_dir, self.df.iloc[idx]['image_name'])
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return image, label
        except Exception as e:
            print(f"Error loading image {self.df.iloc[idx]['image_name']}: {str(e)}")
            raise

def validate_image_existence(df, img_dir):
    """
    Check and filter the DataFrame to keep only rows where images exist and can be opened.
    """
    missing_images = []
    corrupted_images = []
    valid_rows = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Checking images"):
        img_path = os.path.join(img_dir, row['image_name'])
        try:
            # Try to open the image to check if it's valid
            with Image.open(img_path) as img:
                img.verify()  # Verify it's actually an image
            valid_rows.append(True)
        except (FileNotFoundError, Image.UnidentifiedImageError, IOError):
            valid_rows.append(False)
            if not os.path.exists(img_path):
                missing_images.append(row['image_name'])
            else:
                corrupted_images.append(row['image_name'])
    
    valid_df = df[valid_rows].copy()
    
    print(f"\nTotal images in CSV: {len(df)}")
    print(f"Missing images: {len(missing_images)}")
    print(f"Corrupted images: {len(corrupted_images)}")
    print(f"Valid images: {len(valid_df)}")
    
    return valid_df, missing_images, corrupted_images

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

def train_trademark_classifier(df, target_column, img_dir, num_epochs=30, batch_size=32, patience=5):
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    # Create datasets with appropriate transforms
    train_dataset = TrademarkDataset(train_df, target_column, img_dir, get_transforms(train=True))
    val_dataset = TrademarkDataset(val_df, target_column, img_dir, get_transforms(train=False))
    test_dataset = TrademarkDataset(test_df, target_column, img_dir, get_transforms(train=False))
    
    # Create weighted sampler for training data
    sample_weights = [train_dataset.class_weights[label] for label in train_dataset.labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    num_classes = len(train_dataset.label_encoder.classes_)
    model = ResNetForImageClassification.from_pretrained(
        "microsoft/resnet-50",
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Define optimizer, loss function, and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = FocalLoss(gamma=2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=3, verbose=True
    )
    
    # Training loop with early stopping
    best_val_acc = 0
    no_improve = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).logits
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # Calculate metrics
        train_acc = 100.*train_correct/train_total
        val_acc = 100.*val_correct/val_total
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Print epoch statistics
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, '
              f'Val Acc: {val_acc:.2f}%')
        
        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_model_{target_column}.pth')
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'\nEarly stopping triggered after {epoch+1} epochs')
                break
    
    # Test phase
    model.load_state_dict(torch.load(f'best_model_{target_column}.pth'))
    model.eval()
    
    test_predictions = []
    test_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images).logits
            _, predicted = outputs.max(1)
            test_predictions.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    # Convert numeric predictions back to original labels
    pred_labels = train_dataset.label_encoder.inverse_transform(test_predictions)
    true_labels = train_dataset.label_encoder.inverse_transform(test_labels)
    
    # Print classification report and confusion matrix
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels))
    
    # Save confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    np.save(f'confusion_matrix_{target_column}.npy', cm)
    
    return model, train_dataset.label_encoder

def main():
    print("Starting trademark classification pipeline...")
    
    # Load data
    df = pd.read_csv('./data/csv/pretrain_fill.csv')
    img_dir = './data/img/'  # Updated path to match your structure
    
    # Validate images first
    print("\nValidating image files...")
    valid_df, missing_images, corrupted_images = validate_image_existence(df, img_dir)
    
    # Save validation results
    if len(missing_images) > 0 or len(corrupted_images) > 0:
        print("\nWarning: Some images are missing or corrupted!")
        
        # Save missing images list
        with open('missing_images.txt', 'w') as f:
            f.write("Missing images:\n")
            for img in missing_images:
                f.write(f"{img}\n")
            f.write("\nCorrupted images:\n")
            for img in corrupted_images:
                f.write(f"{img}\n")
        print("Image issues list saved to 'missing_images.txt'")
        
        # Save valid dataset
        valid_df.to_csv('pretrain_fill_valid.csv', index=False)
        print("Valid dataset saved to 'pretrain_fill_valid.csv'")
    
    # Check if we have enough data to proceed
    if len(valid_df) < 10:
        print("Error: Not enough valid images to train. Please check your dataset.")
        return
    
    # Train models for different hierarchical levels
    # hierarchical_levels = ['target', 'target_h1', 'target_h2', 'target_h3']
    hierarchical_levels = ['target_h1']
    
    for level in hierarchical_levels:
        print(f"\n{'='*50}")
        print(f"Training model for {level}")
        print(f"{'='*50}")
        
        try:
            model, label_encoder = train_trademark_classifier(valid_df, level, img_dir)
            
            # Save label encoder for later use
            joblib.dump(label_encoder, f'label_encoder_{level}.pkl')
            print(f"\nModel and label encoder for {level} saved successfully")
            
        except Exception as e:
            print(f"\nError training model for {level}: {str(e)}")
            continue

if __name__ == "__main__":
    main()
    
# Starting trademark classification pipeline...

# Validating image files...
# Checking images: 100%|█████████████████████████████████████████████████████████████████████████████| 158511/158511 [01:34<00:00, 1685.36it/s]

# Total images in CSV: 158511
# Missing images: 55178
# Corrupted images: 12
# Valid images: 103321

# Warning: Some images are missing or corrupted!
# Image issues list saved to 'missing_images.txt'
# Valid dataset saved to 'pretrain_fill_valid.csv'

# ==================================================
# Training model for target_h1
# ==================================================
# Some weights of ResNetForImageClassification were not initialized from the model checkpoint at microsoft/resnet-50 and are newly initialized because the shapes did not match:
# - classifier.1.bias: found shape torch.Size([1000]) in the checkpoint and torch.Size([30]) in the model instantiated
# - classifier.1.weight: found shape torch.Size([1000, 2048]) in the checkpoint and torch.Size([30, 2048]) in the model instantiated
# You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
# /home/pham_dinh_vu/code/img-text-classification/venv/lib/python3.10/site-packages/torch/optim/lr_scheduler.py:62: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
#   warnings.warn(
# Epoch 1/30: 100%|████████████████████████████████████████████████████████████████████████████████████████| 2261/2261 [20:04<00:00,  1.88it/s]
# Epoch 1/30:
# Train Loss: 3.1178, Train Acc: 7.57%
# Val Loss: 3.0241, Val Acc: 5.31%
# Epoch 2/30: 100%|████████████████████████████████████████████████████████████████████████████████████████| 2261/2261 [20:10<00:00,  1.87it/s]
# Epoch 2/30:
# Train Loss: 2.8970, Train Acc: 13.92%
# Val Loss: 2.9319, Val Acc: 6.74%
# Epoch 3/30: 100%|████████████████████████████████████████████████████████████████████████████████████████| 2261/2261 [19:54<00:00,  1.89it/s]
# Epoch 3/30:
# Train Loss: 2.7139, Train Acc: 17.43%
# Val Loss: 2.7559, Val Acc: 11.76%
# Epoch 4/30: 100%|████████████████████████████████████████████████████████████████████████████████████████| 2261/2261 [19:45<00:00,  1.91it/s]
# Epoch 4/30:
# Train Loss: 2.5953, Train Acc: 19.99%
# Val Loss: 2.6437, Val Acc: 13.63%
# Epoch 5/30: 100%|████████████████████████████████████████████████████████████████████████████████████████| 2261/2261 [19:52<00:00,  1.90it/s]
# Epoch 5/30:
# Train Loss: 2.5116, Train Acc: 22.03%
# Val Loss: 2.6211, Val Acc: 13.81%
# Epoch 6/30: 100%|████████████████████████████████████████████████████████████████████████████████████████| 2261/2261 [19:50<00:00,  1.90it/s]
# Epoch 6/30:
# Train Loss: 2.4260, Train Acc: 24.14%
# Val Loss: 2.4956, Val Acc: 16.97%
# Epoch 7/30: 100%|████████████████████████████████████████████████████████████████████████████████████████| 2261/2261 [19:51<00:00,  1.90it/s]
# Epoch 7/30:
# Train Loss: 2.3684, Train Acc: 25.69%
# Val Loss: 2.6075, Val Acc: 13.77%
# Epoch 8/30: 100%|████████████████████████████████████████████████████████████████████████████████████████| 2261/2261 [19:51<00:00,  1.90it/s]
# Epoch 8/30:
# Train Loss: 2.3109, Train Acc: 27.29%
# Val Loss: 2.5559, Val Acc: 15.21%
# Epoch 9/30: 100%|████████████████████████████████████████████████████████████████████████████████████████| 2261/2261 [19:42<00:00,  1.91it/s]
# Epoch 9/30:
# Train Loss: 2.2583, Train Acc: 28.48%
# Val Loss: 2.5873, Val Acc: 15.29%
# Epoch 10/30: 100%|███████████████████████████████████████████████████████████████████████████████████████| 2261/2261 [19:37<00:00,  1.92it/s]
# Epoch 10/30:
# Train Loss: 2.2001, Train Acc: 30.23%
# Val Loss: 2.5222, Val Acc: 15.64%
# Epoch 11/30: 100%|███████████████████████████████████████████████████████████████████████████████████████| 2261/2261 [19:48<00:00,  1.90it/s]
# Epoch 11/30:
# Train Loss: 2.1808, Train Acc: 30.56%
# Val Loss: 2.4426, Val Acc: 17.07%
# Epoch 12/30: 100%|███████████████████████████████████████████████████████████████████████████████████████| 2261/2261 [19:44<00:00,  1.91it/s]
# Epoch 12/30:
# Train Loss: 2.1714, Train Acc: 30.83%
# Val Loss: 2.5129, Val Acc: 16.12%
# Epoch 13/30: 100%|███████████████████████████████████████████████████████████████████████████████████████| 2261/2261 [19:49<00:00,  1.90it/s]
# Epoch 13/30:
# Train Loss: 2.1587, Train Acc: 31.07%
# Val Loss: 2.4440, Val Acc: 17.50%
# Epoch 14/30: 100%|███████████████████████████████████████████████████████████████████████████████████████| 2261/2261 [19:46<00:00,  1.91it/s]
# Epoch 14/30:
# Train Loss: 2.1575, Train Acc: 31.30%
# Val Loss: 2.4524, Val Acc: 17.43%
# Epoch 15/30: 100%|███████████████████████████████████████████████████████████████████████████████████████| 2261/2261 [19:47<00:00,  1.90it/s]
# Epoch 15/30:
# Train Loss: 2.1408, Train Acc: 31.63%
# Val Loss: 2.4821, Val Acc: 16.72%
# Epoch 16/30: 100%|███████████████████████████████████████████████████████████████████████████████████████| 2261/2261 [19:47<00:00,  1.90it/s]
# Epoch 16/30:
# Train Loss: 2.1411, Train Acc: 31.60%
# Val Loss: 2.5036, Val Acc: 16.47%
# Epoch 17/30: 100%|███████████████████████████████████████████████████████████████████████████████████████| 2261/2261 [19:50<00:00,  1.90it/s]
# Epoch 17/30:
# Train Loss: 2.1447, Train Acc: 31.58%
# Val Loss: 2.4913, Val Acc: 16.60%
# Epoch 18/30: 100%|███████████████████████████████████████████████████████████████████████████████████████| 2261/2261 [19:45<00:00,  1.91it/s]
# Epoch 18/30:
# Train Loss: 2.1385, Train Acc: 31.75%
# Val Loss: 2.4798, Val Acc: 16.31%

# Early stopping triggered after 18 epochs

# Classification Report:
#               precision    recall  f1-score   support

#           01       0.36      0.16      0.22      2106
#           02       0.54      0.14      0.22      1546
#           03       0.43      0.18      0.25      1345
#           04       0.04      0.35      0.08       113
#           05       0.35      0.38      0.37      1044
#           06       0.06      0.27      0.10       123
#           07       0.09      0.35      0.14       268
#           08       0.06      0.31      0.10        71
#           09       0.12      0.12      0.12       165
#           10       0.01      0.05      0.02        38
#           11       0.03      0.18      0.05        78
#           12       0.01      0.06      0.01        16
#           13       0.01      0.11      0.02        35
#           14       0.07      0.09      0.08       172
#           15       0.12      0.25      0.16       102
#           16       0.02      0.31      0.03        87
#           17       0.05      0.28      0.08       123
#           18       0.07      0.14      0.09       177
#           19       0.39      0.42      0.40       176
#           20       0.03      0.28      0.06       115
#           21       0.04      0.22      0.06       116
#           22       0.01      0.10      0.02        21
#           23       0.02      0.29      0.04        31
#           24       0.12      0.07      0.09       854
#           25       0.09      0.34      0.14        89
#           26       0.76      0.12      0.20      5911
#           27       0.02      0.33      0.04        15
#           28       0.77      0.38      0.51       536
#           29       0.01      0.05      0.01        21
#           []       0.00      0.00      0.00         5

#     accuracy                           0.17     15499
#    macro avg       0.16      0.21      0.12     15499
# weighted avg       0.50      0.17      0.21     15499


# Model and label encoder for target_h1 saved successfully