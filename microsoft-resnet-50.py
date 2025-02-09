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
    hierarchical_levels = ['target_h2']
    
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
# Checking images: 100%|███████████████████████████████████████████████████████████████████████| 158511/158511 [01:16<00:00, 2071.73it/s]

# Total images in CSV: 158511
# Missing images: 55178
# Corrupted images: 12
# Valid images: 103321

# Warning: Some images are missing or corrupted!
# Image issues list saved to 'missing_images.txt'
# Valid dataset saved to 'pretrain_fill_valid.csv'

# ==================================================
# Training model for target_h2
# ==================================================
# Some weights of ResNetForImageClassification were not initialized from the model checkpoint at microsoft/resnet-50 and are newly initialized because the shapes did not match:
# - classifier.1.bias: found shape torch.Size([1000]) in the checkpoint and torch.Size([152]) in the model instantiated
# - classifier.1.weight: found shape torch.Size([1000, 2048]) in the checkpoint and torch.Size([152, 2048]) in the model instantiated
# You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
# /home/pham_dinh_vu/code/.venv/lib/python3.10/site-packages/torch/optim/lr_scheduler.py:62: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
#   warnings.warn(
# Epoch 1/30: 100%|██████████████████████████████████████████████████████████████████████████████████| 2261/2261 [19:18<00:00,  1.95it/s]
# Epoch 1/30:
# Train Loss: 4.9102, Train Acc: 2.36%
# Val Loss: 4.8806, Val Acc: 0.62%
# Epoch 2/30: 100%|██████████████████████████████████████████████████████████████████████████████████| 2261/2261 [19:13<00:00,  1.96it/s]
# Epoch 2/30:
# Train Loss: 4.6697, Train Acc: 7.12%
# Val Loss: 4.8174, Val Acc: 0.82%
# Epoch 3/30: 100%|██████████████████████████████████████████████████████████████████████████████████| 2261/2261 [19:12<00:00,  1.96it/s]
# Epoch 3/30:
# Train Loss: 4.3671, Train Acc: 10.50%
# Val Loss: 4.7676, Val Acc: 1.60%
# Epoch 4/30: 100%|██████████████████████████████████████████████████████████████████████████████████| 2261/2261 [19:09<00:00,  1.97it/s]
# Epoch 4/30:
# Train Loss: 4.1494, Train Acc: 12.93%
# Val Loss: 4.7857, Val Acc: 2.27%
# Epoch 5/30: 100%|██████████████████████████████████████████████████████████████████████████████████| 2261/2261 [19:10<00:00,  1.97it/s]
# Epoch 5/30:
# Train Loss: 3.9642, Train Acc: 15.80%
# Val Loss: 4.7371, Val Acc: 2.94%
# Epoch 6/30: 100%|██████████████████████████████████████████████████████████████████████████████████| 2261/2261 [19:09<00:00,  1.97it/s]
# Epoch 6/30:
# Train Loss: 3.8239, Train Acc: 17.66%
# Val Loss: 4.8123, Val Acc: 3.47%
# Epoch 7/30: 100%|██████████████████████████████████████████████████████████████████████████████████| 2261/2261 [19:05<00:00,  1.97it/s]
# Epoch 7/30:
# Train Loss: 3.6949, Train Acc: 19.64%
# Val Loss: 4.7844, Val Acc: 4.15%
# Epoch 8/30: 100%|██████████████████████████████████████████████████████████████████████████████████| 2261/2261 [19:13<00:00,  1.96it/s]
# Epoch 8/30:
# Train Loss: 3.5892, Train Acc: 21.34%
# Val Loss: 4.8136, Val Acc: 4.53%
# Epoch 9/30: 100%|██████████████████████████████████████████████████████████████████████████████████| 2261/2261 [19:08<00:00,  1.97it/s]
# Epoch 9/30:
# Train Loss: 3.5005, Train Acc: 22.74%
# Val Loss: 4.8780, Val Acc: 4.61%
# Epoch 10/30: 100%|█████████████████████████████████████████████████████████████████████████████████| 2261/2261 [19:09<00:00,  1.97it/s]
# Epoch 10/30:
# Train Loss: 3.4301, Train Acc: 23.88%
# Val Loss: 4.7848, Val Acc: 5.39%
# Epoch 11/30: 100%|█████████████████████████████████████████████████████████████████████████████████| 2261/2261 [19:12<00:00,  1.96it/s]
# Epoch 11/30:
# Train Loss: 3.3421, Train Acc: 25.35%
# Val Loss: 4.8583, Val Acc: 5.63%
# Epoch 12/30: 100%|█████████████████████████████████████████████████████████████████████████████████| 2261/2261 [19:04<00:00,  1.97it/s]
# Epoch 12/30:
# Train Loss: 3.2778, Train Acc: 26.65%
# Val Loss: 4.8358, Val Acc: 5.59%
# Epoch 13/30:  46%|█████████████████████████████████████                                            | 1033/2261 [08:45<10:02,  2.04it/s]Epoch 13/30: 100%|█████████████████████████████████████████████████████████████████████████████████| 2261/2261 [19:10<00:00,  1.96it/s]
# Epoch 13/30:
# Train Loss: 3.1985, Train Acc: 27.99%
# Val Loss: 4.9352, Val Acc: 5.76%
# Epoch 14/30: 100%|█████████████████████████████████████████████████████████████████████████████████| 2261/2261 [19:09<00:00,  1.97it/s]
# Epoch 14/30:
# Train Loss: 3.1513, Train Acc: 28.46%
# Val Loss: 4.9462, Val Acc: 6.11%
# Epoch 15/30: 100%|█████████████████████████████████████████████████████████████████████████████████| 2261/2261 [19:08<00:00,  1.97it/s]
# Epoch 15/30:
# Train Loss: 3.0679, Train Acc: 30.45%
# Val Loss: 4.9407, Val Acc: 6.14%
# Epoch 16/30: 100%|█████████████████████████████████████████████████████████████████████████████████| 2261/2261 [19:01<00:00,  1.98it/s]
# Epoch 16/30:
# Train Loss: 3.0366, Train Acc: 30.72%
# Val Loss: 4.8761, Val Acc: 6.30%
# Epoch 17/30: 100%|█████████████████████████████████████████████████████████████████████████████████| 2261/2261 [19:09<00:00,  1.97it/s]
# Epoch 17/30:
# Train Loss: 2.9853, Train Acc: 31.93%
# Val Loss: 5.0207, Val Acc: 6.39%
# Epoch 18/30: 100%|█████████████████████████████████████████████████████████████████████████████████| 2261/2261 [19:05<00:00,  1.97it/s]
# Epoch 18/30:
# Train Loss: 2.9454, Train Acc: 32.67%
# Val Loss: 4.9936, Val Acc: 6.50%
# Epoch 19/30: 100%|█████████████████████████████████████████████████████████████████████████████████| 2261/2261 [19:05<00:00,  1.97it/s]
# Epoch 19/30:
# Train Loss: 2.8840, Train Acc: 33.83%
# Val Loss: 4.8769, Val Acc: 6.71%
# Epoch 20/30: 100%|█████████████████████████████████████████████████████████████████████████████████| 2261/2261 [19:12<00:00,  1.96it/s]
# Epoch 20/30:
# Train Loss: 2.8391, Train Acc: 34.77%
# Val Loss: 4.9540, Val Acc: 6.61%
# Epoch 21/30: 100%|█████████████████████████████████████████████████████████████████████████████████| 2261/2261 [19:05<00:00,  1.97it/s]
# Epoch 21/30:
# Train Loss: 2.7957, Train Acc: 35.34%
# Val Loss: 5.0208, Val Acc: 6.85%
# Epoch 22/30: 100%|█████████████████████████████████████████████████████████████████████████████████| 2261/2261 [19:09<00:00,  1.97it/s]
# Epoch 22/30:
# Train Loss: 2.7441, Train Acc: 36.22%
# Val Loss: 4.9833, Val Acc: 6.90%
# Epoch 23/30: 100%|█████████████████████████████████████████████████████████████████████████████████| 2261/2261 [19:04<00:00,  1.98it/s]
# Epoch 23/30:
# Train Loss: 2.7171, Train Acc: 36.87%
# Val Loss: 5.0006, Val Acc: 7.21%
# Epoch 24/30: 100%|█████████████████████████████████████████████████████████████████████████████████| 2261/2261 [19:04<00:00,  1.98it/s]
# Epoch 24/30:
# Train Loss: 2.6806, Train Acc: 37.52%
# Val Loss: 5.0649, Val Acc: 6.99%
# Epoch 25/30: 100%|█████████████████████████████████████████████████████████████████████████████████| 2261/2261 [19:08<00:00,  1.97it/s]
# Epoch 25/30:
# Train Loss: 2.6351, Train Acc: 38.48%
# Val Loss: 5.0885, Val Acc: 7.11%
# Epoch 26/30: 100%|█████████████████████████████████████████████████████████████████████████████████| 2261/2261 [19:02<00:00,  1.98it/s]
# Epoch 26/30:
# Train Loss: 2.6082, Train Acc: 38.79%
# Val Loss: 5.0051, Val Acc: 7.36%
# Epoch 27/30: 100%|█████████████████████████████████████████████████████████████████████████████████| 2261/2261 [19:08<00:00,  1.97it/s]
# Epoch 27/30:
# Train Loss: 2.5671, Train Acc: 39.74%
# Val Loss: 5.0572, Val Acc: 7.47%
# Epoch 28/30: 100%|█████████████████████████████████████████████████████████████████████████████████| 2261/2261 [19:09<00:00,  1.97it/s]
# Epoch 28/30:
# Train Loss: 2.5423, Train Acc: 40.21%
# Val Loss: 5.0635, Val Acc: 7.57%
# Epoch 29/30: 100%|█████████████████████████████████████████████████████████████████████████████████| 2261/2261 [19:08<00:00,  1.97it/s]
# Epoch 29/30:
# Train Loss: 2.5049, Train Acc: 40.94%
# Val Loss: 5.0891, Val Acc: 7.56%
# Epoch 30/30: 100%|█████████████████████████████████████████████████████████████████████████████████| 2261/2261 [19:01<00:00,  1.98it/s]
# Epoch 30/30:
# Train Loss: 2.4778, Train Acc: 41.30%
# Val Loss: 5.0793, Val Acc: 7.38%

# Classification Report:
#               precision    recall  f1-score   support

#        01.01       0.42      0.19      0.26       609
#        01.03       0.03      0.42      0.06        12
#        01.05       0.43      0.28      0.34       206
#        01.07       0.45      0.40      0.42       250
#        01.09       0.10      0.23      0.14        44
#        01.11       0.02      0.11      0.03        45
#        01.15       0.10      0.00      0.00       889
#        01.17       0.15      0.24      0.18        51
#        02.01       0.61      0.02      0.05       690
#        02.03       0.35      0.33      0.34       183
#        02.05       0.22      0.31      0.26        86
#        02.07       0.01      0.17      0.02         6
#        02.09       0.01      0.11      0.02        19
#        02.11       0.50      0.00      0.01       562
#        03.01       0.38      0.08      0.13       309
#        03.03       0.08      0.21      0.12        43
#        03.05       0.09      0.17      0.12        59
#        03.07       0.33      0.19      0.24       105
#        03.09       0.05      0.14      0.07        37
#        03.11       0.02      0.21      0.04        14
#        03.13       0.07      0.07      0.07        76
#        03.15       0.28      0.10      0.15       308
#        03.17       0.12      0.32      0.18        84
#        03.19       0.15      0.12      0.14       121
#        03.21       0.14      0.23      0.17        75
#        03.23       0.20      0.28      0.23       111
#        03.25       0.17      0.67      0.27         3
#        04.01       0.05      0.07      0.06        56
#        04.03       0.00      0.00      0.00         4
#        04.05       0.08      0.23      0.12        47
#        04.07       0.00      0.00      0.00         2
#        04.09       0.01      0.50      0.01         4
#        05.01       0.29      0.18      0.22       171
#        05.03       0.27      0.13      0.18       401
#        05.05       0.26      0.24      0.25       221
#        05.07       0.07      0.32      0.11        60
#        05.09       0.16      0.18      0.17        68
#        05.11       0.03      0.14      0.05        29
#        05.13       0.06      0.30      0.11        44
#        05.15       0.14      0.28      0.19        50
#        06.01       0.13      0.20      0.16        50
#        06.03       0.04      0.20      0.07        44
#        06.07       0.02      0.19      0.03        16
#        06.09       0.00      0.00      0.00        13
#        07.01       0.10      0.24      0.14        62
#        07.03       0.05      0.10      0.06        49
#        07.05       0.01      0.12      0.03         8
#        07.07       0.02      0.03      0.03        36
#        07.09       0.02      0.21      0.04        42
#        07.11       0.03      0.10      0.05        51
#        07.13       0.04      0.27      0.07        11
#        07.15       0.02      0.22      0.03         9
#        08.01       0.03      0.10      0.04        30
#        08.03       0.07      0.21      0.11        14
#        08.05       0.00      0.00      0.00         3
#        08.07       0.00      0.00      0.00         2
#        08.09       0.02      0.17      0.04        12
#        08.11       0.00      0.00      0.00         2
#        08.13       0.01      0.12      0.01         8
#        09.01       0.02      0.02      0.02        66
#        09.03       0.27      0.39      0.32        31
#        09.05       0.06      0.08      0.07        25
#        09.07       0.02      0.08      0.03        25
#        09.09       0.02      0.11      0.04        18
#        10.01       0.00      0.00      0.00         4
#        10.03       0.03      0.21      0.05        14
#        10.05       0.00      0.00      0.00         8
#        10.07       0.00      0.00      0.00         5
#        10.09       0.01      0.43      0.02         7
#        11.01       0.05      0.18      0.08        39
#        11.03       0.01      0.03      0.02        29
#        11.05       0.00      0.00      0.00         6
#        11.07       0.00      0.00      0.00         4
#        11.09       0.01      0.08      0.02        13
#        12.01       0.00      0.00      0.00         3
#        12.03       0.00      0.00      0.00        26
#        13.01       0.00      0.00      0.00         9
#        13.03       0.02      0.04      0.03        52
#        14.01       0.02      0.02      0.02        48
#        14.03       0.05      0.06      0.05        33
#        14.05       0.00      0.00      0.00         4
#        14.07       0.00      0.00      0.00         1
#        14.09       0.00      0.00      0.00        34
#        14.11       0.00      0.00      0.00        22
#        15.01       0.00      0.00      0.00         1
#        15.03       0.00      0.00      0.00        12
#        15.05       0.00      0.00      0.00        49
#        15.07       0.00      0.00      0.00        18
#        15.09       0.00      0.00      0.00        42
#        16.01       0.00      0.02      0.01        45
#        16.03       0.01      0.06      0.01        16
#        17.01       0.00      0.00      0.00        54
#        17.03       0.00      0.00      0.00         5
#        17.05       0.02      0.02      0.02        48
#        17.07       0.00      0.07      0.01        15
#        18.01       0.00      0.00      0.00        49
#        18.03       0.00      0.00      0.00        27
#        18.05       0.01      0.04      0.02        27
#        18.07       0.01      0.02      0.01        52
#        18.09       0.00      0.00      0.00         4
#        18.11       0.00      0.00      0.00         3
#        18.13       0.00      0.00      0.00        15
#        18.15       0.00      0.00      0.00        14
#        19.01       0.01      0.02      0.01        45
#        19.05       0.03      0.01      0.02        91
#        19.07       0.02      0.25      0.03         4
#        19.09       0.00      0.00      0.00         7
#        19.11       0.00      0.00      0.00        13
#        19.13       0.00      0.00      0.00        86
#        20.01       0.00      0.00      0.00        16
#        20.03       0.00      0.00      0.00        39
#        20.05       0.00      0.00      0.00        77
#        21.01       0.00      0.00      0.00        16
#        21.03       0.00      0.00      0.00         2
#        22.01       0.00      0.00      0.00         3
#        22.03       0.00      0.00      0.00        20
#        22.05       0.00      0.00      0.00        10
#        23.01       0.00      0.00      0.00         1
#        23.03       0.00      0.00      0.00       103
#        23.05       0.00      0.00      0.00         7
#        24.01       0.00      0.00      0.00         7
#        24.03       0.00      0.00      0.00        12
#        24.05       0.01      0.01      0.01        75
#        24.07       0.00      0.00      0.00        83
#        24.09       0.01      0.01      0.01        76
#        24.11       0.00      0.00      0.00       247
#        24.13       0.03      0.03      0.03       232
#        24.15       0.00      0.00      0.00        12
#        24.17       0.00      0.00      0.00        35
#        24.19       0.00      0.00      0.00        54
#        24.21       0.29      0.00      0.00      2019
#        25.01       0.03      0.01      0.01       451
#        25.03       0.00      0.00      0.00       403
#        26.01       0.00      0.01      0.00       160
#        26.03       0.02      0.01      0.01       524
#        26.05       0.04      0.00      0.01       886
#        26.07       0.06      0.05      0.05       190
#        26.09       0.01      0.02      0.01       127
#        26.11       0.11      0.01      0.02      1089
#        26.13       0.04      0.02      0.02        62
#        26.15       0.00      0.00      0.00         3
#        26.17       0.00      0.00      0.00         5
#        26.19       0.00      0.00      0.00         7
#        27.01       0.26      0.01      0.02       536
#        27.03       0.01      0.05      0.02        20
#        27.05       0.00      0.00      0.00         1
#        28.01       0.00      0.00      0.00         5
#        29.01       0.00      0.00      0.00         0
#        29.03       0.00      0.00      0.00         0
#        29.04       0.00      0.00      0.00         0
#           []       0.00      0.00      0.00         0

#     accuracy                           0.06     15499
#    macro avg       0.06      0.09      0.05     15499
# weighted avg       0.19      0.06      0.07     15499


# Model and label encoder for target_h2 saved successfully