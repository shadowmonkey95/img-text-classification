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
    hierarchical_levels = ['target_h3']
    
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
    
#     01.01.04       0.00      0.00      0.00        38
#     01.01.05       0.00      0.00      0.00        85
#     01.01.06       0.00      0.00      0.00        10
#     01.01.07       0.00      0.00      0.00         9
#     01.01.08       0.00      0.00      0.00         1
#     01.01.09       0.00      0.00      0.00        45
#     01.01.10       0.00      0.00      0.00       122
#     01.01.11       0.00      0.00      0.00         7
#     01.01.12       0.00      0.00      0.00         5
#     01.01.13       0.00      0.00      0.00        31
#     01.01.14       0.00      0.00      0.00         3
#     01.03.01       0.00      0.00      0.00         1
#     01.03.02       0.00      0.00      0.00         3
#     01.03.03       0.00      0.25      0.00         8
#     01.03.04       0.00      0.00      0.00        90
#     01.05.01       0.00      0.00      0.00         8
#     01.05.03       0.00      0.00      0.00        67
#     01.05.04       0.00      0.00      0.00        41
#     01.05.25       0.00      0.00      0.00       127
#     01.07.01       0.00      0.00      0.00        61
#     01.07.02       0.00      0.00      0.00         5
#     01.07.04       0.00      0.00      0.00         1
#     01.07.05       0.00      0.00      0.00         1
#     01.07.06       0.00      0.00      0.00        14
#     01.07.07       0.00      0.00      0.00         7
#     01.07.08       0.00      0.00      0.00        34
#     01.07.25       0.00      0.00      0.00         3
#     01.09.01       0.00      0.00      0.00         3
#     01.09.03       0.00      0.00      0.00        32
#     01.09.05       0.00      0.00      0.00         6
#     01.09.25       0.00      0.00      0.00         5
#     01.11.01       0.00      0.00      0.00        37
#     01.11.02       0.00      0.00      0.00         2
#     01.11.03       0.00      0.00      0.00         1
#     01.11.25       0.00      0.00      0.00         5
#     01.15.01       0.00      0.00      0.00        60
#     01.15.02       0.00      0.00      0.00        67
#     01.15.03       0.00      0.00      0.00        15
#     01.15.04       0.00      0.00      0.00        46
#     01.15.05       0.00      0.00      0.00        64
#     01.15.06       0.00      0.00      0.00         1
#     01.15.07       0.00      0.00      0.00       142
#     01.15.08       0.00      0.00      0.00        22
#     01.15.09       0.00      0.00      0.00        27
#     01.15.10       0.00      0.00      0.00        12
#     01.15.11       0.00      0.00      0.00        27
#     01.15.12       0.00      0.00      0.00         2
#     01.15.13       0.00      0.00      0.00        45
#     01.15.14       0.00      0.00      0.00        68
#     01.15.15       0.00      0.01      0.01        88
#     01.15.17       0.00      0.00      0.00        56
#     01.15.18       0.00      0.00      0.00       142
#     01.15.24       0.00      0.00      0.00         1
#     01.15.25       0.00      0.00      0.00         3
#     01.17.01       0.00      0.00      0.00         3
#     01.17.02       0.00      0.00      0.00         4
#     01.17.03       0.00      0.00      0.00         9
#     01.17.04       0.00      0.00      0.00        14
#     01.17.05       0.00      0.00      0.00         2
#     01.17.06       0.00      0.00      0.00         1
#     01.17.08       0.00      0.00      0.00         8
#     01.17.09       0.00      0.00      0.00         6
#     01.17.11       0.00      0.00      0.00       102
#     01.17.12       0.00      0.00      0.00       164
#     01.17.13       0.00      0.00      0.00         5
#     01.17.14       0.00      0.00      0.00        10
#     01.17.25       0.00      0.00      0.00         4
#     02.01.01       0.00      0.00      0.00         4
#     02.01.02       0.00      0.00      0.00         5
#     02.01.03       0.00      0.00      0.00         2
#     02.01.04       0.00      0.00      0.00         4
#     02.01.05       0.00      0.00      0.00         1
#     02.01.06       0.00      0.00      0.00         6
#     02.01.07       0.00      0.00      0.00         6
#     02.01.08       0.00      0.00      0.00         5
#     02.01.09       0.00      0.00      0.00         3
#     02.01.10       0.00      0.00      0.00         1
#     02.01.11       0.00      0.00      0.00         2
#     02.01.12       0.00      0.00      0.00        11
#     02.01.13       0.00      0.00      0.00         7
#     02.01.14       0.00      0.00      0.00        17
#     02.01.15       0.00      0.00      0.00         9
#     02.01.16       0.00      0.00      0.00         5
#     02.01.17       0.00      0.00      0.00         3
#     02.01.18       0.00      0.00      0.00        10
#     02.01.19       0.00      0.00      0.00         4
#     02.01.20       0.00      0.00      0.00         3
#     02.01.21       0.00      0.00      0.00        13
#     02.01.22       0.00      0.00      0.00         1
#     02.01.23       0.00      0.00      0.00         1
#     02.01.24       0.00      0.00      0.00         4
#     02.01.25       0.00      0.00      0.00         4
#     02.01.26       0.00      0.00      0.00        39
#     02.01.27       0.01      0.50      0.02         8
#     02.01.28       0.00      0.00      0.00       150
#     02.01.29       0.00      0.00      0.00        44
#     02.01.30       0.00      0.00      0.00        29
#     02.01.31       0.00      0.00      0.00         2
#     02.01.32       0.00      0.00      0.00         2
#     02.01.33       0.00      0.00      0.00        32
#     02.01.34       0.00      0.00      0.00        53
#     02.01.35       0.00      0.00      0.00         3
#     02.01.37       0.00      0.00      0.00         3
#     02.01.38       0.00      0.00      0.00         2
#     02.01.39       0.00      0.00      0.00         2
#     02.03.01       0.00      0.00      0.00         2
#     02.03.02       0.00      0.00      0.00         2
#     02.03.03       0.00      0.00      0.00         3
#     02.03.04       0.00      0.00      0.00         5
#     02.03.05       0.00      0.00      0.00         1
#     02.03.06       0.00      0.00      0.00         7
#     02.03.07       0.00      0.00      0.00         5
#     02.03.08       0.00      0.00      0.00         1
#     02.03.09       0.00      0.00      0.00         7
#     02.03.10       0.00      0.00      0.00         2
#     02.03.11       0.00      0.00      0.00        18
#     02.03.13       0.00      0.00      0.00        19
#     02.03.14       0.02      0.20      0.03         5
#     02.03.15       0.00      0.00      0.00         6
#     02.03.16       0.00      0.00      0.00         2
#     02.03.17       0.00      0.00      0.00         3
#     02.03.18       0.00      0.00      0.00        18
#     02.03.19       0.00      0.00      0.00        13
#     02.03.21       0.00      0.00      0.00        21
#     02.03.22       0.00      0.00      0.00        19
#     02.03.24       0.00      0.00      0.00         8
#     02.03.25       0.00      0.00      0.00         2
#     02.03.26       0.00      0.00      0.00         3
#     02.03.27       0.00      0.00      0.00         1
#     02.03.28       0.00      0.00      0.00         1
#     02.05.01       0.00      0.00      0.00         1
#     02.05.02       0.00      0.00      0.00         1
#     02.05.04       0.00      0.00      0.00         4
#     02.05.05       0.00      0.00      0.00         1
#     02.05.06       0.00      0.00      0.00         2
#     02.05.24       0.00      0.00      0.00         3
#     02.05.26       0.00      0.00      0.00         1
#     02.07.01       0.00      0.00      0.00         2
#     02.07.02       0.00      0.00      0.00         1
#     02.07.03       0.00      0.00      0.00         1
#     02.07.04       0.00      0.00      0.00         1
#     02.07.05       0.00      0.00      0.00         1
#     02.07.25       0.00      0.00      0.00         5
#     02.07.26       0.00      0.00      0.00         1
#     02.09.01       0.00      0.00      0.00       189
#     02.09.02       0.00      0.00      0.00        79
#     02.09.03       0.00      0.10      0.01        10
#     02.09.04       0.00      0.00      0.00         7
#     02.09.05       0.00      0.00      0.00        16
#     02.09.09       0.00      0.00      0.00        72
#     02.09.11       0.00      0.00      0.00        25
#     02.09.12       0.00      0.00      0.00         3
#     02.09.14       0.01      0.04      0.01        24
#     02.09.15       0.00      0.00      0.00        13
#     02.09.17       0.00      0.00      0.00         1
#     02.09.19       0.00      0.00      0.00        32
#     02.11.01       0.00      0.00      0.00        14
#     02.11.02       0.00      0.00      0.00        13
#     02.11.03       0.00      0.00      0.00        50
#     02.11.04       0.00      0.00      0.00        14
#     02.11.05       0.00      0.00      0.00        35
#     02.11.06       0.00      0.00      0.00        51
#     02.11.07       0.00      0.00      0.00        34
#     02.11.08       0.00      0.00      0.00        30
#     02.11.09       0.00      0.00      0.00        16
#     02.11.10       0.00      0.00      0.00        61
#     02.11.11       0.00      0.00      0.00        13
#     02.11.12       0.00      0.00      0.00        11
#     02.11.13       0.00      0.00      0.00        10
#     02.11.14       0.00      0.00      0.00        30
#     02.11.15       0.00      0.00      0.00         1
#     02.11.16       0.00      0.00      0.00         1
#     02.11.25       0.00      0.00      0.00         4
#     03.01.01       0.00      0.00      0.00         1
#     03.01.02       0.00      0.00      0.00         4
#     03.01.03       0.00      0.00      0.00         7
#     03.01.04       0.00      0.00      0.00        30
#     03.01.07       0.00      0.00      0.00        10
#     03.01.08       0.00      0.00      0.00         1
#     03.01.09       0.00      0.00      0.00         2
#     03.01.11       0.00      0.00      0.00        48
#     03.01.13       0.00      0.00      0.00         4
#     03.01.14       0.00      0.00      0.00         3
#     03.01.18       0.00      0.00      0.00         4
#     03.01.19       0.00      0.00      0.00        32
#     03.01.20       0.00      0.00      0.00         3
#     03.01.21       0.00      0.00      0.00         3
#     03.01.24       0.00      0.00      0.00        17
#     03.01.25       0.00      0.00      0.00         3
#     03.01.26       0.00      0.00      0.00        18
#     03.03.01       0.00      0.00      0.00        27
#     03.03.03       0.00      0.00      0.00         1
#     03.03.05       0.00      0.00      0.00         1
#     03.03.07       0.00      0.00      0.00        14
#     03.03.09       0.00      0.00      0.00         7
#     03.03.16       0.00      0.00      0.00         2
#     03.03.24       0.00      0.00      0.00         2
#     03.03.26       0.00      0.00      0.00        10
#     03.05.01       0.00      0.00      0.00         1
#     03.05.02       0.00      0.00      0.00         1
#     03.05.03       0.00      0.00      0.00        13
#     03.05.16       0.00      0.00      0.00         1
#     03.05.24       0.00      0.00      0.00        37
#     03.07.01       0.00      0.00      0.00         9
#     03.07.03       0.00      0.00      0.00        17
#     03.07.05       0.00      0.00      0.00         2
#     03.07.07       0.00      0.00      0.00         5
#     03.07.08       0.00      0.00      0.00         6
#     03.07.09       0.00      0.00      0.00        34
#     03.07.10       0.00      0.00      0.00         4
#     03.07.11       0.00      0.00      0.00        34
#     03.07.21       0.00      0.00      0.00         2
#     03.07.22       0.00      0.00      0.00        30
#     03.07.24       0.00      0.00      0.00        18
#     03.09.01       0.00      0.00      0.00        19
#     03.09.02       0.00      0.00      0.00        10
#     03.09.06       0.00      0.00      0.00         7
#     03.09.07       0.00      0.00      0.00         4
#     03.09.09       0.00      0.00      0.00         1
#     03.09.24       0.00      0.00      0.00         6
#     03.09.26       0.00      0.00      0.00         9
#     03.11.01       0.00      0.00      0.00         4
#     03.11.16       0.00      0.00      0.00        86
#     03.11.24       0.02      0.03      0.03        30
#     03.13.01       0.00      0.00      0.00        10
#     03.13.02       0.00      0.00      0.00        65
#     03.13.03       0.00      0.00      0.00        13
#     03.13.04       0.00      0.00      0.00         1
#     03.13.05       0.00      0.00      0.00         1
#     03.13.25       0.00      0.00      0.00         1
#     03.15.01       0.00      0.00      0.00         3
#     03.15.02       0.00      0.00      0.00        12
#     03.15.03       0.00      0.00      0.00         6
#     03.15.05       0.00      0.00      0.00         3
#     03.15.06       0.00      0.00      0.00         1
#     03.15.07       0.00      0.00      0.00        12
#     03.15.08       0.00      0.00      0.00         4
#     03.15.09       0.00      0.00      0.00         4
#     03.15.10       0.00      0.00      0.00         6
#     03.15.12       0.00      0.00      0.00         2
#     03.15.13       0.00      0.00      0.00         2
#     03.15.14       0.00      0.00      0.00         3
#     03.15.15       0.00      0.00      0.00         2
#     03.15.16       0.00      0.00      0.00         9
#     03.15.19       0.00      0.00      0.00        36
#     03.15.24       0.00      0.00      0.00        19
#     03.15.25       0.00      0.00      0.00        14
#     03.17.01       0.00      0.00      0.00        19
#     03.17.02       0.00      0.00      0.00         7
#     03.17.03       0.00      0.00      0.00        16
#     03.17.05       0.00      0.00      0.00         8
#     03.17.06       0.00      0.00      0.00         8
#     03.17.16       0.00      0.00      0.00         2
#     03.17.25       0.00      0.00      0.00         1
#     03.19.01       0.00      0.00      0.00        41
#     03.19.02       0.00      0.00      0.00         3
#     03.19.03       0.00      0.00      0.00         3
#     03.19.04       0.00      0.00      0.00        30
#     03.19.05       0.00      0.00      0.00         1
#     03.19.07       0.00      0.00      0.00         6
#     03.19.08       0.00      0.00      0.00         5
#     03.19.09       0.00      0.00      0.00         3
#     03.19.11       0.00      0.00      0.00         3
#     03.19.13       0.00      0.00      0.00         6
#     03.19.14       0.00      0.00      0.00         8
#     03.19.17       0.00      0.00      0.00         1
#     03.19.18       0.00      0.00      0.00         1
#     03.19.19       0.00      0.00      0.00         3
#     03.19.21       0.00      0.00      0.00         1
#     03.19.23       0.00      0.00      0.00        16
#     03.19.24       0.00      0.00      0.00         7
#     03.19.25       0.00      0.00      0.00         4
#     03.21.01       0.00      0.00      0.00         5
#     03.21.02       0.00      0.00      0.00         3
#     03.21.05       0.00      0.00      0.00        11
#     03.21.06       0.00      0.00      0.00         1
#     03.21.07       0.00      0.00      0.00         1
#     03.21.08       0.00      0.00      0.00         7
#     03.21.24       0.00      0.00      0.00         2
#     03.21.26       0.00      0.00      0.00         2
#     03.23.01       0.00      0.00      0.00        30
#     03.23.02       0.00      0.00      0.00         7
#     03.23.05       0.00      0.00      0.00         6
#     03.23.06       0.00      0.00      0.00         1
#     03.23.07       0.00      0.00      0.00         3
#     03.23.08       0.00      0.00      0.00         1
#     03.23.09       0.00      0.00      0.00         1
#     03.23.10       0.00      0.00      0.00         4
#     03.23.11       0.00      0.00      0.00        26
#     03.23.12       0.00      0.00      0.00        37
#     03.23.14       0.00      0.00      0.00        20
#     03.23.15       0.00      0.00      0.00        18
#     03.23.21       0.00      0.00      0.00        10
#     03.23.24       0.00      0.00      0.00         4
#     03.23.25       0.00      0.00      0.00         4
#     03.25.01       0.00      0.00      0.00         9
#     03.25.02       0.00      0.00      0.00        43
#     04.01.01       0.00      0.00      0.00         2
#     04.01.02       0.00      0.00      0.00         6
#     04.01.03       0.00      0.00      0.00        11
#     04.01.04       0.00      0.00      0.00         1
#     04.01.05       0.00      0.00      0.00         8
#     04.01.06       0.00      0.00      0.00         9
#     04.01.07       0.00      0.00      0.00       154
#     04.01.08       0.00      0.00      0.00         1
#     04.01.25       0.00      0.00      0.00        34
#     04.03.01       0.00      0.00      0.00       175
#     04.03.03       0.00      0.00      0.00         7
#     04.03.25       0.00      0.00      0.00        19
#     04.05.01       0.00      0.00      0.00        24
#     04.05.03       0.00      0.00      0.00         7
#     04.05.04       0.00      0.00      0.00         9
#     04.05.05       0.00      0.00      0.00       155
#     04.05.25       0.00      0.00      0.00         4
#     04.07.01       0.00      0.00      0.00        31
#     04.07.02       0.00      0.00      0.00        14
#     04.07.03       0.00      0.00      0.00         5
#     04.07.05       0.00      0.00      0.00         3
#     04.09.01       0.00      0.00      0.00         1
#     05.01.01       0.00      0.00      0.00         1
#     05.01.02       0.00      0.00      0.00         1
#     05.01.03       0.00      0.00      0.00        11
#     05.01.04       0.00      0.00      0.00        12
#     05.01.05       0.00      0.00      0.00         5
#     05.01.06       0.00      0.00      0.00         6
#     05.01.08       0.00      0.00      0.00         9
#     05.01.10       0.00      0.00      0.00         8
#     05.01.25       0.00      0.00      0.00         3
#     05.03.01       0.00      0.00      0.00         2
#     05.03.02       0.00      0.00      0.00         2
#     05.03.03       0.00      0.00      0.00         1
#     05.03.04       0.00      0.00      0.00         9
#     05.03.05       0.00      0.00      0.00         6
#     05.03.06       0.00      0.00      0.00         2
#     05.03.07       0.00      0.00      0.00         9
#     05.03.08       0.00      0.00      0.00         2
#     05.03.09       0.00      0.00      0.00         1
#     05.03.10       0.00      0.00      0.00         2
#     05.03.25       0.00      0.00      0.00         5
#     05.05.01       0.00      0.00      0.00         2
#     05.05.02       0.00      0.00      0.00         2
#     05.05.03       0.00      0.00      0.00         8
#     05.05.05       0.00      0.00      0.00         4
#     05.05.06       0.00      0.00      0.00         1
#     05.05.25       0.00      0.00      0.00         1
#     05.07.01       0.00      0.00      0.00         3
#     05.07.02       0.00      0.00      0.00         4
#     05.07.03       0.00      0.00      0.00        21
#     05.07.04       0.00      0.00      0.00        27
#     05.07.05       0.00      0.00      0.00         1
#     05.07.06       0.00      0.00      0.00        22
#     05.07.07       0.00      0.00      0.00         6
#     05.07.25       0.00      0.00      0.00        44
#     05.09.01       0.00      0.00      0.00         1
#     05.09.02       0.00      0.00      0.00        41
#     05.09.03       0.00      0.00      0.00         1
#     05.09.04       0.00      0.00      0.00         1
#     05.09.05       0.00      0.00      0.00        15
#     05.09.06       0.00      0.00      0.00         1
#     05.09.07       0.00      0.00      0.00         3
#     05.09.08       0.00      0.00      0.00         1
#     05.09.09       0.00      0.00      0.00         2
#     05.09.10       0.00      0.00      0.00         2
#     05.09.11       0.00      0.00      0.00         3
#     05.09.12       0.00      0.00      0.00         2
#     05.09.13       0.00      0.00      0.00        15
#     05.09.14       0.00      0.00      0.00         1
#     05.09.25       0.00      0.00      0.00        23
#     05.11.01       0.00      0.00      0.00        18
#     05.11.02       0.00      0.00      0.00         2
#     05.11.03       0.00      0.00      0.00         2
#     05.11.04       0.00      0.00      0.00         1
#     05.11.05       0.00      0.00      0.00         3
#     05.11.06       0.00      0.00      0.00         2
#     05.11.07       0.00      0.00      0.00         2
#     05.11.08       0.00      0.00      0.00         8
#     05.11.09       0.00      0.00      0.00         7
#     05.11.10       0.00      0.00      0.00         4
#     05.11.25       0.00      0.00      0.00         4
#     05.13.01       0.00      0.00      0.00         1
#     05.13.02       0.00      0.00      0.00        18
#     05.13.03       0.00      0.00      0.00         1
#     05.13.04       0.00      0.00      0.00         1
#     05.13.05       0.00      0.00      0.00         2
#     05.13.06       0.00      0.00      0.00         2
#     05.13.07       0.00      0.00      0.00         2
#     05.13.08       0.00      0.00      0.00        13
#     05.13.09       0.00      0.00      0.00         5
#     05.13.25       0.00      0.00      0.00        12
#     05.15.01       0.00      0.00      0.00         1
#     05.15.02       0.00      0.00      0.00         2
#     05.15.04       0.00      0.00      0.00         3
#     05.15.25       0.00      0.00      0.00         3
#     06.01.01       0.00      0.00      0.00         4
#     06.01.02       0.00      0.00      0.00        12
#     06.01.03       0.00      0.00      0.00        16
#     06.01.04       0.00      0.00      0.00         6
#     06.03.02       0.00      0.00      0.00         1
#     06.03.03       0.00      0.00      0.00         3
#     06.03.04       0.00      0.00      0.00         3
#     06.03.06       0.00      0.00      0.00         3
#     06.03.07       0.00      0.00      0.00        21
#     06.03.08       0.00      0.00      0.00        10
#     06.03.25       0.00      0.00      0.00         9
#     06.07.01       0.00      0.00      0.00         2
#     06.07.02       0.00      0.00      0.00         1
#     06.07.03       0.00      0.00      0.00         9
#     06.09.01       0.00      0.00      0.00         1
#     06.09.02       0.00      0.00      0.00         3
#     06.09.03       0.00      0.00      0.00         1
#     06.09.05       0.00      0.00      0.00         4
#     06.09.06       0.00      0.00      0.00         1
#     06.09.08       0.00      0.00      0.00         2
#     06.09.09       0.00      0.00      0.00         1
#     06.09.25       0.00      0.00      0.00         8
#     07.01.01       0.00      0.00      0.00         7
#     07.01.02       0.00      0.00      0.00         3
#     07.01.03       0.00      0.00      0.00         2
#     07.01.04       0.00      0.00      0.00         5
#     07.01.05       0.00      0.00      0.00         2
#     07.01.06       0.00      0.00      0.00        10
#     07.01.07       0.00      0.00      0.00         4
#     07.01.08       0.00      0.00      0.00         2
#     07.01.09       0.00      0.00      0.00         1
#     07.01.25       0.00      0.00      0.00         2
#     07.03.01       0.00      0.00      0.00         7
#     07.03.02       0.00      0.00      0.00         3
#     07.03.03       0.00      0.00      0.00         1
#     07.03.04       0.00      0.00      0.00         1
#     07.03.05       0.00      0.00      0.00         2
#     07.03.06       0.00      0.00      0.00         4
#     07.03.07       0.00      0.00      0.00         3
#     07.03.08       0.00      0.00      0.00         1
#     07.03.09       0.00      0.00      0.00         3
#     07.03.10       0.00      0.00      0.00         7
#     07.03.25       0.00      0.00      0.00         2
#     07.05.03       0.00      0.00      0.00        29
#     07.05.05       0.00      0.00      0.00         1
#     07.05.07       0.00      0.00      0.00         2
#     07.05.08       0.00      0.00      0.00         1
#     07.05.10       0.00      0.00      0.00         3
#     07.05.25       0.00      0.00      0.00        16
#     07.07.01       0.00      0.00      0.00         2
#     07.07.02       0.00      0.00      0.00         2
#     07.07.03       0.00      0.00      0.00         3
#     07.07.05       0.00      0.00      0.00         2
#     07.07.07       0.00      0.00      0.00         2
#     07.07.09       0.00      0.00      0.00         3
#     07.07.25       0.00      0.00      0.00         2
#     07.09.01       0.00      0.00      0.00         5
#     07.09.02       0.00      0.00      0.00         1
#     07.09.03       0.00      0.00      0.00        10
#     07.09.04       0.00      0.00      0.00         1
#     07.09.05       0.00      0.00      0.00         3
#     07.09.06       0.00      0.00      0.00         4
#     07.09.07       0.00      0.00      0.00        10
#     07.09.08       0.00      0.00      0.00         3
#     07.09.09       0.00      0.00      0.00         5
#     07.09.10       0.00      0.00      0.00         1
#     07.09.25       0.00      0.00      0.00         8
#     07.11.01       0.00      0.00      0.00         3
#     07.11.02       0.00      0.00      0.00         2
#     07.11.03       0.00      0.00      0.00         4
#     07.11.04       0.00      0.00      0.00         5
#     07.11.07       0.00      0.00      0.00         2
#     07.11.08       0.00      0.00      0.00         9
#     07.11.09       0.00      0.00      0.00         2
#     07.11.11       0.00      0.00      0.00         1
#     07.11.25       0.00      0.00      0.00         1
#     07.13.01       0.00      0.00      0.00         4
#     07.13.02       0.00      0.00      0.00         1
#     07.13.03       0.00      0.00      0.00         1
#     07.15.01       0.00      0.00      0.00         2
#     07.15.03       0.00      0.00      0.00         1
#     07.15.04       0.00      0.00      0.00         4
#     07.15.05       0.00      0.00      0.00         1
#     07.15.25       0.00      0.00      0.00         9
#     08.01.01       0.00      0.00      0.00         1
#     08.01.02       0.00      0.00      0.00         2
#     08.01.03       0.00      0.00      0.00         1
#     08.01.04       0.00      0.00      0.00         1
#     08.01.05       0.00      0.00      0.00         3
#     08.01.06       0.00      0.00      0.00         2
#     08.01.07       0.00      0.00      0.00         3
#     08.01.08       0.00      0.00      0.00         6
#     08.01.09       0.00      0.00      0.00         1
#     08.01.10       0.00      0.00      0.00        13
#     08.01.11       0.00      0.00      0.00         4
#     08.01.12       0.00      0.00      0.00         8
#     08.01.25       0.00      0.00      0.00         1
#     08.03.01       0.00      0.00      0.00         2
#     08.03.02       0.00      0.00      0.00         1
#     08.03.03       0.00      0.00      0.00         1
#     08.03.04       0.00      0.00      0.00         1
#     08.03.25       0.00      0.00      0.00         8
#     08.05.01       0.00      0.00      0.00         3
#     08.05.02       0.00      0.00      0.00         3
#     08.05.03       0.00      0.12      0.00         8
#     08.05.25       0.00      0.00      0.00         2
#     08.07.01       0.00      0.00      0.00         3
#     08.07.02       0.00      0.00      0.00         6
#     08.07.03       0.00      0.00      0.00         4
#     08.07.25       0.00      0.00      0.00         2
#     08.09.01       0.00      0.00      0.00         4
#     08.09.03       0.00      0.00      0.00         2
#     08.09.04       0.00      0.00      0.00         1
#     08.09.05       0.00      0.00      0.00         1
#     08.11.01       0.00      0.00      0.00         1
#     08.11.03       0.00      0.00      0.00         4
#     08.11.05       0.00      0.00      0.00         1
#     08.11.06       0.00      0.00      0.00         3
#     08.11.25       0.00      0.00      0.00         2
#     08.13.01       0.00      0.00      0.00         2
#     08.13.02       0.00      0.00      0.00         1
#     08.13.03       0.00      0.00      0.00         2
#     08.13.04       0.00      0.00      0.00         3
#     08.13.06       0.00      0.00      0.00         2
#     08.13.25       0.00      0.00      0.00         1
#     09.01.01       0.00      0.00      0.00         3
#     09.01.02       0.00      0.00      0.00         1
#     09.01.03       0.00      0.00      0.00        12
#     09.01.04       0.00      0.00      0.00         4
#     09.01.05       0.00      0.00      0.00         1
#     09.01.06       0.00      0.00      0.00         8
#     09.01.07       0.00      0.00      0.00        10
#     09.01.08       0.00      0.00      0.00        12
#     09.01.09       0.00      0.00      0.00         6
#     09.01.10       0.00      0.00      0.00        14
#     09.01.11       0.00      0.00      0.00         2
#     09.01.12       0.00      0.00      0.00         3
#     09.01.13       0.00      0.00      0.00         3
#     09.01.25       0.00      0.00      0.00         2
#     09.03.01       0.00      0.00      0.00         1
#     09.03.02       0.00      0.00      0.00        29
#     09.03.03       0.00      0.00      0.00         2
#     09.03.04       0.00      0.00      0.00         3
#     09.03.05       0.00      0.00      0.00         3
#     09.03.06       0.00      0.00      0.00         1
#     09.03.07       0.00      0.00      0.00         9
#     09.03.08       0.00      0.00      0.00         4
#     09.03.09       0.00      0.00      0.00         6
#     09.03.10       0.00      0.00      0.00         1
#     09.03.12       0.00      0.00      0.00         5
#     09.03.13       0.00      0.00      0.00         2
#     09.03.14       0.00      0.00      0.00         6
#     09.03.15       0.00      0.00      0.00         1
#     09.03.16       0.00      0.00      0.00         1
#     09.03.25       0.00      0.00      0.00         7
#     09.05.01       0.00      0.00      0.00         1
#     09.05.02       0.00      0.00      0.00         1
#     09.05.03       0.00      0.00      0.00         1
#     09.05.04       0.00      0.00      0.00         1
#     09.05.06       0.00      0.00      0.00         1
#     09.05.08       0.00      0.00      0.00        12
#     09.05.10       0.00      0.00      0.00         7
#     09.05.11       0.00      0.00      0.00         1
#     09.05.25       0.00      0.00      0.00        14
#     09.07.01       0.00      0.00      0.00         1
#     09.07.02       0.00      0.00      0.00         7
#     09.07.04       0.00      0.00      0.00         1
#     09.07.05       0.00      0.00      0.00         3
#     09.07.06       0.00      0.00      0.00         6
#     09.07.07       0.00      0.00      0.00         1
#     09.07.08       0.00      0.00      0.00         3
#     09.07.25       0.00      0.00      0.00         1
#     09.09.01       0.00      0.00      0.00         1
#     09.09.03       0.00      0.00      0.00         4
#     09.09.04       0.00      0.00      0.00         3
#     09.09.05       0.00      0.00      0.00         4
#     09.09.06       0.00      0.00      0.00        36
#     09.09.07       0.00      0.00      0.00         6
#     09.09.08       0.00      0.00      0.00         7
#     09.09.25       0.00      0.00      0.00         3
#     10.01.01       0.00      0.00      0.00         1
#     10.01.02       0.00      0.00      0.00         3
#     10.01.04       0.00      0.00      0.00        11
#     10.01.05       0.00      0.00      0.00         5
#     10.01.06       0.00      0.00      0.00         7
#     10.01.25       0.00      0.00      0.00         6
#     10.03.01       0.00      0.00      0.00         5
#     10.03.02       0.00      0.00      0.00         1
#     10.03.03       0.00      0.00      0.00         8
#     10.05.01       0.00      0.00      0.00         8
#     10.05.02       0.00      0.00      0.00         2
#     10.05.03       0.00      0.00      0.00         8
#     10.05.05       0.00      0.00      0.00         5
#     10.05.06       0.00      0.00      0.00         5
#     10.05.08       0.00      0.00      0.00         1
#     10.05.09       0.00      0.00      0.00         9
#     10.05.10       0.00      0.00      0.00         8
#     10.05.12       0.00      0.00      0.00         3
#     10.05.13       0.00      0.00      0.00         6
#     10.05.25       0.00      0.00      0.00         1
#     10.07.01       0.00      0.00      0.00        13
#     10.07.03       0.00      0.00      0.00         2
#     10.07.04       0.00      0.00      0.00        36
#     10.07.05       0.00      0.00      0.00         9
#     10.07.25       0.00      0.00      0.00         4
#     10.09.01       0.00      0.00      0.00         1
#     10.09.02       0.00      0.00      0.00         4
#     11.01.01       0.00      0.00      0.00         5
#     11.01.02       0.00      0.00      0.00         6
#     11.01.03       0.00      0.00      0.00         8
#     11.01.04       0.00      0.00      0.00        11
#     11.01.05       0.00      0.00      0.00         1
#     11.01.06       0.00      0.00      0.00        22
#     11.01.08       0.00      0.00      0.00         7
#     11.01.09       0.00      0.00      0.00         1
#     11.01.10       0.00      0.00      0.00         4
#     11.01.11       0.00      0.00      0.00         1
#     11.01.25       0.00      0.00      0.00         1
#     11.03.01       0.00      0.00      0.00         1
#     11.03.02       0.00      0.00      0.00        30
#     11.03.03       0.00      0.00      0.00         2
#     11.03.04       0.00      0.00      0.00         2
#     11.03.05       0.00      0.00      0.00         1
#     11.03.06       0.00      0.00      0.00         6
#     11.03.07       0.00      0.00      0.00         1
#     11.03.08       0.00      0.00      0.00         3
#     11.03.09       0.00      0.00      0.00         3
#     11.03.12       0.00      0.00      0.00         1
#     11.03.13       0.00      0.00      0.00         3
#     11.03.15       0.00      0.00      0.00         2
#     11.03.16       0.00      0.00      0.00         6
#     11.03.25       0.00      0.00      0.00         7
#     11.05.06       0.00      0.00      0.00         3
#     11.07.01       0.00      0.00      0.00         2
#     11.07.25       0.00      0.00      0.00         3
#     11.09.01       0.00      0.00      0.00         1
#     11.09.02       0.00      0.00      0.00        23
#     11.09.03       0.00      0.00      0.00         2
#     11.09.04       0.00      0.00      0.00         1
#     11.09.25       0.00      0.00      0.00         1
#     12.01.01       0.00      0.00      0.00         9
#     12.01.03       0.00      0.00      0.00         6
#     12.01.04       0.00      0.00      0.00         5
#     12.01.05       0.00      0.00      0.00         9
#     12.01.07       0.00      0.00      0.00         4
#     12.01.08       0.00      0.00      0.00         1
#     12.01.09       0.00      0.00      0.00        18
#     12.01.11       0.00      0.00      0.00         2
#     12.01.25       0.00      0.00      0.00         2
#     12.03.01       0.00      0.00      0.00         3
#     12.03.02       0.00      0.00      0.00         4
#     12.03.03       0.00      0.00      0.00         1
#     12.03.04       0.00      0.00      0.00         1
#     12.03.25       0.00      0.00      0.00         3
#     13.01.01       0.00      0.00      0.00         6
#     13.01.02       0.00      0.00      0.00         3
#     13.01.03       0.00      0.00      0.00         2
#     13.01.05       0.00      0.00      0.00         1
#     13.01.06       0.00      0.00      0.00         1
#     13.01.08       0.00      0.00      0.00         7
#     13.01.10       0.00      0.00      0.00         1
#     13.01.11       0.00      0.00      0.00         5
#     13.01.13       0.00      0.00      0.00         4
#     13.01.25       0.00      0.00      0.00         3
#     13.03.01       0.00      0.00      0.00         1
#     13.03.02       0.00      0.00      0.00         5
#     13.03.03       0.00      0.00      0.00         1
#     13.03.06       0.00      0.00      0.00         8
#     13.03.07       0.00      0.00      0.00         2
#     13.03.25       0.00      0.00      0.00         3
#     14.01.01       0.00      0.00      0.00         3
#     14.01.02       0.00      0.00      0.00         2
#     14.01.03       0.00      0.00      0.00         1
#     14.01.04       0.00      0.00      0.00         1
#     14.01.05       0.00      0.00      0.00         5
#     14.01.06       0.00      0.00      0.00         4
#     14.01.07       0.00      0.00      0.00        16
#     14.01.08       0.00      0.00      0.00        52
#     14.01.25       0.00      0.00      0.00         3
#     14.03.01       0.00      0.00      0.00         1
#     14.03.02       0.00      0.00      0.00         3
#     14.03.03       0.00      0.00      0.00        12
#     14.03.04       0.00      0.00      0.00         4
#     14.03.05       0.00      0.00      0.00         3
#     14.03.07       0.00      0.00      0.00         2
#     14.03.25       0.00      0.00      0.00         2
#     14.05.01       0.00      0.00      0.00         9
#     14.05.02       0.00      1.00      0.00         1
#     14.05.03       0.00      0.00      0.00         2
#     14.05.05       0.00      0.00      0.00         1
#     14.05.06       0.00      0.00      0.00         9
#     14.05.07       0.00      0.00      0.00         5
#     14.05.09       0.00      0.00      0.00         8
#     14.05.10       0.00      0.00      0.00         1
#     14.05.11       0.00      0.00      0.00        15
#     14.05.25       0.00      0.00      0.00        12
#     14.07.01       0.00      0.00      0.00         8
#     14.07.02       0.00      0.00      0.00         6
#     14.07.03       0.00      0.00      0.00        13
#     14.07.05       0.00      0.00      0.00         7
#     14.07.06       0.00      0.00      0.00         2
#     14.07.25       0.00      0.00      0.00         2
#     14.09.01       0.00      0.00      0.00         1
#     14.09.02       0.00      0.00      0.00         1
#     14.11.01       0.00      0.00      0.00        12
#     14.11.02       0.00      0.00      0.00         4
#     14.11.07       0.00      0.00      0.00         4
#     14.11.08       0.00      0.00      0.00         5
#     14.11.09       0.00      0.00      0.00         1
#     15.01.01       0.00      0.00      0.00         1
#     15.01.02       0.00      0.00      0.00         7
#     15.01.03       0.00      0.00      0.00         5
#     15.01.04       0.00      0.00      0.00         1
#     15.01.05       0.00      0.00      0.00         3
#     15.01.06       0.00      0.00      0.00         2
#     15.01.07       0.00      0.00      0.00         2
#     15.01.09       0.00      0.00      0.00         4
#     15.01.25       0.00      0.00      0.00        13
#     15.03.01       0.00      0.00      0.00         1
#     15.03.02       0.00      0.00      0.00         1
#     15.03.25       0.00      0.00      0.00         1
#     15.05.02       0.00      0.00      0.00         2
#     15.05.03       0.00      0.00      0.00         6
#     15.05.06       0.00      0.00      0.00         1
#     15.05.07       0.00      0.00      0.00         1
#     15.05.08       0.00      0.00      0.00        14
#     15.05.25       0.00      0.00      0.00         1
#     15.07.01       0.00      0.00      0.00         7
#     15.07.03       0.00      0.00      0.00         7
#     15.07.04       0.00      0.00      0.00         2
#     15.07.25       0.00      0.00      0.00         1
#     15.09.01       0.00      0.00      0.00         1
#     15.09.02       0.00      0.00      0.00        13
#     15.09.03       0.00      0.00      0.00         4
#     15.09.25       0.00      0.00      0.00         1
#     16.01.01       0.00      0.00      0.00         3
#     16.01.03       0.00      0.00      0.00         1
#     16.01.04       0.00      0.00      0.00         3
#     16.01.05       0.00      0.00      0.00         1
#     16.01.06       0.00      0.00      0.00         3
#     16.01.07       0.00      0.00      0.00         3
#     16.01.08       0.00      0.00      0.00         2
#     16.01.25       0.00      0.00      0.00         1
#     16.03.01       0.00      0.00      0.00         1
#     16.03.02       0.00      0.00      0.00         3
#     16.03.03       0.00      0.00      0.00        12
#     16.03.04       0.00      0.00      0.00         3
#     16.03.05       0.00      0.00      0.00         1
#     16.03.06       0.00      0.00      0.00         4
#     16.03.07       0.00      0.00      0.00         2
#     16.03.08       0.00      0.00      0.00         1
#     16.03.25       0.00      0.00      0.00         1
#     17.01.01       0.00      0.00      0.00         4
#     17.01.02       0.00      0.00      0.00         2
#     17.01.03       0.00      0.00      0.00         1
#     17.01.05       0.00      0.00      0.00         8
#     17.01.25       0.00      0.00      0.00        52
#     17.03.01       0.00      0.00      0.00        41
#     17.03.02       0.00      0.00      0.00         2
#     17.03.03       0.00      0.00      0.00         1
#     17.03.04       0.00      0.00      0.00         5
#     17.03.25       0.00      0.00      0.00         1
#     17.05.01       0.00      0.00      0.00         7
#     17.05.02       0.00      0.00      0.00         3
#     17.05.25       0.00      0.00      0.00         1
#     17.07.01       0.00      0.00      0.00         3
#     17.07.02       0.00      0.00      0.00         1
#     17.07.03       0.00      0.00      0.00         4
#     17.07.04       0.00      0.00      0.00         5
#     17.07.05       0.00      0.00      0.00         8
#     17.07.06       0.00      0.00      0.00         5
#     17.07.07       0.00      0.00      0.00         3
#     17.07.08       0.00      0.00      0.00        39
#     17.07.09       0.00      0.00      0.00         2
#     17.07.10       0.00      0.00      0.00        13
#     17.07.25       0.00      0.00      0.00        30
#     18.01.03       0.00      0.00      0.00        53
#     18.01.25       0.00      0.00      0.00        12
#     18.03.01       0.00      0.00      0.00        43
#     18.03.03       0.00      0.00      0.00        21
#     18.03.04       0.00      0.00      0.00        36
#     18.03.05       0.00      0.00      0.00        24
#     18.03.06       0.00      0.00      0.00        38
#     18.03.07       0.00      0.03      0.01        60
#     18.03.25       0.00      0.00      0.00        39
#     18.05.01       0.00      0.00      0.00        50
#     18.05.02       0.00      0.00      0.00         1
#     18.05.03       0.00      0.00      0.00         6
#     18.05.04       0.00      0.00      0.00         4
#     18.05.05       0.00      0.00      0.00         1
#     18.05.07       0.00      0.00      0.00         1
#     18.05.08       0.00      0.00      0.00        50
#     18.05.09       0.00      0.00      0.00         1
#     18.05.10       0.00      0.00      0.00         4
#     18.05.11       0.00      0.00      0.00        22
#     18.05.12       0.00      0.00      0.00        10
#     18.05.25       0.00      0.00      0.00        54
#     18.07.01       0.00      0.00      0.00        42
#     18.07.02       0.00      0.00      0.00         1
#     18.07.03       0.00      0.00      0.00         1
#     18.07.04       0.00      0.00      0.00         9
#     18.07.05       0.00      0.00      0.00         1
#     18.07.06       0.00      0.00      0.00         1
#     18.07.08       0.00      0.00      0.00         2
#     18.07.10       0.00      0.00      0.00        21
#     18.07.11       0.00      0.00      0.00         3
#     18.07.12       0.00      0.00      0.00         9
#     18.07.13       0.00      0.00      0.00         5
#     18.07.25       0.00      0.00      0.00        30
#     18.09.01       0.00      0.00      0.00         8
#     18.09.02       0.00      0.00      0.00         8
#     18.09.03       0.00      0.00      0.00         3
#     18.09.04       0.00      0.00      0.00         7
#     18.09.05       0.00      0.00      0.00         6
#     18.09.06       0.00      0.00      0.00        22
#     18.09.25       0.00      0.00      0.00       121
#     18.11.01       0.00      0.00      0.00       314
#     18.11.02       0.00      0.00      0.00       169
#     18.11.03       0.00      0.00      0.00        42
#     18.11.04       0.00      0.00      0.00        20
#     18.11.05       0.00      0.00      0.00        94
#     18.11.06       0.00      0.00      0.00        31
#     18.11.07       0.00      0.00      0.00        36
#     18.11.25       0.00      0.00      0.00        22
#     18.13.01       0.00      0.00      0.00        36
#     18.13.02       0.00      0.00      0.00        64
#     18.13.03       0.00      0.00      0.00        74
#     18.13.04       0.00      0.00      0.00        63
#     18.13.05       0.00      0.00      0.00        27
#     18.13.06       0.00      0.00      0.00        84
#     18.13.25       0.00      0.00      0.00        42
#     18.15.01       0.00      0.00      0.00        43
#     18.15.03       0.00      0.00      0.00       609
#     18.15.04       0.00      0.00      0.00        80
#     18.15.25       0.00      0.00      0.00         1
#     19.01.01       0.00      0.00      0.00        22
#     19.01.02       0.00      0.00      0.00        13
#     19.01.04       0.00      0.00      0.00         1
#     19.01.05       0.00      0.00      0.00        11
#     19.01.06       0.00      0.00      0.00        50
#     19.01.07       0.00      0.00      0.00        76
#     19.01.25       0.00      0.00      0.00        44
#     19.05.01       0.00      0.00      0.00        18
#     19.05.02       0.00      0.00      0.00         5
#     19.05.03       0.00      0.00      0.00         2
#     19.05.04       0.00      0.00      0.00         3
#     19.05.06       0.00      0.00      0.00         2
#     19.05.09       0.00      0.00      0.00         7
#     19.05.25       0.00      0.00      0.00        10
#     19.07.01       0.00      0.00      0.00        21
#     19.07.02       0.00      0.00      0.00        30
#     19.07.03       0.00      0.00      0.00         1
#     19.07.04       0.00      0.00      0.00        46
#     19.07.05       0.00      0.00      0.00       116
#     19.07.06       0.00      0.00      0.00        20
#     19.07.08       0.00      0.00      0.00         4
#     19.07.09       0.00      0.00      0.00        44
#     19.07.10       0.00      0.00      0.00        31
#     19.07.11       0.00      0.00      0.00         5
#     19.07.13       0.00      0.00      0.00         1
#     19.07.14       0.00      0.00      0.00         4
#     19.07.15       0.00      0.00      0.00        10
#     19.07.16       0.00      0.00      0.00        16
#     19.07.17       0.00      0.00      0.00        61
#     19.07.18       0.00      0.00      0.00        27
#     19.07.19       0.00      0.00      0.00        37
#     19.07.20       0.00      0.00      0.00         9
#     19.07.21       0.00      0.00      0.00       135
#     19.07.23       0.00      0.00      0.00         7
#     19.07.25       0.00      0.00      0.00        12
#     19.09.01       0.00      0.00      0.00        29
#     19.09.02       0.00      0.00      0.00         4
#     19.09.03       0.00      0.00      0.00        10
#     19.09.04       0.00      0.00      0.00        11
#     19.09.05       0.00      0.00      0.00        12
#     19.09.06       0.00      0.00      0.00         8
#     19.09.07       0.00      0.00      0.00        21
#     19.09.12       0.00      0.00      0.00        56
#     19.09.25       0.00      0.00      0.00         9
#     19.11.01       0.00      0.00      0.00        30
#     19.13.01       0.00      0.00      0.00        79
#     19.13.02       0.00      0.00      0.00        27
#     19.13.25       0.00      0.00      0.00         5
#     20.01.01       0.00      0.00      0.00         3
#     20.01.02       0.00      0.00      0.00         5
#     20.01.03       0.00      0.00      0.00         9
#     20.01.04       0.00      0.00      0.00        27
#     20.01.05       0.00      0.00      0.00        26
#     20.01.06       0.00      0.00      0.00        87
#     20.01.07       0.00      0.00      0.00         2
#     20.01.08       0.00      0.00      0.00        13
#     20.01.09       0.00      0.00      0.00       204
#     20.01.25       0.00      0.00      0.00         4
#     20.03.01       0.00      0.00      0.00         3
#     20.03.02       0.00      0.00      0.00        85
#     20.03.03       0.00      0.00      0.00        88
#     20.03.04       0.00      0.00      0.00        21
#     20.03.05       0.00      0.00      0.00         6
#     20.03.06       0.00      0.00      0.00         9
#     20.03.07       0.00      0.00      0.00         2
#     20.03.08       0.00      0.00      0.00         3
#     20.03.09       0.00      0.00      0.00        38
#     20.03.10       0.00      0.00      0.00         8
#     20.03.24       0.00      0.00      0.00        16
#     20.03.25       0.00      0.00      0.00        35
#     20.05.01       0.00      0.00      0.00        59
#     20.05.02       0.00      0.00      0.00         1
#     20.05.04       0.00      0.00      0.00        44
#     20.05.05       0.00      0.00      0.00       414
#     21.01.01       0.00      0.00      0.00        11
#     21.01.02       0.00      0.00      0.00        29
#     21.01.03       0.00      0.00      0.00         9
#     21.01.04       0.00      0.00      0.00         8
#     21.01.05       0.00      0.00      0.00        12
#     21.01.06       0.00      0.00      0.00        19
#     21.01.07       0.00      0.00      0.00         4
#     21.01.08       0.00      0.00      0.00         1
#     21.01.09       0.00      0.00      0.00         1
#     21.01.10       0.00      0.00      0.00         2
#     21.01.11       0.00      0.00      0.00        31
#     21.01.12       0.00      0.00      0.00        51
#     21.01.13       0.00      0.00      0.00         2
#     21.01.14       0.00      0.00      0.00        64
#     21.01.15       0.00      0.00      0.00         3
#     21.01.25       0.00      0.00      0.00         7
#     21.03.01       0.00      0.00      0.00        22
#     21.03.02       0.00      0.00      0.00        14
#     21.03.03       0.00      0.00      0.00         2
#     21.03.04       0.00      0.00      0.00         3
#     21.03.05       0.00      0.00      0.00         1
#     21.03.06       0.00      0.00      0.00        11
#     21.03.07       0.00      0.00      0.00        17
#     21.03.08       0.00      0.00      0.00         9
#     21.03.09       0.00      0.00      0.00        38
#     21.03.11       0.00      0.00      0.00         1
#     21.03.12       0.00      0.00      0.00         2
#     21.03.13       0.00      0.00      0.00       383
#     21.03.14       0.00      0.00      0.00       112
#     21.03.15       0.00      0.00      0.00        12
#     21.03.16       0.00      0.00      0.00        13
#     21.03.17       0.00      0.00      0.00        12
#     21.03.18       0.00      0.00      0.00        22
#     21.03.19       0.00      0.00      0.00         7
#     21.03.20       0.00      0.00      0.00         6
#     21.03.22       0.00      0.00      0.00       293
#     21.03.23       0.00      0.00      0.00        14
#     21.03.24       0.00      0.00      0.00        47
#     21.03.25       0.00      0.00      0.00       136
#     21.03.26       0.00      0.00      0.00        32
#     21.03.27       0.00      0.00      0.00        13
#     22.01.01       0.00      0.00      0.00         2
#     22.01.02       0.00      0.00      0.00        19
#     22.01.04       0.00      0.00      0.00         6
#     22.01.06       0.00      0.00      0.00        22
#     22.01.07       0.00      0.00      0.00         2
#     22.01.09       0.00      0.00      0.00         1
#     22.01.10       0.00      0.00      0.00         4
#     22.01.12       0.00      0.00      0.00         1
#     22.01.14       0.00      0.00      0.00         7
#     22.01.25       0.00      0.00      0.00        11
#     22.03.01       0.00      0.00      0.00       404
#     22.03.02       0.00      0.00      0.00         8
#     22.03.24       0.00      0.00      0.00        52
#     22.05.03       0.00      0.00      0.00         4
#     22.05.25       0.00      0.00      0.00        57
#     23.01.01       0.00      0.00      0.00        20
#     23.01.02       0.00      0.00      0.00         1
#     23.01.03       0.00      0.00      0.00         5
#     23.03.01       0.00      0.00      0.00         0
#     23.03.03       0.00      0.00      0.00         0
#     23.03.07       0.00      0.00      0.00         0
#     23.03.12       0.00      0.00      0.00         0
#     24.01.01       0.00      0.00      0.00         0
#     24.01.02       0.00      0.00      0.00         0
#     24.01.04       0.00      0.00      0.00         0
#     24.01.05       0.00      0.00      0.00         0
#     24.03.04       0.00      0.00      0.00         0
#     24.07.10       0.00      0.00      0.00         0
#     24.09.07       0.00      0.00      0.00         0
#     24.13.01       0.00      0.00      0.00         0
#     24.13.03       0.00      0.00      0.00         0
#     24.17.07       0.00      0.00      0.00         0
#     24.17.18       0.00      0.00      0.00         0
#     24.21.01       0.00      0.00      0.00         0
#     25.01.01       0.00      0.00      0.00         0
#     26.01.04       0.00      0.00      0.00         0
#     26.01.08       0.00      0.00      0.00         0
#     26.01.18       0.00      0.00      0.00         0
#     26.01.27       0.00      0.00      0.00         0
#     26.03.09       0.00      0.00      0.00         0
#     26.03.21       0.00      0.00      0.00         0
#     26.09.08       0.00      0.00      0.00         0
#     26.09.13       0.00      0.00      0.00         0
#     26.11.02       0.00      0.00      0.00         0
#     26.11.11       0.00      0.00      0.00         0
#     26.11.12       0.00      0.00      0.00         0
#     26.11.13       0.00      0.00      0.00         0
#     26.13.01       0.00      0.00      0.00         0
#     26.13.03       0.00      0.00      0.00         0
#     26.13.08       0.00      0.00      0.00         0
#     26.13.09       0.00      0.00      0.00         0
#     26.13.13       0.00      0.00      0.00         0
#     26.15.01       0.00      0.00      0.00         0
#     26.15.03       0.00      0.00      0.00         0
#     26.15.07       0.00      0.00      0.00         0
#     26.15.27       0.00      0.00      0.00         0
#     26.15.28       0.00      0.00      0.00         0
#     26.17.01       0.00      0.00      0.00         0
#     26.17.13       0.00      0.00      0.00         0
#     26.19.01       0.00      0.00      0.00         0
#     29.02.02       0.00      0.00      0.00         0
#     29.02.10       0.00      0.00      0.00         0
#     29.03.07       0.00      0.00      0.00         0

#     accuracy                           0.00     15499
#    macro avg       0.00      0.00      0.00     15499
# weighted avg       0.00      0.00      0.00     15499


# Model and label encoder for target_h3 saved successfully