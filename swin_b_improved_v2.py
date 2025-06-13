import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
from tqdm import tqdm
from torchvision.models import swin_v2_b, Swin_V2_B_Weights
from multiprocessing import freeze_support
import numpy as np
import random
import json
from datetime import datetime
import cv2
import argparse

# Normalization values for ImageNet
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

class AddGaussianNoise(object):
    """
    Add Gaussian noise to simulate real-world ECG recording variations.
    """
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        """
        Add noise to the tensor while preserving the ECG signal characteristics.
        """
        # Add noise only to the signal areas (non-white regions)
        mask = (tensor < 0.9).float()  # Create mask for signal areas
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise * mask  # Apply noise only to signal areas
    
    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

class AtrialSpecificAugmentation:
    """
    Specialized augmentation for atrial patterns to help model learn better features.
    """
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, img):
        if random.random() < self.p:
            # Convert to numpy for easier manipulation
            img_np = np.array(img)
            
            # 1. Simulate different P-wave morphologies (characteristic of atrial patterns)
            if random.random() < 0.3:
                # Add small variations to simulate different P-wave shapes
                height, width = img_np.shape[:2]
                center_y = height // 2
                for x in range(width):
                    if random.random() < 0.1:  # Only modify some points
                        offset = random.randint(-2, 2)
                        # Apply offset to all channels
                        img_np[center_y-5:center_y+5, x, :] = np.clip(
                            img_np[center_y-5:center_y+5, x, :] + offset, 0, 255
                        )
            
            # 2. Simulate different PR intervals
            if random.random() < 0.3:
                # Stretch or compress the PR interval slightly
                height, width = img_np.shape[:2]
                center_x = width // 2
                stretch_factor = random.uniform(0.95, 1.05)
                left_half = img_np[:, :center_x, :]  # Include all channels
                right_half = img_np[:, center_x:, :]  # Include all channels
                if stretch_factor > 1:
                    # Stretch
                    left_half = cv2.resize(left_half, (int(center_x * stretch_factor), height))
                    img_np = np.concatenate([left_half[:, :center_x, :], right_half], axis=1)
                else:
                    # Compress
                    right_half = cv2.resize(right_half, (int(center_x * (2-stretch_factor)), height))
                    img_np = np.concatenate([left_half, right_half[:, :center_x, :]], axis=1)
            
            # 3. Add subtle baseline wander (common in atrial recordings)
            if random.random() < 0.3:
                height, width = img_np.shape[:2]
                x = np.linspace(0, 2*np.pi, width)
                baseline = np.sin(x) * random.uniform(1, 3)
                # Apply baseline to all channels
                for c in range(img_np.shape[2]):
                    img_np[:, :, c] = np.clip(img_np[:, :, c] + baseline[:, np.newaxis], 0, 255)
            
            # Add more specific augmentations for atrial patterns
            # 1. P-wave variations
            if random.random() < 0.3:
                # Add more subtle P-wave variations
                height, width = img_np.shape[:2]
                center_y = height // 2
                for x in range(width):
                    if random.random() < 0.05:  # Reduced probability
                        offset = random.randint(-1, 1)  # Smaller variations
                        img_np[center_y-3:center_y+3, x, :] = np.clip(
                            img_np[center_y-3:center_y+3, x, :] + offset, 0, 255
                        )
            
            # 2. Add more realistic atrial patterns
            if random.random() < 0.3:
                # Simulate different atrial rates
                rate_variation = random.uniform(0.95, 1.05)
                # Apply rate variation to signal
            
            return Image.fromarray(img_np.astype(np.uint8))
        return img

class NormalSpecificAugmentation:
    """
    Specialized augmentation for Normal patterns to help model learn better features.
    """
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, img):
        if random.random() < self.p:
            # Convert to numpy for easier manipulation
            img_np = np.array(img)
            
            # 1. Simulate different P-wave morphologies (characteristic of Normal patterns)
            if random.random() < 0.3:
                # Add small variations to simulate different P-wave shapes
                height, width = img_np.shape[:2]
                center_y = height // 2
                for x in range(width):
                    if random.random() < 0.1:  # Only modify some points
                        offset = random.randint(-2, 2)
                        # Apply offset to all channels
                        img_np[center_y-5:center_y+5, x, :] = np.clip(
                            img_np[center_y-5:center_y+5, x, :] + offset, 0, 255
                        )
            
            # 2. Simulate different PR intervals
            if random.random() < 0.3:
                # Stretch or compress the PR interval slightly
                height, width = img_np.shape[:2]
                center_x = width // 2
                stretch_factor = random.uniform(0.95, 1.05)
                left_half = img_np[:, :center_x, :]  # Include all channels
                right_half = img_np[:, center_x:, :]  # Include all channels
                if stretch_factor > 1:
                    # Stretch
                    left_half = cv2.resize(left_half, (int(center_x * stretch_factor), height))
                    img_np = np.concatenate([left_half[:, :center_x, :], right_half], axis=1)
                else:
                    # Compress
                    right_half = cv2.resize(right_half, (int(center_x * (2-stretch_factor)), height))
                    img_np = np.concatenate([left_half, right_half[:, :center_x, :]], axis=1)
            
            # 3. Add subtle baseline wander (common in Normal recordings)
            if random.random() < 0.3:
                height, width = img_np.shape[:2]
                x = np.linspace(0, 2*np.pi, width)
                baseline = np.sin(x) * random.uniform(1, 3)
                # Apply baseline to all channels
                for c in range(img_np.shape[2]):
                    img_np[:, :, c] = np.clip(img_np[:, :, c] + baseline[:, np.newaxis], 0, 255)
            
            return Image.fromarray(img_np.astype(np.uint8))
        return img

class ClassBalancedFocalLoss(nn.Module):
    """
    Focal Loss with class balancing and dynamic weighting.
    Specifically designed for ECG classification with emphasis on atrial patterns.
    """
    def __init__(self, alpha=None, gamma=2, reduction='mean', class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = class_weights
        
    def forward(self, inputs, targets):
        # Ensure targets are on the correct device
        targets = targets.to(inputs.device)
        
        # Get class weights if not provided
        if self.class_weights is None:
            # Calculate class weights based on batch distribution
            unique_labels, counts = torch.unique(targets, return_counts=True)
            total_samples = counts.sum().float()
            class_weights = total_samples / (len(unique_labels) * counts.float())
            class_weights = class_weights ** 0.5  # Moderate the weights
            self.class_weights = class_weights.to(inputs.device)
        elif not isinstance(self.class_weights, torch.Tensor) or self.class_weights.device != inputs.device:
            # Ensure class weights are on the correct device
            if hasattr(self.class_weights, 'get_class_weights'):
                self.class_weights = self.class_weights.get_class_weights(inputs.device)
            else:
                self.class_weights = self.class_weights.to(inputs.device)
        
        # Calculate focal loss
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        
        # Get weights for current batch
        alpha_weight = self.class_weights[targets.long()]  # Ensure targets are long type
        
        # Add class-specific gamma values
        gamma_values = torch.ones_like(targets, dtype=torch.float32) * self.gamma
        gamma_values[targets == 0] = 1.5  # Lower gamma for atrial class
        gamma_values[targets == 2] = 2.5  # Higher gamma for normal class
        
        # Calculate focal loss with class-specific gamma
        focal_loss = alpha_weight * (1-pt)**gamma_values * ce_loss
        
        # Add a regularization term for atrial class
        atrial_mask = (targets == 0)
        if atrial_mask.any():
            atrial_outputs = inputs[atrial_mask]
            atrial_targets = targets[atrial_mask]
            # Add a margin-based loss for atrial class
            margin = 0.5
            atrial_loss = torch.clamp(
                margin - (atrial_outputs[:, 0] - atrial_outputs[:, 1:].max(dim=1)[0]),
                min=0.0
            ).mean()
            focal_loss = focal_loss.mean() + 0.1 * atrial_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss

class MixupAugmentation:
    """
    Implements mixup augmentation for ECG images.
    Mixup creates new training samples by mixing pairs of images and their labels.
    """
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        
    def __call__(self, img1, img2, label1, label2):
        """
        Mix two images and their labels.
        
        Args:
            img1, img2: Input images (tensors)
            label1, label2: Corresponding labels
            
        Returns:
            mixed_img: Mixed image
            mixed_label: Mixed label (as a tensor of shape [batch_size])
        """
        # Generate mixing coefficient from beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Mix images
        mixed_img = lam * img1 + (1 - lam) * img2
        
        # For labels, we'll use the label with higher mixing coefficient
        # This is a simpler approach that works better with our training setup
        mixed_label = label1 if lam > 0.5 else label2
        
        return mixed_img, mixed_label

class ECGImageDataset(Dataset):
    def __init__(self, base_dir, split, image_size=224, is_train=False):
        """
        Dataset class for ECG image data with enhanced atrial pattern handling.
        
        Args:
            base_dir (str): Base directory containing the dataset
            split (str): Dataset split ('train', 'validate', or 'test')
            image_size (int): Target image size
            is_train (bool): Whether this is training data (enables augmentation)
        """
        self.base_dir = base_dir
        self.split = split
        self.is_train = is_train
        self.image_size = image_size
        
        # ECG-specific classes
        self.classes = ['A', 'L', 'N', 'R', 'V']  # A: Atrial, L: Left bundle, N: Normal, R: Right bundle, V: Ventricular
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        # Create list of (image_path, label) pairs
        self.samples = []
        self.class_counts = {c: 0 for c in self.classes}
        
        # Load and validate data
        for class_name in self.classes:
            class_dir = os.path.join(base_dir, split, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith('.png'):
                        img_path = os.path.join(class_dir, img_name)
                        # Basic validation of image
                        try:
                            with Image.open(img_path) as img:
                                if img.mode != 'RGB':
                                    print(f"Warning: Converting {img_path} to RGB")
                                if img.size[0] < image_size or img.size[1] < image_size:
                                    print(f"Warning: Image {img_path} is smaller than target size")
                        except Exception as e:
                            print(f"Error loading {img_path}: {str(e)}")
                            continue
                            
                        self.samples.append((img_path, self.class_to_idx[class_name]))
                        self.class_counts[class_name] += 1
        
        if not self.samples:
            raise RuntimeError(f"No valid images found in {base_dir}/{split}")
            
        print(f"\nDataset statistics for {split} split:")
        print(f"Total samples: {len(self.samples)}")
        for cls, count in self.class_counts.items():
            print(f"Class {cls}: {count} samples ({count/len(self.samples)*100:.1f}%)")
        
        # Define transforms with atrial-specific augmentations
        if self.is_train:
            self.transform = transforms.Compose([
                # Preserve ECG signal characteristics
                transforms.CenterCrop(image_size + 32),
                transforms.RandomCrop(image_size, padding=8),
                
                # Atrial-specific augmentations
                AtrialSpecificAugmentation(p=0.5),  # Apply atrial-specific augmentations
                
                # ECG-specific augmentations
                transforms.RandomAffine(
                    degrees=(-5, 5),
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05),
                    fill=255
                ),
                
                # Signal quality variations
                transforms.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0,
                    hue=0
                ),
                
                transforms.ToTensor(),
                transforms.Normalize(imagenet_mean, imagenet_std),
                
                # Add realistic ECG noise
                AddGaussianNoise(std=0.02),
                
                # Random horizontal flip with very low probability for atrial class
                transforms.RandomHorizontalFlip(p=0.05)  # Reduced probability
            ])
            
            # Initialize mixup augmentation
            self.mixup = MixupAugmentation(alpha=0.2)
            self.use_mixup = True  # Flag to control mixup usage
        else:
            self.transform = transforms.Compose([
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(imagenet_mean, imagenet_std),
            ])
            self.use_mixup = False
        
        # Calculate initial class weights as a tensor
        self.class_weights = self._calculate_class_weights()
        # Convert dictionary to tensor and move to CPU (will be moved to GPU when needed)
        self.class_weights = torch.tensor([self.class_weights[i] for i in range(len(self.classes))], 
                                        device='cpu')

    def _calculate_class_weights(self):
        """Calculate balanced class weights with moderate emphasis on minority classes"""
        total_samples = sum(self.class_counts.values())
        weights = {}
        
        # Calculate base weights with square root to reduce extreme differences
        for idx, cls in enumerate(self.classes):
            # Use square root to moderate the inverse frequency effect
            base_weight = (total_samples / (len(self.classes) * self.class_counts[cls])) ** 0.5
            
            # Apply class-specific adjustments
            if cls == 'A':  # Atrial
                base_weight *= 1.2  # Moderate boost for atrial
            elif cls == 'N':  # Normal
                base_weight = max(base_weight, 0.4)  # Ensure minimum weight for Normal
            else:  # Other classes
                base_weight *= 1.1  # Slight boost for other minority classes
            
            weights[idx] = base_weight
        
        # Normalize weights
        max_weight = max(weights.values())
        return {idx: weight/max_weight for idx, weight in weights.items()}

    def get_class_weights(self, device):
        """Get class weights tensor on the specified device."""
        return self.class_weights.to(device)

    def __getitem__(self, idx):
        """
        Get a single ECG image and its label.
        If training and mixup is enabled, returns mixed image and label.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (image tensor, label tensor)
        """
        img_path, label = self.samples[idx]
        try:
            # Load and convert to RGB
            image = Image.open(img_path).convert("RGB")
            
            # Apply transforms
            image = self.transform(image)
            
            # Validate transformed image
            if torch.isnan(image).any():
                print(f"Warning: NaN values in transformed image {img_path}")
                # Replace NaN with zeros
                image = torch.nan_to_num(image, 0.0)
            
            # Convert label to tensor
            label = torch.tensor(label, dtype=torch.long)
            
            # Apply mixup if training
            if self.is_train and self.use_mixup and random.random() < 0.5:
                # Get another random sample
                idx2 = random.randint(0, len(self.samples) - 1)
                img_path2, label2 = self.samples[idx2]
                image2 = Image.open(img_path2).convert("RGB")
                image2 = self.transform(image2)
                label2 = torch.tensor(label2, dtype=torch.long)
                
                # Apply mixup
                mixed_image, mixed_label = self.mixup(image, image2, label, label2)
                return mixed_image, mixed_label
            
            return image, label
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            # Return a zero tensor as fallback
            return torch.zeros((3, self.image_size, self.image_size)), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.samples)

def get_class_weights(dataset):
    """
    Calculate class weights for ECG dataset to handle class imbalance.
    Uses inverse frequency weighting with smoothing.
    
    Args:
        dataset (ECGImageDataset): The dataset to calculate weights for
        
    Returns:
        dict: Class weights for each class
    """
    class_counts = dataset.class_counts
    total_samples = sum(class_counts.values())
    
    # Add smoothing to prevent extreme weights
    smoothing_factor = 0.1
    class_weights = {
        cls: (total_samples + smoothing_factor) / 
             (len(class_counts) * (count + smoothing_factor))
        for cls, count in class_counts.items()
    }
    
    # Normalize weights
    max_weight = max(class_weights.values())
    class_weights = {cls: weight/max_weight for cls, weight in class_weights.items()}
    
    return class_weights

def get_loader(split, batch_size=64, num_workers=4):
    """
    Get data loader for the specified split.
    
    Args:
        split (str): Dataset split ('train', 'validate', or 'test')
        batch_size (int): Batch size for the loader
        num_workers (int): Number of workers for data loading
        
    Returns:
        DataLoader: PyTorch DataLoader for the specified split
    """
    base_dir = 'dataset'
    is_train = (split == 'train')
    dataset = ECGImageDataset(base_dir, split, is_train=is_train)
    
    if is_train:
        # Calculate class weights for weighted sampling (inverse‐frequency → [0,1])
        class_weights = get_class_weights(dataset)

        # ─────────────────────────────────────────────────────────────────────────────
        # Clip weights to soften extremes:
        #   · min_weight prevents the majority class from being nearly ignored
        #   · max_weight prevents any minority class from dominating too much
        min_weight, max_weight = 0.2, 0.8
        class_weights = {
            cls: float(max(min(w, max_weight), min_weight))
            for cls, w in class_weights.items()
        }
        # ─────────────────────────────────────────────────────────────────────────────

        # Now build per‐sample weights from the clipped class_weights
        sample_weights = [
            class_weights[ dataset.classes[label] ]
            for _, label in dataset.samples
        ]
        sampler = WeightedRandomSampler(sample_weights,
                                        num_samples=len(sample_weights),
                                        replacement=True)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def save_checkpoint(model, optimizer, scheduler, epoch, best_val_acc, train_metrics, val_metrics, 
                   class_distribution, save_dir='checkpoints'):
    """
    Save model checkpoint and training metadata
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate timestamp for unique checkpoint name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_name = f'checkpoint_{timestamp}.pt'
    metadata_name = f'metadata_{timestamp}.json'
    
    # Save model checkpoint
    checkpoint_path = os.path.join(save_dir, checkpoint_name)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_acc': best_val_acc,
    }, checkpoint_path)
    
    # Save training metadata
    metadata = {
        'timestamp': timestamp,
        'epoch': epoch,
        'best_val_acc': best_val_acc,
        'class_distribution': class_distribution,
        'train_metrics': {
            'accuracies': train_metrics['accuracies'],
            'losses': train_metrics['losses'],
        },
        'val_metrics': {
            'accuracies': val_metrics['accuracies'],
            'losses': val_metrics['losses'],
        },
        'model_config': {
            'image_size': 224,
            'num_classes': 5,
            'classes': ['A', 'L', 'N', 'R', 'V'],
        },
        'optimizer_config': {
            'lr': optimizer.param_groups[0]['lr'],
            'weight_decay': optimizer.param_groups[0]['weight_decay'],
        }
    }
    
    metadata_path = os.path.join(save_dir, metadata_name)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"\nCheckpoint saved:")
    print(f"Model: {checkpoint_path}")
    print(f"Metadata: {metadata_path}")

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """
    Load model checkpoint and optionally optimizer/scheduler states
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['best_val_acc']

def train_model(epochs=30, learning_rate=3e-5, weight_decay=0.02, 
                image_size=224, checkpoint_dir='checkpoints', resume_checkpoint=None):
    """
    Train the Swin Transformer model with enhanced atrial pattern handling.
    
    Args:
        epochs (int): Number of training epochs
        learning_rate (float): Initial learning rate
        weight_decay (float): Weight decay for optimizer
        image_size (int): Input image size
        checkpoint_dir (str): Directory to save checkpoints
        resume_checkpoint (str, optional): Path to checkpoint to resume training from
    """
    # ImageNet Pretrained Weights
    weights = Swin_V2_B_Weights.IMAGENET1K_V1
    model = swin_v2_b(weights=weights)
    model.head = nn.Linear(model.head.in_features, 5)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Use ClassBalancedFocalLoss with dataset's class weights
    criterion = ClassBalancedFocalLoss(
        gamma=2,
        class_weights=train_loader.dataset  # Pass the dataset instead of weights directly
    )
    
    # Use AdamW optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Use CosineAnnealingWarmRestarts scheduler
    # T_0 is the number of epochs for the first restart
    # T_mult is the factor to multiply T_0 after each restart
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,  # First restart after 5 epochs
        T_mult=2,  # Double the restart interval after each restart
        eta_min=learning_rate * 0.01  # Minimum learning rate is 1% of initial lr
    )
    
    # Add gradient clipping
    max_grad_norm = 1.0
    
    # Initialize training state
    start_epoch = 1
    best_val_acc = 0
    train_metrics = {
        'epochs': [],  # Store actual epoch numbers
        'accuracies': [], 
        'losses': [], 
        'class_accuracies': []
    }
    val_metrics = {
        'epochs': [],  # Store actual epoch numbers
        'accuracies': [], 
        'losses': []
    }
    
    # Resume training if checkpoint is provided
    if resume_checkpoint:
        print(f"\nResuming training from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['best_val_acc']
        
        # Load training metrics if available
        if 'train_metrics' in checkpoint:
            train_metrics = checkpoint['train_metrics']
        if 'val_metrics' in checkpoint:
            val_metrics = checkpoint['val_metrics']
        
        print(f"Resumed from epoch {start_epoch-1}")
        print(f"Previous best validation accuracy: {best_val_acc:.4f}")
    
    print(f"Number of samples in train dataset: {len(train_loader.dataset)}")
    print(f"Number of samples in val dataset: {len(val_loader.dataset)}")
    print(f"Training on device: {device}")
    print(f"Learning rate: {learning_rate}")
    print(f"Weight decay: {weight_decay}")
    print(f"Image size: {image_size}")
    
    # Print class distribution
    class_distribution = train_loader.dataset.class_counts
    print("\nClass distribution in training set:")
    for cls, count in class_distribution.items():
        print(f"Class {cls}: {count} samples")
    
    best_model_state = None

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_preds, train_labels = [], []
        running_loss = 0
        class_correct = {cls: 0 for cls in train_loader.dataset.classes}
        class_total = {cls: 0 for cls in train_loader.dataset.classes}

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Training]", leave=False)
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            running_loss += loss.item()
            
            # Track per-class accuracy
            preds = outputs.argmax(1)
            for pred, label in zip(preds, labels):
                cls = train_loader.dataset.classes[label]
                class_total[cls] += 1
                if pred == label:
                    class_correct[cls] += 1
            
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            # Update progress bar with per-class accuracy
            loop.set_postfix({
                'loss': loss.item(),
                'A_acc': f"{class_correct['A']/class_total['A']:.2f}" if class_total['A'] > 0 else "0.00",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })

        # Step the scheduler at the end of each epoch
        scheduler.step()
        
        # Print detailed epoch statistics
        print(f"\nEpoch {epoch}/{epochs} Statistics:")
        print(f"Overall Training Accuracy: {accuracy_score(train_labels, train_preds):.4f}")
        print("Per-class Training Accuracy:")
        for cls in train_loader.dataset.classes:
            acc = class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0
            print(f"  {cls}: {acc:.4f} ({class_correct[cls]}/{class_total[cls]})")
        
        train_acc = accuracy_score(train_labels, train_preds)
        train_loss = running_loss / len(train_loader)
        train_metrics['epochs'].append(epoch)
        train_metrics['accuracies'].append(train_acc)
        train_metrics['losses'].append(train_loss)

        # Store per-class accuracies
        epoch_class_accuracies = {
            cls: class_correct[cls] / class_total[cls] 
            for cls in train_loader.dataset.classes
        }
        train_metrics['class_accuracies'].append(epoch_class_accuracies)

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        val_loss = 0

        loop_val = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Validation]", leave=False)
        with torch.no_grad():
            for imgs, labels in loop_val:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_preds.extend(outputs.argmax(1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                loop_val.set_postfix(loss=loss.item())

        val_acc = accuracy_score(val_labels, val_preds)
        val_loss /= len(val_loader)
        val_metrics['epochs'].append(epoch)
        val_metrics['accuracies'].append(val_acc)
        val_metrics['losses'].append(val_loss)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            # Save checkpoint when we get a new best model
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_val_acc,
                train_metrics, val_metrics, class_distribution
            )

        print(f"Epoch {epoch}/{epochs} | "
              f"Train Acc: {train_acc:.4f}, Loss: {train_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}, Loss: {val_loss:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

    # Load best model for final evaluation
    model.load_state_dict(best_model_state)
    
    # Final evaluation
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            val_preds.extend(outputs.argmax(1).cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds, target_names=train_loader.dataset.classes))
    
    # Plot confusion matrix
    plot_confusion_matrix(val_labels, val_preds, train_loader.dataset.classes)

    # After training, plot the metrics using actual epoch numbers
    plt.figure(figsize=(15, 10))
    
    # Plot accuracy
    plt.subplot(2, 1, 1)
    plt.plot(train_metrics['epochs'], train_metrics['accuracies'], label='Train Acc', marker='o')
    plt.plot(val_metrics['epochs'], val_metrics['accuracies'], label='Val Acc', marker='o')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot loss
    plt.subplot(2, 1, 2)
    plt.plot(train_metrics['epochs'], train_metrics['losses'], label='Train Loss', marker='o')
    plt.plot(val_metrics['epochs'], val_metrics['losses'], label='Val Loss', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

    # Plot per-class accuracy
    plt.figure(figsize=(15, 10))
    for cls in train_loader.dataset.classes:
        class_accuracies = [metrics[cls] for metrics in train_metrics['class_accuracies']]
        plt.plot(train_metrics['epochs'], class_accuracies, label=f'Class {cls}', marker='o')
    
    plt.title('Per-Class Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('per_class_accuracy.png')
    plt.close()

    # Save final metrics
    final_metrics = {
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'best_val_acc': best_val_acc,
        'final_train_acc': train_metrics['accuracies'][-1],
        'final_val_acc': val_metrics['accuracies'][-1],
        'final_train_loss': train_metrics['losses'][-1],
        'final_val_loss': val_metrics['losses'][-1],
        'per_class_final_acc': train_metrics['class_accuracies'][-1]
    }
    
    with open('final_metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=4)

    return model, final_metrics

def regenerate_plots_from_checkpoint(checkpoint_dir='checkpoints', specific_checkpoint=None, max_epoch=None):
    """
    Load a checkpoint and regenerate training plots without retraining.
    
    Args:
        checkpoint_dir (str): Directory containing checkpoints
        specific_checkpoint (str, optional): Specific checkpoint file to use
        max_epoch (int, optional): Maximum epoch to plot up to
    """
    if specific_checkpoint:
        # Handle both relative and absolute paths
        if os.path.isabs(specific_checkpoint):
            checkpoint_path = specific_checkpoint
        else:
            checkpoint_path = os.path.join(checkpoint_dir, os.path.basename(specific_checkpoint))
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Specified checkpoint not found: {checkpoint_path}")
    else:
        # Find the latest checkpoint
        checkpoints = [f for f in os.listdir(checkpoint_dir) 
                      if f.startswith('checkpoint_') and f.endswith('.pt')]
        if not checkpoints:
            raise FileNotFoundError("No checkpoints found in the specified directory")
        
        latest_checkpoint = max(checkpoints, 
                              key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x)))
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    
    # Find corresponding metadata file
    metadata_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('metadata_') and f.endswith('.json')]
    if not metadata_files:
        raise FileNotFoundError("No metadata files found in the specified directory")
    
    # Get the metadata file that was created closest to the checkpoint
    checkpoint_time = os.path.getctime(checkpoint_path)
    metadata_path = min(metadata_files, 
                       key=lambda x: abs(os.path.getctime(os.path.join(checkpoint_dir, x)) - checkpoint_time))
    metadata_path = os.path.join(checkpoint_dir, metadata_path)
    
    print(f"Using checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"Using metadata: {metadata_path}")
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Extract metrics
    train_metrics = metadata['train_metrics']
    val_metrics = metadata['val_metrics']
    
    # Filter metrics up to max_epoch if specified
    if max_epoch is not None:
        train_metrics = {k: v[:max_epoch] for k, v in train_metrics.items()}
        val_metrics = {k: v[:max_epoch] for k, v in val_metrics.items()}
        epochs = max_epoch
    else:
        epochs = len(train_metrics['accuracies'])
    
    # Create output directory
    output_dir = os.path.join(checkpoint_dir, 'plots')
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot training curves
    plt.figure(figsize=(15, 10))
    
    # Accuracy plot
    plt.subplot(2, 1, 1)
    plt.plot(range(1, epochs + 1), train_metrics['accuracies'], label='Train Acc', marker='o')
    plt.plot(range(1, epochs + 1), val_metrics['accuracies'], label='Val Acc', marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy over Epochs (up to epoch {epochs})")
    plt.legend()
    plt.grid(True)

    # Loss plot
    plt.subplot(2, 1, 2)
    plt.plot(range(1, epochs + 1), train_metrics['losses'], label='Train Loss', marker='o')
    plt.plot(range(1, epochs + 1), val_metrics['losses'], label='Val Loss', marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss over Epochs (up to epoch {epochs})")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    
    # Save the plot with epoch information
    plot_filename = f'training_curves_epoch_{epochs}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path)
    print(f"\nPlots have been regenerated and saved to: {plot_path}")
    
    # Also generate per-class accuracy plot
    if 'class_accuracies' in train_metrics:
        plt.figure(figsize=(15, 10))
        classes = list(train_metrics['class_accuracies'][0].keys())
        for cls in classes:
            class_accuracies = [metrics[cls] for metrics in train_metrics['class_accuracies'][:epochs]]
            plt.plot(range(1, epochs + 1), class_accuracies, label=f'Class {cls}', marker='o')
        
        plt.title(f'Per-Class Training Accuracy (up to epoch {epochs})')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Save per-class accuracy plot
        per_class_filename = f'per_class_accuracy_epoch_{epochs}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        per_class_path = os.path.join(output_dir, per_class_filename)
        plt.savefig(per_class_path)
        print(f"Per-class accuracy plot saved to: {per_class_path}")
    
    plt.close('all')

def parse_args():
    parser = argparse.ArgumentParser(description='Swin Transformer Training and Testing')
    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--test', action='store_true', help='Run testing')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.02, help='Weight decay')
    parser.add_argument('--evaluate-test', action='store_true', help='Evaluate on test set')
    parser.add_argument('--regenerate-plots', action='store_true', help='Regenerate plots from saved metrics')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint for evaluation')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--test-results-dir', type=str, default='test_results', help='Directory to save test results')
    parser.add_argument('--image-size', type=int, default=224, help='Input image size')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max-epoch', type=int, help='Maximum epoch to plot up to when regenerating plots')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create directories if they don't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.test_results_dir, exist_ok=True)
    
    # Initialize data loaders
    global train_loader, val_loader, test_loader
    if args.train or args.test or args.evaluate_test:
        train_loader = get_loader('train', batch_size=args.batch_size, num_workers=args.num_workers)
        val_loader = get_loader('validate', batch_size=args.batch_size, num_workers=args.num_workers)
        test_loader = get_loader('test', batch_size=args.batch_size, num_workers=args.num_workers)
    
    # Initialize criterion
    criterion = ClassBalancedFocalLoss(
        gamma=2,
        class_weights=train_loader.dataset if args.train or args.test or args.evaluate_test else None
    )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.train:
        print("Starting training...")
        model, metrics = train_model(
            epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            image_size=args.image_size,
            checkpoint_dir=args.checkpoint_dir,
            resume_checkpoint=args.resume
        )
        print("\nTraining completed!")
        print(f"Best validation accuracy: {metrics['best_val_acc']:.4f}")
        print(f"Final validation accuracy: {metrics['final_val_acc']:.4f}")
        
    elif args.test:
        if not args.checkpoint:
            print("Error: --checkpoint argument is required for testing")
            return
            
        print(f"Loading checkpoint from {args.checkpoint}")
        model = load_model_for_testing(args.checkpoint, args.image_size)
        
        print("Running inference on test set...")
        test_metrics = evaluate_model(
            model, 
            test_loader, 
            criterion, 
            device,
            save_dir=args.test_results_dir
        )
        
        print("\nTest Results:")
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test Loss: {test_metrics['loss']:.4f}")
        print("\nPer-class Metrics:")
        for cls, metrics in test_metrics['per_class'].items():
            print(f"\nClass {cls}:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-score: {metrics['f1']:.4f}")
            print(f"  Support: {metrics['support']}")
            
        # Save test results
        results_file = os.path.join(args.test_results_dir, 'test_results.json')
        with open(results_file, 'w') as f:
            json.dump(test_metrics, f, indent=4)
        print(f"\nTest results saved to {results_file}")
        
    elif args.evaluate_test:
        if not args.checkpoint:
            print("Error: --checkpoint argument is required for evaluation")
            return
            
        print(f"Loading checkpoint from {args.checkpoint}")
        model = load_model_for_testing(args.checkpoint, args.image_size)
        
        print("Evaluating on test set...")
        test_metrics = evaluate_model(
            model, 
            test_loader, 
            criterion, 
            device,
            save_dir=args.test_results_dir
        )
        
        print("\nTest Set Evaluation Results:")
        print(f"Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Loss: {test_metrics['loss']:.4f}")
        print("\nPer-class Metrics:")
        for cls, metrics in test_metrics['per_class'].items():
            print(f"\nClass {cls}:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-score: {metrics['f1']:.4f}")
            print(f"  Support: {metrics['support']}")
            
    elif args.regenerate_plots:
        if not args.checkpoint:
            print("Error: --checkpoint argument is required for regenerating plots")
            return
            
        print(f"Regenerating plots from checkpoint: {args.checkpoint}")
        regenerate_plots_from_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            specific_checkpoint=args.checkpoint,
            max_epoch=args.max_epoch
        )
    else:
        print("Please specify either --train, --test, --evaluate-test, or --regenerate-plots")

def load_model_for_testing(checkpoint_path, image_size=224):
    """
    Load a trained model from checkpoint for testing.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        image_size (int): Input image size
        
    Returns:
        model: Loaded and prepared model
    """
    # Initialize model with ImageNet weights
    weights = Swin_V2_B_Weights.IMAGENET1K_V1
    model = swin_v2_b(weights=weights)
    model.head = nn.Linear(model.head.in_features, 5)  # 5 classes
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set model to evaluation mode
    model.eval()
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    return model

def evaluate_model(model, test_loader, criterion, device, save_dir='test_results'):
    """
    Evaluate the model on the test set and save results.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test set
        criterion: Loss function
        device: Device to run evaluation on
        save_dir (str): Directory to save results
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0
    
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Evaluating"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    avg_loss = running_loss / len(test_loader)
    
    # Calculate per-class metrics
    per_class_metrics = {}
    for cls in test_loader.dataset.classes:
        cls_idx = test_loader.dataset.class_to_idx[cls]
        cls_preds = [1 if p == cls_idx else 0 for p in all_preds]
        cls_labels = [1 if l == cls_idx else 0 for l in all_labels]
        
        precision = precision_score(cls_labels, cls_preds, zero_division=0)
        recall = recall_score(cls_labels, cls_preds, zero_division=0)
        f1 = f1_score(cls_labels, cls_preds, zero_division=0)
        support = sum(cls_labels)
        
        per_class_metrics[cls] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        }
    
    # Generate and save confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=test_loader.dataset.classes,
                yticklabels=test_loader.dataset.classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    
    # Save confusion matrix
    os.makedirs(save_dir, exist_ok=True)
    cm_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    
    # Save classification report
    report = classification_report(all_labels, all_preds, 
                                 target_names=test_loader.dataset.classes,
                                 output_dict=True)
    report_path = os.path.join(save_dir, 'classification_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    # Return metrics
    metrics = {
        'accuracy': accuracy,
        'loss': avg_loss,
        'per_class': per_class_metrics,
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    
    return metrics

if __name__ == '__main__':
    main()