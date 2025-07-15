import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import json
import os
import numpy as np
from tqdm import tqdm
import time

# ROCm-specific optimizations
def setup_rocm():
    """Configure PyTorch for optimal ROCm performance"""
    if torch.cuda.is_available():
        print(f"ROCm detected: {torch.cuda.get_device_name(0)}")
        print(f"ROCm version: {torch.version.cuda}")
        
        # Enable optimizations for RDNA3 (9800 XT)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        # Set memory management for ROCm
        torch.cuda.empty_cache()
        
        device = torch.device('cuda:0')
        print(f"Using device: {device}")
        return device
    else:
        print("ROCm not available, using CPU")
        return torch.device('cpu')

class NuclearThroneDataset(Dataset):
    """Optimized dataset for Nuclear Throne training data"""
    
    def __init__(self, json_path, base_dir, transform=None, cache_images=False, data_format='regression'):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.base_dir = base_dir
        self.transform = transform
        self.cache_images = cache_images
        self.image_cache = {} if cache_images else None
        self.data_format = data_format
        
        # Support both cleaned and original data formats
        self.samples = self.data.get('samples', [])
        self.image_size = self.data.get('image_size', [240, 320])  # height, width
        self.output_names = self.data.get('output_names', ['movement_x', 'movement_y', 'aim_x', 'aim_y', 'shooting'])
        
        # For classification format
        self.num_classes = self.data.get('num_classes', 0)
        self.class_names = self.data.get('class_names', [])
        
        print(f"Loaded {len(self.samples)} {data_format} samples")
        print(f"Image size: {self.image_size}")
        if data_format == 'classification':
            print(f"Number of classes: {self.num_classes}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image (with caching for speed)
        if self.cache_images and idx in self.image_cache:
            image = self.image_cache[idx]
        else:
            # Handle both cleaned data (relative paths) and original data (absolute paths)
            screenshot_path = sample.get('screenshot', sample.get('screenshotFile', ''))
            
            if os.path.isabs(screenshot_path):
                image_path = screenshot_path
            else:
                image_path = os.path.join(self.base_dir, screenshot_path)
            
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                # Return a black image as fallback
                image = Image.new('RGB', (self.image_size[1], self.image_size[0]), color=(0, 0, 0))
            
            if self.cache_images:
                self.image_cache[idx] = image
        
        if self.transform:
            image = self.transform(image)
        
        # Handle different data formats
        if self.data_format == 'regression':
            outputs = sample['outputs']
            target = torch.tensor([
                outputs['movement_x'],
                outputs['movement_y'], 
                outputs['aim_x'],      # Already normalized in cleaned data
                outputs['aim_y'],      # Already normalized in cleaned data
                outputs['shooting']
            ], dtype=torch.float32)
        
        elif self.data_format == 'classification':
            class_index = sample.get('class_index', 0)
            target = torch.tensor(class_index, dtype=torch.long)
        
        else:
            raise ValueError(f"Unsupported data format: {self.data_format}")
        
        return image, target

class GameAIModel(nn.Module):
    """Optimized model for Nuclear Throne AI"""
    
    def __init__(self, num_outputs=5, backbone='efficientnet_b0'):
        super().__init__()
        
        if backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=True)
            backbone_features = 1280
        elif backbone == 'mobilenet_v3_small':
            self.backbone = models.mobilenet_v3_small(pretrained=True)
            backbone_features = 576
        elif backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            backbone_features = 512
        
        # Replace final layer
        if hasattr(self.backbone, 'classifier'):
            self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, 'fc'):
            self.backbone.fc = nn.Identity()
        
        # Custom head for game actions
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(backbone_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_outputs)
        )
        
        # Freeze backbone initially (optional)
        self.freeze_backbone = True
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.freeze_backbone = False
    
    def forward(self, x):
        # Extract features
        features = self.backbone.features(x) if hasattr(self.backbone, 'features') else self.backbone(x)
        
        # Get predictions
        outputs = self.head(features)
        
        # Apply appropriate activations
        movement = torch.tanh(outputs[:, :2])      # Movement: -1 to 1
        aim = torch.sigmoid(outputs[:, 2:4])       # Aim: 0 to 1
        shoot = torch.sigmoid(outputs[:, 4:5])     # Shoot: 0 to 1
        
        return torch.cat([movement, aim, shoot], dim=1)

class GameAITrainer:
    """ROCm-optimized trainer for Nuclear Throne AI"""
    
    def __init__(self, model, device, learning_rate=1e-3):
        self.model = model.to(device)
        self.device = device
        
        # Use AdamW optimizer (good for fine-tuning)
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Custom loss function
        self.criterion = self.create_loss_function()
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
    
    def create_loss_function(self):
        """Create weighted loss for different action types"""
        def game_loss(predictions, targets):
            # Split predictions and targets
            pred_movement = predictions[:, :2]
            pred_aim = predictions[:, 2:4]
            pred_shoot = predictions[:, 4:5]
            
            target_movement = targets[:, :2]
            target_aim = targets[:, 2:4]
            target_shoot = targets[:, 4:5]
            
            # Different losses for different actions
            movement_loss = nn.MSELoss()(pred_movement, target_movement)
            aim_loss = nn.MSELoss()(pred_aim, target_aim)
            shoot_loss = nn.BCELoss()(pred_shoot, target_shoot)
            
            # Weighted combination (adjust weights based on importance)
            total_loss = (
                2.0 * movement_loss +  # Movement is most important
                1.5 * aim_loss +       # Aiming is important
                1.0 * shoot_loss       # Shooting is less critical
            )
            
            return total_loss, {
                'movement_loss': movement_loss.item(),
                'aim_loss': aim_loss.item(), 
                'shoot_loss': shoot_loss.item()
            }
        
        return game_loss
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        loss_components = {'movement_loss': 0, 'aim_loss': 0, 'shoot_loss': 0}
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (images, targets) in enumerate(pbar):
            images, targets = images.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(images)
            loss, components = self.criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            for key, value in components.items():
                loss_components[key] += value
            
            # Update progress bar
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Mov': f'{components["movement_loss"]:.3f}',
                    'Aim': f'{components["aim_loss"]:.3f}',
                    'Shoot': f'{components["shoot_loss"]:.3f}'
                })
        
        avg_loss = total_loss / len(train_loader)
        avg_components = {k: v / len(train_loader) for k, v in loss_components.items()}
        
        return avg_loss, avg_components
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        loss_components = {'movement_loss': 0, 'aim_loss': 0, 'shoot_loss': 0}
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc='Validation'):
                images, targets = images.to(self.device), targets.to(self.device)
                
                predictions = self.model(images)
                loss, components = self.criterion(predictions, targets)
                
                total_loss += loss.item()
                for key, value in components.items():
                    loss_components[key] += value
        
        avg_loss = total_loss / len(val_loader)
        avg_components = {k: v / len(val_loader) for k, v in loss_components.items()}
        
        return avg_loss, avg_components
    
    def train(self, train_loader, val_loader, epochs=50, save_path='nuclear_throne_model.pth', start_epoch=0):
        """Full training loop"""
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 10
        
        print(f"Starting training for {epochs} epochs (starting from epoch {start_epoch})...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        for epoch in range(start_epoch, epochs):
            start_time = time.time()
            
            # Training
            train_loss, train_components = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_components = self.validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Track losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Print epoch results
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch+1}/{epochs} ({epoch_time:.1f}s)")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Components - Mov: {val_components['movement_loss']:.3f}, "
                  f"Aim: {val_components['aim_loss']:.3f}, "
                  f"Shoot: {val_components['shoot_loss']:.3f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss
                }, save_path)
                print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= max_patience:
                print(f"\nEarly stopping after {epoch+1} epochs")
                break
            
            # Unfreeze backbone after some epochs for fine-tuning
            if epoch == start_epoch + 10 and hasattr(self.model, 'freeze_backbone') and self.model.freeze_backbone:
                print("\n🔓 Unfreezing backbone for fine-tuning...")
                self.model.unfreeze_backbone()
                # Reduce learning rate for fine-tuning
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.1
        
        print(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")
        return best_val_loss

def load_cleaned_datasets(data_dir, data_format='regression'):
    """Load cleaned training datasets"""
    
    # Check if we have cleaned data
    dataset_info_path = os.path.join(data_dir, 'dataset_info.json')
    if os.path.exists(dataset_info_path):
        print("📁 Loading cleaned datasets...")
        with open(dataset_info_path, 'r') as f:
            dataset_info = json.load(f)
        print(f"Dataset created: {dataset_info['created_at']}")
        print(f"Total samples: {dataset_info['splits']['total']}")
        
        pytorch_dir = os.path.join(data_dir, 'pytorch')
        
        # Load train/val/test datasets
        datasets = {}
        for split in ['train', 'validation', 'test']:
            json_file = f"{split}_{data_format}.json"
            json_path = os.path.join(pytorch_dir, json_file)
            
            if os.path.exists(json_path):
                datasets[split] = json_path
                print(f"  Found {split} dataset: {json_file}")
            else:
                print(f"  Warning: {split} dataset not found: {json_file}")
        
        return datasets, data_dir
    
    else:
        raise FileNotFoundError(
            f"No cleaned dataset found in {data_dir}. "
            "Please run the cleaning script first: "
            "ts-node clean-training-data.ts <input_dir> <output_dir>"
        )

def analyze_dataset_statistics(data_dir):
    """Analyze and print dataset statistics"""
    dataset_info_path = os.path.join(data_dir, 'dataset_info.json')
    
    if os.path.exists(dataset_info_path):
        with open(dataset_info_path, 'r') as f:
            info = json.load(f)
        
        print("\n📊 Dataset Statistics:")
        print(f"  Total samples: {info['splits']['total']}")
        print(f"  Training: {info['splits']['train']} ({info['splits']['train']/info['splits']['total']*100:.1f}%)")
        print(f"  Validation: {info['splits']['validation']} ({info['splits']['validation']/info['splits']['total']*100:.1f}%)")
        print(f"  Test: {info['splits']['test']} ({info['splits']['test']/info['splits']['total']*100:.1f}%)")
        
        # Load training statistics if available
        train_regression_path = os.path.join(data_dir, 'pytorch', 'train_regression.json')
        if os.path.exists(train_regression_path):
            with open(train_regression_path, 'r') as f:
                train_data = json.load(f)
            
            if 'statistics' in train_data:
                stats = train_data['statistics']
                print(f"\n📈 Training Data Statistics:")
                if 'movement_stats' in stats:
                    mov_stats = stats['movement_stats']
                    print(f"  Movement - Mean: ({mov_stats.get('mean_x', 0):.3f}, {mov_stats.get('mean_y', 0):.3f})")
                    print(f"  Movement - Std:  ({mov_stats.get('std_x', 0):.3f}, {mov_stats.get('std_y', 0):.3f})")
                
                if 'shooting_rate' in stats:
                    print(f"  Shooting rate: {stats['shooting_rate']*100:.1f}%")
                
                if 'aim_stats' in stats:
                    aim_stats = stats['aim_stats']
                    print(f"  Aim center: ({aim_stats.get('mean_x', 0):.3f}, {aim_stats.get('mean_y', 0):.3f})")
    
    return info if os.path.exists(dataset_info_path) else None

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Train Nuclear Throne AI with ROCm')
    parser.add_argument('--data-dir', type=str, required=True, 
                       help='Path to cleaned training data directory')
    parser.add_argument('--model', type=str, default='efficientnet_b0',
                       choices=['efficientnet_b0', 'mobilenet_v3_small', 'resnet18'],
                       help='Model backbone to use')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=0, help='Batch size (0 for auto)')
    parser.add_argument('--data-format', type=str, default='regression',
                       choices=['regression', 'classification'],
                       help='Training data format')
    parser.add_argument('--save-path', type=str, default='nuclear_throne_ai_model.pth',
                       help='Path to save the trained model')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    
    args = parser.parse_args()
    
    # Setup ROCm
    device = setup_rocm()
    
    # Load and analyze datasets
    print(f"🔍 Loading datasets from: {args.data_dir}")
    dataset_paths, base_dir = load_cleaned_datasets(args.data_dir, args.data_format)
    dataset_info = analyze_dataset_statistics(args.data_dir)
    
    if not dataset_paths:
        print("❌ No datasets found!")
        return
    
    # Get image size from dataset info
    if dataset_info and 'image_info' in dataset_info:
        resolution = dataset_info['image_info']['resolution']
        width, height = map(int, resolution.split('x'))
        image_size = (height, width)
    else:
        image_size = (240, 320)  # Default Nuclear Throne resolution
    
    print(f"🖼️  Image size: {image_size}")
    
    # Data transforms
    train_transform, val_transform = create_data_transforms(image_size)
    
    # Load datasets
    datasets = {}
    data_loaders = {}
    
    for split, json_path in dataset_paths.items():
        print(f"📥 Loading {split} dataset...")
        
        transform = train_transform if split == 'train' else val_transform
        cache_images = True if split in ['validation', 'test'] else False  # Cache smaller datasets
        
        datasets[split] = NuclearThroneDataset(
            json_path=json_path,
            base_dir=base_dir,
            transform=transform,
            cache_images=cache_images,
            data_format=args.data_format
        )
    
    # Create model
    print(f"🤖 Creating model: {args.model}")
    if args.data_format == 'regression':
        num_outputs = 5  # movement_x, movement_y, aim_x, aim_y, shooting
        model = GameAIModel(num_outputs=num_outputs, backbone=args.model)
    else:
        # For classification, get number of classes from dataset
        with open(dataset_paths['train'], 'r') as f:
            train_data = json.load(f)
        num_outputs = train_data.get('num_classes', 10)
        model = GameAIModel(num_outputs=num_outputs, backbone=args.model)
    
    # Determine optimal batch size
    if args.batch_size == 0:
        batch_size = get_optimal_batch_size(device, model, 
                                          sample_input_size=(3, image_size[0], image_size[1]))
    else:
        batch_size = args.batch_size
    
    print(f"📦 Using batch size: {batch_size}")
    
    # Create data loaders
    for split, dataset in datasets.items():
        shuffle = (split == 'train')
        data_loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True if device.type == 'cuda' else False,
            persistent_workers=True,
            drop_last=(split == 'train')  # Drop last incomplete batch for training
        )
        print(f"  {split}: {len(dataset)} samples, {len(data_loaders[split])} batches")
    
    # Create trainer
    trainer = GameAITrainer(model, device, learning_rate=args.lr)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"📂 Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"  Resuming from epoch {start_epoch}")
    
    # Train model
    print(f"\n🚀 Starting training...")
    print(f"  Model: {args.model}")
    print(f"  Data format: {args.data_format}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {device}")
    
    best_loss = trainer.train(
        train_loader=data_loaders['train'],
        val_loader=data_loaders.get('validation', data_loaders.get('test')),
        epochs=args.epochs,
        save_path=args.save_path,
        start_epoch=start_epoch
    )
    
    # Test on test set if available
    if 'test' in data_loaders and 'test' not in ['validation']:
        print(f"\n🧪 Testing on test set...")
        test_loss, test_components = trainer.validate(data_loaders['test'])
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Components - Mov: {test_components['movement_loss']:.3f}, "
              f"Aim: {test_components['aim_loss']:.3f}, "
              f"Shoot: {test_components['shoot_loss']:.3f}")
    
    print(f"\n✅ Training complete! Best validation loss: {best_loss:.4f}")
    print(f"💾 Model saved to: {args.save_path}")

if __name__ == "__main__":
    main()