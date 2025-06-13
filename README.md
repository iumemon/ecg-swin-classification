# Enhancing Swin-B for Imbalanced ECG Classification: Softened Sampling, Class-Balanced Focal Loss, and ECG-Specific Augmentations

This project implements a deep learning model for ECG signal classification using the Swin Transformer architecture. The model is specifically designed to handle class imbalance and improve performance on atrial patterns.

## Project Overview

The model classifies ECG signals into five categories:
- A: Atrial patterns
- L: Left bundle branch block
- N: Normal
- R: Right bundle branch block
- V: Ventricular patterns

## Key Features

### 1. Model Architecture
- Based on Swin Transformer V2-B architecture with ImageNet pretrained weights
- Custom classification head for 5-class ECG classification
- Input image size: 224x224 pixels
- Enhanced feature extraction with improved attention mechanisms
- Optimized for ECG signal characteristics

### 2. Class Imbalance Handling
The project implements multiple strategies to address class imbalance:

#### 2.1 Custom Loss Function
- Enhanced Class-Balanced Focal Loss
  - Precomputed class weights (inverse-frequency ^0.5, normalized, with per-class tweaks)
  - Class-specific gamma values:
    - Lower gamma (1.5) for atrial class to reduce focus on hard examples
    - Higher gamma (2.5) for normal class to increase focus on hard examples
  - Margin-based regularization term specifically for atrial class discrimination
  - Fixed class weights computed at dataset initialization

#### 2.2 Data Sampling
- Weighted Random Sampling with frequency-based weighting
  - Inverse frequency weighting with smoothing
  - Weight normalization and clipping to [0.2, 0.8] range
  - Sampling with replacement for balanced training
  - Weights computed once at dataset initialization

#### 2.3 ECG-Specific Augmentations
- Atrial-specific augmentations:
  - P-wave morphology variations
  - PR interval adjustments
  - Baseline wander simulation
  - Signal quality variations
  - Gaussian noise with signal-aware masking
  - Realistic ECG noise addition
- Note: Normal-specific augmentations are defined but not currently used in the training pipeline

### 3. Training Features

#### 3.1 Optimization
- AdamW optimizer with weight decay (0.02)
- Cosine annealing learning rate with warm restarts
  - Initial learning rate: 3e-5
  - T_0: 5 epochs (first restart)
  - T_mult: 2 (doubles restart interval)
  - Minimum learning rate: 1% of initial lr
- Gradient clipping (max_norm=1.0)
- Comprehensive regularization

#### 3.2 Training Monitoring
- Per-class accuracy tracking
- Comprehensive metrics logging
- Training and validation curves
- Confusion matrix visualization
- Class distribution analysis
- Learning rate scheduling visualization
- Loss component analysis

#### 3.3 Checkpointing
- Automatic checkpoint saving
- Training resumption capability
- Best model preservation
- Detailed training metadata storage
- Model state and optimizer state preservation
- Learning rate scheduler state tracking

### 4. Evaluation
- Comprehensive test set evaluation
- Per-class performance metrics
- Confusion matrix generation
- Detailed classification reports
- Performance visualization
- Model confidence analysis
- Error case analysis

## Usage

### Training
```bash
python swin_b_improved_v2.py --train [options]
```

Options:
- `--epochs`: Number of training epochs (default: 30)
- `--batch-size`: Batch size (default: 64)
- `--lr`: Learning rate (default: 3e-5)
- `--weight-decay`: Weight decay (default: 0.02)
- `--resume`: Path to checkpoint for resuming training
- `--image-size`: Input image size (default: 224)
- `--num-workers`: Data loading workers (default: 4)
- `--seed`: Random seed (default: 42)
- `--checkpoint-dir`: Directory for saving checkpoints (default: 'checkpoints')

### Evaluation
```bash
python swin_b_improved_v2.py --evaluate-test [options]
```

Options:
- `--checkpoint`: Specific checkpoint to use
- `--test-results-dir`: Directory for test results (default: 'test_results')
- `--save-predictions`: Save individual predictions
- `--generate-plots`: Generate detailed performance plots

### Plot Regeneration
```bash
python swin_b_improved_v2.py --regenerate-plots [options]
```

Options:
- `--checkpoint`: Checkpoint to use for plot generation
- `--plot-types`: Types of plots to generate (default: all)
- `--output-dir`: Directory for saving plots

## Directory Structure
```
.
├── dataset/
│   ├── train/
│   │   ├── A/
│   │   ├── L/
│   │   ├── N/
│   │   ├── R/
│   │   └── V/
│   ├── validate/
│   └── test/
├── checkpoints/
│   ├── checkpoint_*.pt
│   └── metadata_*.json
├── test_results/
│   ├── test_results_*.json
│   └── test_confusion_matrix_*.png
└── swin_b_improved_v2.py
```

## Implementation Details

### Data Processing
- ImageNet normalization
- Atrial-specific augmentations (see section 2.3)
- Signal quality preservation
- Multi-channel support
- Advanced noise handling with signal-aware masking
- Note: Normal-specific augmentations are available but not currently used

### Model Architecture
- Swin Transformer V2-B backbone
- Custom classification head
- Pre-trained weights initialization
- Device-agnostic implementation
- Enhanced attention mechanisms
- Optimized for ECG characteristics

### Training Process
- Static weighted sampling with precomputed weights
- Fixed class weights in focal loss
- Learning rate scheduling with warm restarts
- Gradient management
- Comprehensive monitoring
- Advanced regularization
- Margin-based atrial class optimization

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Per-class performance
- Confusion matrix
- Model confidence scores
- Error analysis metrics

## Requirements
- Python 3.8+
- PyTorch 2.0+
- torchvision
- scikit-learn
- seaborn
- tqdm
- PIL
- OpenCV
- NumPy
- Matplotlib

## Notes
- The model is specifically optimized for atrial pattern recognition
- Class imbalance is handled through multiple complementary strategies
- Training can be resumed from any checkpoint
- Comprehensive evaluation metrics are provided
- All training parameters are configurable through command-line arguments

## Features

- **Advanced Architecture**: Uses Swin Transformer (Swin-B) with ImageNet pretrained weights
- **ECG-Specific Augmentations**: 
  - Atrial pattern-specific augmentations
  - P-wave morphology variations
  - PR interval adjustments
  - Baseline wander simulation
  - Realistic ECG noise addition
- **Class Imbalance Handling**:
  - Class-balanced focal loss
  - Weighted random sampling
  - Special emphasis on atrial patterns
- **Comprehensive Training Pipeline**:
  - Learning rate scheduling with warm restarts
  - Gradient clipping
  - Early stopping
  - Model checkpointing
- **Detailed Evaluation**:
  - Per-class metrics
  - Confusion matrix visualization
  - Training curves
  - Test set evaluation

## Dataset Structure

The dataset should be organized as follows:
```bash
dataset/
├── train/
│   ├── A/  # Atrial
│   ├── L/  # Left Bundle Branch Block
│   ├── N/  # Normal
│   ├── R/  # Right Bundle Branch Block
│   └── V/  # Ventricular
├── validate/
│   ├── A/
│   ├── L/
│   ├── N/
│   ├── R/
│   └── V/
└── test/
    ├── A/
    ├── L/
    ├── N/
    ├── R/
    └── V/
```

Each class directory should contain PNG images of ECG signals.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

Copyright (c) 2025 Irfanullah Memon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Citation

If you use this code in your research, please cite:
```bibtex
@software{memon_ecg_swin_2024,
  author = {Memon, Irfanullah},
  title = {Enhancing Swin-B for Imbalanced ECG Classification: Softened Sampling, Class-Balanced Focal Loss, and ECG-Specific Augmentations},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/iumemon/ecg-swin-classification},
  note = {An improved implementation of ECG classification using Swin Transformer V2 with enhanced class imbalance handling and atrial pattern recognition}
}
```

## Acknowledgments

- Swin Transformer paper and implementation
- ImageNet pretrained weights
