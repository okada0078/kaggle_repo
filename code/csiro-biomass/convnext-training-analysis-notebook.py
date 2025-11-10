#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
CSIRO Biomass Competition - Two-Stream Three-Head Regression Model with Visualization
================================================================================
This script implements a pipeline for predicting biomass from images with comprehensive visualization.

Pipeline Overview:
1. Data Preparation: Load CSV → Pivot → Stratified K-Fold split
2. Preprocessing: Image split (left/right) → Augmentation → Normalization
3. Model: Shared Backbone → Feature concatenation → 3 dedicated heads
4. Training: Two-stage learning (Freeze→Unfreeze) → Weighted loss → R² evaluation
5. Visualization: Learning curves, Fold comparison, Prediction scatter plots
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
import os
import gc
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
import cv2
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# Configuration Management
# ============================================================================

@dataclass
class Config:
    """
    Data class for managing pipeline-wide configuration.
    
    Centralizing all hyperparameters and constants ensures
    experiment reproducibility and facilitates configuration changes.
    """
    
    # Path settings
    base_path: Path = Path('/kaggle/input/csiro-biomass')
    train_csv: Path = field(init=False)
    image_dir: Path = field(init=False)
    output_dir: Path = Path('./results')  # Directory for saving visualizations
    
    # Model settings
    model_name: str = 'convnext_tiny'  # timm-compatible model name
    pretrained: bool = True
    img_size: int = 1024
    
    # Device settings
    device: torch.device = field(default_factory=lambda: torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    ))
    
    # Training settings
    batch_size: int = 4  # Adjust according to GPU memory
    epochs: int = 30
    learning_rate: float = 1e-4  # Learning rate for Stage 1
    finetune_lr: float = 1e-5     # Learning rate for Stage 2
    freeze_epochs: int = 10        # Number of epochs to freeze backbone
    num_workers: int = 2
    
    # Cross-validation settings
    n_folds: int = 5
    random_state: int = 42
    
    # Target settings
    # Three targets used for training (loss calculation)
    train_target_cols: list[str] = field(default_factory=lambda: [
        'Dry_Total_g', 'GDM_g', 'Dry_Green_g'
    ])
    
    # Five targets used for evaluation (R² score calculation)
    all_target_cols: list[str] = field(default_factory=lambda: [
        'Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g'
    ])
    
    # Loss function weights (corresponding to 3 training targets)
    loss_weights: dict[str, float] = field(default_factory=lambda: {
        'total_loss': 0.5,
        'gdm_loss': 0.2,
        'green_loss': 0.1
    })
    
    # R² score weights (corresponding to 5 evaluation targets)
    r2_weights: list[float] = field(default_factory=lambda: [
        0.1, 0.1, 0.1, 0.2, 0.5
    ])
    
    def __post_init__(self) -> None:
        """Construct paths after initialization and create output directory"""
        self.train_csv = self.base_path / 'train.csv'
        self.image_dir = self.base_path / 'train'
        
        # Create output directory for visualizations
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def display_info(self) -> None:
        """Display configuration information"""
        print(f"{'='*70}")
        print(f"Configuration")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Backbone: {self.model_name}")
        print(f"Image Size: {self.img_size}x{self.img_size}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Total Epochs: {self.epochs}")
        print(f"  - Stage 1 (Freeze): Epoch 1-{self.freeze_epochs} (LR={self.learning_rate})")
        print(f"  - Stage 2 (Finetune): Epoch {self.freeze_epochs+1}-{self.epochs} (LR={self.finetune_lr})")
        print(f"Cross-Validation: {self.n_folds}-Fold")
        print(f"Output Directory: {self.output_dir}")
        print(f"{'='*70}\n")


# ============================================================================
# Data Preparation
# ============================================================================

class DataPreparator:
    """
    Class responsible for data loading and preprocessing.
    
    Main functions:
    - Load and pivot CSV
    - Stratified K-Fold splitting
    """
    
    def __init__(self, config: Config):
        """
        Args:
            config: Configuration object
        """
        self.config = config
        self.df_wide: Optional[pd.DataFrame] = None
        
    def load_and_pivot(self) -> pd.DataFrame:
        """
        Load CSV and convert from long format to wide format.
        
        Returns:
            Wide-format DataFrame (each row is one image, each column is one target)
            
        Why not: Using pivot() instead of pivot_table()
            → Image paths are guaranteed to be unique
        """
        print(f"Loading CSV: {self.config.train_csv}")
        
        try:
            df_long = pd.read_csv(self.config.train_csv)
            print(f"Long format: {len(df_long)} rows")
            
            # Pivot transformation: image_path × target_name → values
            df_wide = df_long.pivot(
                index='image_path',
                columns='target_name',
                values='target'
            ).reset_index()
            
            df_wide.columns.name = None  # Clean up column names
            print(f"Wide format: {len(df_wide)} rows × {len(df_wide.columns)} columns")
            print(f"\nFirst 5 rows:\n{df_wide.head()}\n")
            
            self.df_wide = df_wide
            return df_wide
            
        except FileNotFoundError:
            print(f"Error: {self.config.train_csv} not found")
            # Return dummy DataFrame on error (prevent downstream crashes)
            return pd.DataFrame(columns=['image_path'] + self.config.all_target_cols)
    
    def create_stratified_folds(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign fold numbers for cross-validation using Stratified K-Fold.
        
        Args:
            df: Wide-format DataFrame
            
        Returns:
            DataFrame with 'fold' column added
            
        Why not: Using StratifiedKFold instead of GroupKFold
            → For regression, stratify after binning to maintain target distribution
        """
        print(f"\nPreparing {self.config.n_folds}-Fold Cross-Validation...")
        
        df = df.copy()
        df['fold'] = -1
        
        # Bin targets (continuous → discrete)
        # Determine number of bins using Sturges' formula
        num_bins = min(10, int(np.floor(1 + np.log2(len(df)))))
        print(f"Stratifying Dry_Total_g into {num_bins} bins")
        
        df['total_bin'] = pd.cut(
            df['Dry_Total_g'], 
            bins=num_bins, 
            labels=False,
            duplicates='drop'  # Remove duplicate edges
        )
        
        # Stratified K-Fold split
        skf = StratifiedKFold(
            n_splits=self.config.n_folds,
            shuffle=True,
            random_state=self.config.random_state
        )
        
        for fold_num, (_, valid_idx) in enumerate(skf.split(df, df['total_bin'])):
            df.loc[valid_idx, 'fold'] = fold_num
        
        # Remove binning column (no longer needed)
        df = df.drop(columns=['total_bin'])
        
        print("\nFold distribution:")
        print(df['fold'].value_counts().sort_index())
        
        self.df_wide = df
        return df


# ============================================================================
# Data Augmentation
# ============================================================================

class AugmentationFactory:
    """
    Factory class for generating Albumentations pipelines.
    
    Provides different pipelines for training and validation.
    """
    
    def __init__(self, img_size: int):
        """
        Args:
            img_size: Image size after resizing
        """
        self.img_size = img_size
    
    def get_train_transforms(self) -> A.Compose:
        """
        Augmentation pipeline for training.
        
        Returns:
            Albumentations Compose object
            
        Why not: Not adding stronger augmentations
            → Balancing overfitting risk and training time
        """
        return A.Compose([
            # Geometric transforms
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            
            # Color transforms
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.75
            ),
            
            # Resize and normalize
            A.Resize(self.img_size, self.img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def get_valid_transforms(self) -> A.Compose:
        """
        Pipeline for validation (no augmentation).
        
        Returns:
            Albumentations Compose object
        """
        return A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


# ============================================================================
# Dataset
# ============================================================================

class BiomassDataset(Dataset):
    """
    Custom Dataset for two-stream architecture.
    
    Splits 2000x1000 images into left and right halves,
    applying independent augmentation to each.
    
    Returns:
        tuple: (img_left, img_right, train_targets, all_targets)
            - img_left: Left image tensor [C, H, W]
            - img_right: Right image tensor [C, H, W]
            - train_targets: Training targets [3]
            - all_targets: Evaluation targets [5]
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        transforms_fn: Callable[[], A.Compose],
        image_dir: Path,
        train_target_cols: list[str],
        all_target_cols: list[str]
    ):
        """
        Args:
            df: DataFrame containing image paths and targets
            transforms_fn: Function returning augmentation pipeline
            image_dir: Image directory path
            train_target_cols: Training target column names
            all_target_cols: Evaluation target column names
        """
        self.df = df
        self.transforms_fn = transforms_fn
        self.image_dir = image_dir
        
        # Convert to numpy arrays for fast access
        self.image_paths = df['image_path'].values
        self.train_targets = df[train_target_cols].values
        self.all_targets = df[all_target_cols].values
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(
        self, 
        idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get one sample.
        
        Args:
            idx: Sample index
            
        Returns:
            (left_image, right_image, train_targets, eval_targets)
            
        Why not: Not pre-splitting and saving images
            → Considering trade-off between storage and I/O time
        """
        # Get image path and targets
        img_path = self.image_paths[idx]
        train_target = self.train_targets[idx]
        all_target = self.all_targets[idx]
        
        # Load image
        full_path = self.image_dir / Path(img_path).name
        image = cv2.imread(str(full_path))
        
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {full_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Split into left and right (2000x1000 → 2 x 1000x1000)
        height, width = image.shape[:2]
        mid_point = width // 2
        img_left = image[:, :mid_point]
        img_right = image[:, mid_point:]
        
        # Apply augmentation independently
        # Why not: Not reusing the same transform
        #   → Applying different augmentations to left/right improves data diversity
        transform_left = self.transforms_fn()
        transform_right = self.transforms_fn()
        
        img_left_tensor = transform_left(image=img_left)['image']
        img_right_tensor = transform_right(image=img_right)['image']
        
        # Convert targets to tensors
        train_target_tensor = torch.tensor(train_target, dtype=torch.float32)
        all_target_tensor = torch.tensor(all_target, dtype=torch.float32)
        
        return img_left_tensor, img_right_tensor, train_target_tensor, all_target_tensor


# ============================================================================
# Model
# ============================================================================

class BiomassModel(nn.Module):
    """
    Two-stream, three-head regression model.
    
    Architecture:
    1. Shared Backbone (ConvNeXt etc.) extracts features from left/right images
    2. Concatenate two feature vectors
    3. Three dedicated MLP heads predict each target
    """
    
    def __init__(self, model_name: str, pretrained: bool):
        """
        Args:
            model_name: timm model name
            pretrained: Whether to use ImageNet pretrained weights
        """
        super().__init__()
        
        # Shared Backbone (no classification layer, with Global Average Pooling)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,      # Remove classification layer
            global_pool='avg'   # Add GAP
        )
        
        # Get feature dimension
        self.n_features = self.backbone.num_features
        self.n_combined = self.n_features * 2  # Dimension after concatenation
        
        # Three dedicated heads
        # Why not: Not using a single shared head
        #   → Dedicated heads improve accuracy as each target has different characteristics
        self.head_total = self._create_head()
        self.head_gdm = self._create_head()
        self.head_green = self._create_head()
    
    def _create_head(self) -> nn.Sequential:
        """
        Generate MLP structure for a single head.
        
        Returns:
            Two-layer MLP (with ReLU + Dropout in middle layer)
        """
        return nn.Sequential(
            nn.Linear(self.n_combined, self.n_combined // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.n_combined // 2, 1)
        )
    
    def forward(
        self, 
        img_left: torch.Tensor, 
        img_right: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            img_left: Left image [B, C, H, W]
            img_right: Right image [B, C, H, W]
            
        Returns:
            (total_pred, gdm_pred, green_pred) each [B, 1]
        """
        # Feature extraction (shared backbone)
        feat_left = self.backbone(img_left)    # [B, n_features]
        feat_right = self.backbone(img_right)  # [B, n_features]
        
        # Concatenate features
        combined = torch.cat([feat_left, feat_right], dim=1)  # [B, n_combined]
        
        # Predict with each head
        out_total = self.head_total(combined)
        out_gdm = self.head_gdm(combined)
        out_green = self.head_green(combined)
        
        return out_total, out_gdm, out_green


# ============================================================================
# Loss Function
# ============================================================================

class WeightedBiomassLoss(nn.Module):
    """
    Weighted loss for three targets.
    
    Weight losses according to importance of each target,
    using SmoothL1Loss for robust learning against outliers.
    """
    
    def __init__(self, loss_weights: dict[str, float]):
        """
        Args:
            loss_weights: Weights for each target
        """
        super().__init__()
        self.criterion = nn.SmoothL1Loss()  # A variant of Huber loss
        self.weights = loss_weights
    
    def forward(
        self,
        predictions: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate weighted loss.
        
        Args:
            predictions: (total, gdm, green) each [B, 1]
            targets: [B, 3] (in order: total, gdm, green)
            
        Returns:
            Weighted total loss
        """
        pred_total, pred_gdm, pred_green = predictions
        
        # Decompose targets
        true_total = targets[:, 0:1]  # Maintain [B, 1] shape
        true_gdm = targets[:, 1:2]
        true_green = targets[:, 2:3]
        
        # Calculate each loss
        loss_total = self.criterion(pred_total, true_total)
        loss_gdm = self.criterion(pred_gdm, true_gdm)
        loss_green = self.criterion(pred_green, true_green)
        
        # Weighted sum
        total_loss = (
            self.weights['total_loss'] * loss_total +
            self.weights['gdm_loss'] * loss_gdm +
            self.weights['green_loss'] * loss_green
        )
        
        return total_loss


# ============================================================================
# Evaluation Metrics
# ============================================================================

class CompetitionScorer:
    """
    Class for calculating competition evaluation metric (weighted R²).
    
    Reconstructs five targets from three predictions,
    then calculates weighted average of R² scores for each target.
    """
    
    def __init__(self, r2_weights: list[float]):
        """
        Args:
            r2_weights: R² weights for five targets
        """
        self.r2_weights = np.array(r2_weights)
    
    def calculate_score(
        self,
        preds_dict: dict[str, np.ndarray],
        targets_5: np.ndarray
    ) -> float:
        """
        Calculate weighted R² score.
        
        Args:
            preds_dict: {'total': [N], 'gdm': [N], 'green': [N]}
            targets_5: [N, 5] (in order: green, dead, clover, gdm, total)
            
        Returns:
            Weighted R² score
            
        Why not: Not using simple MSE
            → Following competition evaluation rules
        """
        # Get predictions
        pred_total = preds_dict['total']
        pred_gdm = preds_dict['gdm']
        pred_green = preds_dict['green']
        
        # Estimate remaining two (clip negative values)
        pred_clover = np.maximum(0, pred_gdm - pred_green)
        pred_dead = np.maximum(0, pred_total - pred_gdm)
        
        # Combine five predictions (green, dead, clover, gdm, total)
        y_preds = np.stack([
            pred_green, pred_dead, pred_clover, pred_gdm, pred_total
        ], axis=1)
        
        # Calculate R² for each target
        r2_scores = r2_score(targets_5, y_preds, multioutput='raw_values')
        
        # Weighted sum
        weighted_score = np.sum(r2_scores * self.r2_weights)
        
        return float(weighted_score)
    
    def calculate_individual_scores(
        self,
        preds_dict: dict[str, np.ndarray],
        targets_5: np.ndarray
    ) -> dict[str, float]:
        """
        Calculate individual R² scores for each target.
        
        Args:
            preds_dict: {'total': [N], 'gdm': [N], 'green': [N]}
            targets_5: [N, 5] (in order: green, dead, clover, gdm, total)
            
        Returns:
            Dictionary of R² scores for each target
        """
        # Get predictions
        pred_total = preds_dict['total']
        pred_gdm = preds_dict['gdm']
        pred_green = preds_dict['green']
        
        # Estimate remaining two (clip negative values)
        pred_clover = np.maximum(0, pred_gdm - pred_green)
        pred_dead = np.maximum(0, pred_total - pred_gdm)
        
        # Combine five predictions (green, dead, clover, gdm, total)
        y_preds = np.stack([
            pred_green, pred_dead, pred_clover, pred_gdm, pred_total
        ], axis=1)
        
        # Calculate R² for each target
        r2_scores = r2_score(targets_5, y_preds, multioutput='raw_values')
        
        target_names = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
        
        return {name: float(score) for name, score in zip(target_names, r2_scores)}


# ============================================================================
# Training History Tracker
# ============================================================================

@dataclass
class TrainingHistory:
    """
    Data class for tracking training history.
    
    Stores losses, scores, and predictions for visualization.
    """
    train_losses: list[float] = field(default_factory=list)
    valid_losses: list[float] = field(default_factory=list)
    valid_scores: list[float] = field(default_factory=list)
    
    # For final validation predictions
    final_preds: Optional[dict[str, np.ndarray]] = None
    final_targets: Optional[np.ndarray] = None
    
    def add_epoch(
        self,
        train_loss: float,
        valid_loss: float,
        valid_score: float
    ) -> None:
        """Add one epoch's metrics."""
        self.train_losses.append(train_loss)
        self.valid_losses.append(valid_loss)
        self.valid_scores.append(valid_score)
    
    def set_final_predictions(
        self,
        preds: dict[str, np.ndarray],
        targets: np.ndarray
    ) -> None:
        """Store final predictions for scatter plots."""
        self.final_preds = preds
        self.final_targets = targets


# ============================================================================
# Training and Validation Loops
# ============================================================================

class Trainer:
    """
    Class for managing training and validation loops with history tracking.
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        device: torch.device,
        scorer: CompetitionScorer
    ):
        """
        Args:
            model: Model to train
            criterion: Loss function
            device: Device (CPU/GPU)
            scorer: Evaluation metric calculator
        """
        self.model = model
        self.criterion = criterion
        self.device = device
        self.scorer = scorer
        self.history = TrainingHistory()
    
    def train_one_epoch(
        self,
        loader: DataLoader,
        optimizer: optim.Optimizer
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            loader: Training DataLoader
            optimizer: Optimizer
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(loader, desc="Training", leave=False)
        
        for img_left, img_right, train_targets, _ in pbar:
            # Transfer to device
            img_left = img_left.to(self.device)
            img_right = img_right.to(self.device)
            targets = train_targets.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(img_left, img_right)
            
            # Calculate loss
            loss = self.criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Record
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.4f}')
        
        return epoch_loss / len(loader)
    
    def validate_one_epoch(
        self,
        loader: DataLoader,
        store_predictions: bool = False
    ) -> tuple[float, float]:
        """
        Validate for one epoch.
        
        Args:
            loader: Validation DataLoader
            store_predictions: Whether to store predictions for visualization
            
        Returns:
            (average loss for epoch, R² score)
        """
        self.model.eval()
        epoch_loss = 0.0
        
        # Collect predictions and targets
        all_preds = {'total': [], 'gdm': [], 'green': []}
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(loader, desc="Validating", leave=False)
            
            for img_left, img_right, train_targets, all_targets_5 in pbar:
                img_left = img_left.to(self.device)
                img_right = img_right.to(self.device)
                train_targets = train_targets.to(self.device)
                
                # Forward pass
                pred_total, pred_gdm, pred_green = self.model(img_left, img_right)
                
                # Calculate loss
                predictions = (pred_total, pred_gdm, pred_green)
                loss = self.criterion(predictions, train_targets)
                epoch_loss += loss.item()
                
                # Collect predictions (convert to numpy on CPU)
                all_preds['total'].append(pred_total.cpu().numpy())
                all_preds['gdm'].append(pred_gdm.cpu().numpy())
                all_preds['green'].append(pred_green.cpu().numpy())
                all_targets.append(all_targets_5.cpu().numpy())
        
        # Concatenate batches
        preds_np = {
            k: np.concatenate(v).flatten() 
            for k, v in all_preds.items()
        }
        targets_np = np.concatenate(all_targets)
        
        # Store predictions if requested
        if store_predictions:
            self.history.set_final_predictions(preds_np, targets_np)
        
        # Calculate R² score
        score = self.scorer.calculate_score(preds_np, targets_np)
        
        avg_loss = epoch_loss / len(loader)
        return avg_loss, score


# ============================================================================
# Visualization
# ============================================================================

class TrainingVisualizer:
    """
    Class for creating comprehensive training visualizations.
    
    Generates and saves:
    1. Learning curves (loss and R² over epochs)
    2. Fold comparison (bar chart of scores across folds)
    3. Prediction scatter plots (predicted vs. actual for all targets)
    """
    
    def __init__(self, output_dir: Path, target_names: list[str]):
        """
        Args:
            output_dir: Directory to save plots
            target_names: List of target names for labeling
        """
        self.output_dir = output_dir
        self.target_names = target_names
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10
    
    def plot_learning_curves(
        self,
        history: TrainingHistory,
        fold: int,
        freeze_epoch: int
    ) -> None:
        """
        Plot learning curves showing loss and R² score over epochs.
        
        Args:
            history: Training history object
            fold: Fold number
            freeze_epoch: Epoch where backbone was unfrozen
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(history.train_losses) + 1)
        
        # Plot 1: Loss curves
        axes[0].plot(epochs, history.train_losses, label='Train Loss', marker='o', linewidth=2)
        axes[0].plot(epochs, history.valid_losses, label='Valid Loss', marker='s', linewidth=2)
        axes[0].axvline(x=freeze_epoch, color='red', linestyle='--', 
                        label=f'Unfreeze (Epoch {freeze_epoch})', alpha=0.7)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'Fold {fold}: Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: R² score curve
        axes[1].plot(epochs, history.valid_scores, label='Valid R² Score', 
                     marker='D', color='green', linewidth=2)
        axes[1].axvline(x=freeze_epoch, color='red', linestyle='--', 
                        label=f'Unfreeze (Epoch {freeze_epoch})', alpha=0.7)
        axes[1].axhline(y=max(history.valid_scores), color='orange', linestyle=':', 
                        label=f'Best R² = {max(history.valid_scores):.4f}', alpha=0.7)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('R² Score')
        axes[1].set_title(f'Fold {fold}: Validation R² Score')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / f'fold{fold}_learning_curves.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Saved learning curves: {save_path}")
    
    def plot_prediction_scatter(
        self,
        history: TrainingHistory,
        fold: int,
        scorer: CompetitionScorer
    ) -> None:
        """
        Plot scatter plots of predicted vs. actual values for all targets.
        
        Args:
            history: Training history with final predictions
            fold: Fold number
            scorer: Scorer to calculate individual R² scores
        """
        if history.final_preds is None or history.final_targets is None:
            print(f"  ⚠️  No predictions stored for fold {fold}")
            return
        
        # Reconstruct all 5 predictions
        pred_total = history.final_preds['total']
        pred_gdm = history.final_preds['gdm']
        pred_green = history.final_preds['green']
        pred_clover = np.maximum(0, pred_gdm - pred_green)
        pred_dead = np.maximum(0, pred_total - pred_gdm)
        
        predictions = [pred_green, pred_dead, pred_clover, pred_gdm, pred_total]
        targets = history.final_targets
        
        # Calculate individual R² scores
        individual_r2 = scorer.calculate_individual_scores(
            history.final_preds,
            history.final_targets
        )
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (pred, target_name) in enumerate(zip(predictions, self.target_names)):
            ax = axes[idx]
            true_vals = targets[:, idx]
            
            # Scatter plot
            ax.scatter(true_vals, pred, alpha=0.5, s=20)
            
            # Perfect prediction line
            min_val = min(true_vals.min(), pred.min())
            max_val = max(true_vals.max(), pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 
                   'r--', linewidth=2, label='Perfect Prediction')
            
            # Labels and title
            ax.set_xlabel('Actual Value (g)')
            ax.set_ylabel('Predicted Value (g)')
            r2 = individual_r2[target_name]
            ax.set_title(f'{target_name}\nR² = {r2:.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remove extra subplot
        axes[5].axis('off')
        
        plt.suptitle(f'Fold {fold}: Prediction vs. Actual Scatter Plots', 
                     fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        save_path = self.output_dir / f'fold{fold}_scatter_plots.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Saved scatter plots: {save_path}")
    
    def plot_fold_comparison(
        self,
        fold_scores: dict[int, float]
    ) -> None:
        """
        Plot bar chart comparing R² scores across all folds.
        
        Args:
            fold_scores: Dictionary mapping fold number to best R² score
        """
        folds = sorted(fold_scores.keys())
        scores = [fold_scores[f] for f in folds]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Bar plot
        bars = ax.bar(folds, scores, color='skyblue', edgecolor='navy', linewidth=1.5)
        
        # Add value labels on bars
        for fold, score, bar in zip(folds, scores, bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                   f'{score:.4f}',
                   ha='center', va='bottom', fontweight='bold')
        
        # Mean line
        ax.axhline(y=mean_score, color='red', linestyle='--', linewidth=2,
                  label=f'Mean R² = {mean_score:.4f} ± {std_score:.4f}')
        
        # Labels and title
        ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
        ax.set_ylabel('Best R² Score', fontsize=12, fontweight='bold')
        ax.set_title('Cross-Validation Performance: R² Score Comparison Across Folds', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(folds)
        ax.set_xticklabels([f'Fold {f}' for f in folds])
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = self.output_dir / 'fold_comparison.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        print(f"\n📊 Saved fold comparison: {save_path}")
        print(f"📈 Overall CV Score: {mean_score:.4f} ± {std_score:.4f}")


# ============================================================================
# Two-Stage Training Pipeline with Visualization
# ============================================================================

class TwoStageTrainingPipeline:
    """
    Pipeline executing two-stage training: Freeze→Unfreeze with visualization.
    
    Stage 1: Freeze backbone and train heads only
    Stage 2: Fine-tune entire model
    """
    
    def __init__(self, config: Config):
        """
        Args:
            config: Configuration object
        """
        self.config = config
        self.data_prep = DataPreparator(config)
        self.aug_factory = AugmentationFactory(config.img_size)
        self.scorer = CompetitionScorer(config.r2_weights)
        self.visualizer = TrainingVisualizer(config.output_dir, config.all_target_cols)
        
        # Store fold scores for comparison
        self.fold_scores: dict[int, float] = {}
    
    def run_fold(self, fold: int) -> None:
        """
        Execute training for one fold with visualization.
        
        Args:
            fold: Fold number
        """
        print(f"\n{'='*70}")
        print(f"🚀 Starting Fold {fold} Training (Two-Stage)")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        # Prepare data
        df = self.data_prep.df_wide
        train_df = df[df['fold'] != fold].reset_index(drop=True)
        valid_df = df[df['fold'] == fold].reset_index(drop=True)
        
        print(f"Training data: {len(train_df)} images | Validation data: {len(valid_df)} images")
        
        # Create datasets
        train_dataset = BiomassDataset(
            train_df,
            self.aug_factory.get_train_transforms,
            self.config.image_dir,
            self.config.train_target_cols,
            self.config.all_target_cols
        )
        
        valid_dataset = BiomassDataset(
            valid_df,
            self.aug_factory.get_valid_transforms,
            self.config.image_dir,
            self.config.train_target_cols,
            self.config.all_target_cols
        )
        
        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.config.batch_size * 2,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        # Initialize model
        model_base = BiomassModel(
            self.config.model_name,
            self.config.pretrained
        )
        
        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs (DataParallel)")
            model = nn.DataParallel(model_base)
        else:
            model = model_base
        
        model.to(self.config.device)
        
        # Loss function
        criterion = WeightedBiomassLoss(self.config.loss_weights).to(self.config.device)
        
        # Trainer instance
        trainer = Trainer(model, criterion, self.config.device, self.scorer)
        
        # ===== Stage 1: Freeze Backbone =====
        print(f"\n--- Stage 1: Backbone Frozen (Epoch 1-{self.config.freeze_epochs}) ---")
        self._freeze_backbone(model)
        
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.config.learning_rate
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=2
        )
        
        best_score = -float('inf')
        
        for epoch in range(1, self.config.freeze_epochs + 1):
            print(f"\nEpoch {epoch}/{self.config.epochs} (Stage 1)")
            
            train_loss = trainer.train_one_epoch(train_loader, optimizer)
            valid_loss, score = trainer.validate_one_epoch(valid_loader)
            
            # Record history
            trainer.history.add_epoch(train_loss, valid_loss, score)
            
            scheduler.step(valid_loss)
            
            print(f"Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f} | R²: {score:.4f}")
            
            if score > best_score:
                best_score = score
                self._save_model(model, fold)
                print("✨ R² score improved! Saving model")
        
        # ===== Stage 2: Fine-tune Entire Model =====
        print(f"\n--- Stage 2: Full Fine-tuning (Epoch {self.config.freeze_epochs+1}-{self.config.epochs}) ---")
        self._unfreeze_backbone(model)
        
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config.finetune_lr
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, patience=3
        )
        
        for epoch in range(self.config.freeze_epochs + 1, self.config.epochs + 1):
            print(f"\nEpoch {epoch}/{self.config.epochs} (Stage 2)")
            
            train_loss = trainer.train_one_epoch(train_loader, optimizer)
            
            # Store predictions on last epoch for visualization
            store_preds = (epoch == self.config.epochs)
            valid_loss, score = trainer.validate_one_epoch(valid_loader, store_predictions=store_preds)
            
            # Record history
            trainer.history.add_epoch(train_loss, valid_loss, score)
            
            scheduler.step(valid_loss)
            
            print(f"Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f} | R²: {score:.4f}")
            
            if score > best_score:
                best_score = score
                self._save_model(model, fold)
                print("✨ R² score improved! Saving model")
        
        # Store fold score
        self.fold_scores[fold] = best_score
        
        # Generate visualizations for this fold
        print(f"\n📊 Generating visualizations for Fold {fold}...")
        self.visualizer.plot_learning_curves(trainer.history, fold, self.config.freeze_epochs)
        self.visualizer.plot_prediction_scatter(trainer.history, fold, self.scorer)
        
        # Finish
        elapsed = (time.time() - start_time) / 60
        print(f"\n🎉 Fold {fold} completed ({elapsed:.2f} minutes)")
        print(f"Best R² score: {best_score:.4f}")
        
        # Free memory
        del model, train_loader, valid_loader
        gc.collect()
        torch.cuda.empty_cache()
    
    def _freeze_backbone(self, model: nn.Module) -> None:
        """Freeze backbone parameters"""
        backbone = model.module.backbone if isinstance(model, nn.DataParallel) else model.backbone
        for param in backbone.parameters():
            param.requires_grad = False
    
    def _unfreeze_backbone(self, model: nn.Module) -> None:
        """Unfreeze backbone parameters"""
        backbone = model.module.backbone if isinstance(model, nn.DataParallel) else model.backbone
        for param in backbone.parameters():
            param.requires_grad = True
    
    def _save_model(self, model: nn.Module, fold: int) -> None:
        """Save model to output directory"""
        state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        save_path = self.config.output_dir / f'best_model_fold{fold}.pth'
        torch.save(state_dict, save_path)
    
    def run_all_folds(self) -> None:
        """Execute training for all folds and generate summary visualization"""
        # Prepare data
        self.data_prep.load_and_pivot()
        self.data_prep.create_stratified_folds(self.data_prep.df_wide)
        
        # Train each fold
        for fold in range(self.config.n_folds):
            try:
                self.run_fold(fold)
            except Exception as e:
                print(f"\n❌ Error occurred in Fold {fold}: {e}")
                gc.collect()
                torch.cuda.empty_cache()
                raise
        
        # Generate fold comparison plot
        print(f"\n{'='*70}")
        print("📊 Generating cross-validation summary...")
        print(f"{'='*70}")
        self.visualizer.plot_fold_comparison(self.fold_scores)


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == '__main__':
    # Initialize configuration
    config = Config()
    config.display_info()
    
    # Execute pipeline
    pipeline = TwoStageTrainingPipeline(config)
    pipeline.run_all_folds()
    
    print("\n" + "="*70)
    print("🎊 All Fold Training Completed!")
    print("="*70)


# In[ ]:




