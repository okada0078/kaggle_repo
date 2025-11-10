# %% [markdown]
# ##### 🌾 CSIRO Biomass Prediction - Two-Stream ConvNeXt Inference
# 
# ### Reference: https://www.kaggle.com/code/none00000/lb-0-57-infer-model-code
# ### training: https://www.kaggle.com/code/takahitomizunobyts/convnext-training-analysis-notebook
# 
# 
# ## 📊 Performance
# - **Public LB Score:** 0.61
# - **Model:** ConvNeXt-Tiny with Two-Stream Architecture
# - **Ensemble:** 5-Fold + 3-View TTA
# 
# ---
# 
# ## 🎯 What's in This Notebook?
# 
# This inference notebook implements a **two-stream architecture** for predicting pasture biomass from top-view images. Key features:
# 
# - ✅ **Two-Stream CNN**: Processes left/right image halves independently
# - ✅ **Three-Head Regression**: Dedicated heads for Total, GDM, and Green biomass
# - ✅ **5-Fold Ensemble**: Robust predictions through cross-validation
# - ✅ **Test-Time Augmentation**: Original + HFlip + VFlip (3 views)
# - ✅ **Clean & Documented Code**: Easy to understand and modify
# 
# ---
# 
# ## 💬 Feedback & Discussion
# 
# If you find this notebook helpful:
# - 👍 **Please upvote** to support my work
# - 💬 **Leave a comment** with your questions or suggestions
# - 🔔 **Follow me** for more competitions and insights
# 
# **Questions? Issues?** Feel free to comment below, and I'll respond ASAP!
# 
# 

# %%
"""
CSIRO Biomass Competition - Inference Pipeline (TTA + Ensemble)
================================================================================
This script performs predictions on test data using trained models.

Pipeline Overview:
1. Test Data Preparation: Load CSV → Extract unique images
2. Model Loading: Load 5-Fold trained models
3. TTA Inference: 3 Views × 5-Fold Ensemble
4. Post-processing: 3 predictions → Reconstruct 5 targets
5. Submission Creation: Wide format → Long format conversion
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from collections import OrderedDict
import os
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
import cv2
from tqdm import tqdm


# ============================================================================
# Configuration Management
# ============================================================================

@dataclass
class InferenceConfig:
    """
    Data class for managing inference pipeline configuration.
    
    Items that must match training configuration:
    - model_name
    - img_size
    - target column names
    """
    
    # Path settings
    base_path: Path = Path('/kaggle/input/csiro-biomass')
    test_csv: Path = field(init=False)
    test_image_dir: Path = field(init=False)
    model_dir: Path = Path('/kaggle/input/csiro-exp3/convnext_exp3')
    submission_file: str = 'submission.csv'
    
    # Model settings (must match training)
    model_name: str = 'convnext_small'
    img_size: int = 1000
    
    # Device settings
    device: torch.device = field(default_factory=lambda: torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    ))
    
    # Inference settings
    batch_size: int = 1
    num_workers: int = 1
    n_folds: int = 5
    
    # Target settings (must match training)
    train_target_cols: list[str] = field(default_factory=lambda: [
        'Dry_Total_g', 'GDM_g', 'Dry_Green_g'
    ])
    
    all_target_cols: list[str] = field(default_factory=lambda: [
        'Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g'
    ])
    
    def __post_init__(self) -> None:
        """Construct paths after initialization"""
        self.test_csv = self.base_path / 'test.csv'
        self.test_image_dir = self.base_path / 'test'
    
    def display_info(self) -> None:
        """Display configuration information"""
        print(f"{'='*70}")
        print(f"Inference Configuration")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Backbone: {self.model_name}")
        print(f"Image Size: {self.img_size}x{self.img_size}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Ensemble: {self.n_folds}-Fold")
        print(f"TTA: 3 Views (Original, HFlip, VFlip)")
        print(f"{'='*70}\n")


# ============================================================================
# TTA Augmentation
# ============================================================================

class TTATransformFactory:
    """
    Factory class for generating Test Time Augmentation transforms.
    
    Provides 3 different views:
    1. Original (no augmentation)
    2. Horizontal flip
    3. Vertical flip
    """
    
    def __init__(self, img_size: int):
        """
        Args:
            img_size: Image size after resizing
        """
        self.img_size = img_size
        
        # Base transforms common to all views
        self.base_transforms = [
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]
    
    def get_tta_transforms(self) -> list[A.Compose]:
        """
        Generate 3 transform pipelines for TTA.
        
        Returns:
            List of 3 Albumentations Compose objects
            
        Why not: Not adding more TTA variations
            → Considering trade-off with inference time
        """
        # View 1: Original
        original = A.Compose([
            A.Resize(self.img_size, self.img_size),
            *self.base_transforms
        ])
        
        # View 2: Horizontal flip
        hflip = A.Compose([
            A.HorizontalFlip(p=1.0),
            A.Resize(self.img_size, self.img_size),
            *self.base_transforms
        ])
        
        # View 3: Vertical flip
        vflip = A.Compose([
            A.VerticalFlip(p=1.0),
            A.Resize(self.img_size, self.img_size),
            *self.base_transforms
        ])
        
        return [original, hflip, vflip]


# ============================================================================
# Dataset
# ============================================================================

class TestBiomassDataset(Dataset):
    """
    Two-stream dataset for testing.
    
    Accepts a specific transform pipeline for TTA and applies
    the same augmentation to both left and right images.
    
    Returns:
        tuple: (img_left, img_right)
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        transform_pipeline: A.Compose,
        image_dir: Path
    ):
        """
        Args:
            df: DataFrame containing image paths
            transform_pipeline: Augmentation pipeline to apply
            image_dir: Image directory path
        """
        self.df = df
        self.transform = transform_pipeline
        self.image_dir = image_dir
        self.image_paths = df['image_path'].values
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get one sample.
        
        Args:
            idx: Sample index
            
        Returns:
            (left_image, right_image)
            
        Why not: Not applying different augmentations to left/right as in training
            → During TTA, apply same transform to both images to preserve symmetry
        """
        img_path = self.image_paths[idx]
        full_path = self.image_dir / Path(img_path).name
        
        # Load image (return black image on error)
        image = cv2.imread(str(full_path))
        
        if image is None:
            print(f"Warning: Failed to load image: {full_path} → Returning black image")
            image = np.zeros((1000, 2000, 3), dtype=np.uint8)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Split into left and right
        height, width = image.shape[:2]
        mid_point = width // 2
        img_left = image[:, :mid_point]
        img_right = image[:, mid_point:]
        
        # Apply same transform to both
        img_left_tensor = self.transform(image=img_left)['image']
        img_right_tensor = self.transform(image=img_right)['image']
        
        return img_left_tensor, img_right_tensor


# ============================================================================
# Model
# ============================================================================

class BiomassModel(nn.Module):
    """
    Two-stream, three-head regression model.
    
    Exactly the same architecture as during training.
    """
    
    def __init__(self, model_name: str, pretrained: bool = False):
        """
        Args:
            model_name: timm model name
            pretrained: Always False (weights loaded separately)
        """
        super().__init__()
        
        # Shared backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )
        
        self.n_features = self.backbone.num_features
        self.n_combined = self.n_features * 2
        
        # Three dedicated heads
        self.head_total = self._create_head()
        self.head_gdm = self._create_head()
        self.head_green = self._create_head()
    
    def _create_head(self) -> nn.Sequential:
        """Generate MLP structure for a single head"""
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
        feat_left = self.backbone(img_left)
        feat_right = self.backbone(img_right)
        combined = torch.cat([feat_left, feat_right], dim=1)
        
        out_total = self.head_total(combined)
        out_gdm = self.head_gdm(combined)
        out_green = self.head_green(combined)
        
        return out_total, out_gdm, out_green


# ============================================================================
# Model Loader
# ============================================================================

class ModelLoader:
    """
    Class for loading trained models.
    
    Handles weights saved with DataParallel.
    """
    
    def __init__(self, config: InferenceConfig):
        """
        Args:
            config: Configuration object
        """
        self.config = config
    
    def load_fold_models(self) -> list[nn.Module]:
        """
        Load all 5-Fold trained models.
        
        Returns:
            List of models (each in eval mode on GPU)
            
        Raises:
            FileNotFoundError: If model file not found
        """
        print(f"\nLoading {self.config.n_folds} trained models...")
        
        models = []
        
        for fold in range(self.config.n_folds):
            model_path = self.config.model_dir / f'best_model_fold{fold}.pth'
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Initialize model
            model = BiomassModel(self.config.model_name, pretrained=False)
            
            # Load weights
            state_dict = torch.load(model_path, map_location=self.config.device)
            
            # Handle DataParallel (remove 'module.' prefix)
            state_dict = self._remove_dataparallel_prefix(state_dict)
            
            model.load_state_dict(state_dict)
            model.eval()  # Evaluation mode
            model.to(self.config.device)
            
            models.append(model)
            print(f"  ✓ Fold {fold} model loaded")
        
        print(f"✓ Successfully loaded {len(models)} models\n")
        return models
    
    @staticmethod
    def _remove_dataparallel_prefix(state_dict: dict) -> dict:
        """
        Remove 'module.' prefix from DataParallel-saved weights.
        
        Args:
            state_dict: Model weight dictionary
            
        Returns:
            Weight dictionary with prefix removed
            
        Why not: Not using try-except with direct load_state_dict
            → Explicitly handling prefix presence improves readability
        """
        if not any(k.startswith('module.') for k in state_dict.keys()):
            return state_dict  # No prefix
        
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            new_key = key.replace('module.', '')
            new_state_dict[new_key] = value
        
        return new_state_dict


# ============================================================================
# Inference Engine
# ============================================================================

class InferenceEngine:
    """
    Engine for executing TTA + Ensemble inference.
    """
    
    def __init__(
        self,
        models: list[nn.Module],
        config: InferenceConfig
    ):
        """
        Args:
            models: List of trained models (5-Fold)
            config: Configuration object
        """
        self.models = models
        self.config = config
    
    def predict_single_view(
        self,
        loader: DataLoader
    ) -> dict[str, np.ndarray]:
        """
        Predict with 5-Fold Ensemble for one TTA view.
        
        Args:
            loader: DataLoader (with specific TTA transform applied)
            
        Returns:
            {'total': [N], 'gdm': [N], 'green': [N]}
        """
        view_preds = {'total': [], 'gdm': [], 'green': []}
        
        with torch.no_grad():
            for img_left, img_right in tqdm(loader, desc="  Predicting", leave=False):
                img_left = img_left.to(self.config.device)
                img_right = img_right.to(self.config.device)
                
                # Collect predictions from 5 folds
                fold_preds = {'total': [], 'gdm': [], 'green': []}
                
                for model in self.models:
                    pred_total, pred_gdm, pred_green = model(img_left, img_right)
                    fold_preds['total'].append(pred_total.cpu())
                    fold_preds['gdm'].append(pred_gdm.cpu())
                    fold_preds['green'].append(pred_green.cpu())
                
                # Average across 5 folds
                avg_total = torch.mean(torch.stack(fold_preds['total']), dim=0)
                avg_gdm = torch.mean(torch.stack(fold_preds['gdm']), dim=0)
                avg_green = torch.mean(torch.stack(fold_preds['green']), dim=0)
                
                view_preds['total'].append(avg_total.numpy())
                view_preds['gdm'].append(avg_gdm.numpy())
                view_preds['green'].append(avg_green.numpy())
        
        # Concatenate batches
        return {
            k: np.concatenate(v).flatten()
            for k, v in view_preds.items()
        }
    
    def predict_with_tta(
        self,
        test_df: pd.DataFrame,
        tta_transforms: list[A.Compose]
    ) -> dict[str, np.ndarray]:
        """
        Execute final prediction with TTA + Ensemble.
        
        Args:
            test_df: Test data DataFrame
            tta_transforms: List of transforms for TTA
            
        Returns:
            {'total': [N], 'gdm': [N], 'green': [N]} (after TTA averaging)
        """
        print(f"\nStarting TTA inference: {len(tta_transforms)} Views × {self.config.n_folds} Folds")
        
        all_view_preds: list[dict[str, np.ndarray]] = []
        
        for i, transform in enumerate(tta_transforms):
            print(f"--- TTA View {i+1}/{len(tta_transforms)} ---")
            
            # Create Dataset/Loader for this view
            dataset = TestBiomassDataset(
                test_df,
                transform,
                self.config.test_image_dir
            )
            
            loader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=True
            )
            
            # 5-Fold Ensemble prediction
            view_preds = self.predict_single_view(loader)
            all_view_preds.append(view_preds)
            
            print(f"  ✓ View {i+1} completed")
        
        # TTA Ensemble (average across all views)
        print("\nCalculating TTA Ensemble (averaging all views)...")
        final_preds = {
            'total': np.mean([p['total'] for p in all_view_preds], axis=0),
            'gdm': np.mean([p['gdm'] for p in all_view_preds], axis=0),
            'green': np.mean([p['green'] for p in all_view_preds], axis=0)
        }
        
        print("✓ Inference completed\n")
        return final_preds


# ============================================================================
# Submission Creation
# ============================================================================

class SubmissionCreator:
    """
    Class for creating Kaggle submission CSV from predictions.
    """
    
    def __init__(self, config: InferenceConfig):
        """
        Args:
            config: Configuration object
        """
        self.config = config
    
    def create(
        self,
        predictions: dict[str, np.ndarray],
        test_df_long: pd.DataFrame,
        test_df_unique: pd.DataFrame
    ) -> None:
        """
        Create and save submission CSV from predictions.
        
        Args:
            predictions: {'total': [N], 'gdm': [N], 'green': [N]}
            test_df_long: Original test.csv (long format)
            test_df_unique: DataFrame of unique images
            
        Processing flow:
        1. Calculate 5 targets from 3 predictions
        2. Create wide format DataFrame
        3. Convert to long format (melt)
        4. Merge with sample_id
        5. Save CSV
        """
        print("Creating submission CSV...")
        
        # 1. Get 3 predictions
        pred_total = predictions['total']
        pred_gdm = predictions['gdm']
        pred_green = predictions['green']
        
        # 2. Calculate remaining 2 (clip negative values)
        pred_clover = np.maximum(0, pred_gdm - pred_green)
        pred_dead = np.maximum(0, pred_total - pred_gdm)
        
        # 3. Create wide format DataFrame
        preds_wide = pd.DataFrame({
            'image_path': test_df_unique['image_path'],
            'Dry_Green_g': pred_green,
            'Dry_Dead_g': pred_dead,
            'Dry_Clover_g': pred_clover,
            'GDM_g': pred_gdm,
            'Dry_Total_g': pred_total
        })
        
        # 4. Convert to long format (unpivot)
        preds_long = preds_wide.melt(
            id_vars=['image_path'],
            value_vars=self.config.all_target_cols,
            var_name='target_name',
            value_name='target'
        )
        
        # 5. Merge with original test.csv to get sample_id
        submission = pd.merge(
            test_df_long[['sample_id', 'image_path', 'target_name']],
            preds_long,
            on=['image_path', 'target_name'],
            how='left'
        )
        
        # 6. Format and save
        submission = submission[['sample_id', 'target']]
        submission.to_csv(self.config.submission_file, index=False)
        
        print(f"\n🎉 Submission saved: {self.config.submission_file}")
        print("\n--- First 5 rows ---")
        print(submission.head())
        print("\n--- Last 5 rows ---")
        print(submission.tail())


# ============================================================================
# Inference Pipeline
# ============================================================================

class InferencePipeline:
    """
    Class that orchestrates the entire inference pipeline.
    """
    
    def __init__(self, config: InferenceConfig):
        """
        Args:
            config: Configuration object
        """
        self.config = config
        self.model_loader = ModelLoader(config)
        self.tta_factory = TTATransformFactory(config.img_size)
        self.submission_creator = SubmissionCreator(config)
    
    def run(self) -> None:
        """
        Execute the entire inference pipeline.
        
        Processing flow:
        1. Load test data
        2. Load models (5-Fold)
        3. TTA inference (3 Views × 5 Folds)
        4. Create submission
        """
        print(f"\n{'='*70}")
        print(f"🚀 Starting Inference Pipeline")
        print(f"{'='*70}")
        
        try:
            # 1. Load test data
            test_df_long, test_df_unique = self._load_test_data()
            
            # 2. Load models
            models = self.model_loader.load_fold_models()
            
            # 3. TTA inference
            engine = InferenceEngine(models, self.config)
            tta_transforms = self.tta_factory.get_tta_transforms()
            predictions = engine.predict_with_tta(test_df_unique, tta_transforms)
            
            # 4. Create submission
            self.submission_creator.create(
                predictions,
                test_df_long,
                test_df_unique
            )
            
            print("\n✨ Inference Pipeline Completed ✨")
            
        except Exception as e:
            print(f"\n❌ Error occurred: {e}")
            raise
        
        finally:
            # Free memory
            gc.collect()
            torch.cuda.empty_cache()
    
    def _load_test_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load test data.
        
        Returns:
            (test_df_long, test_df_unique)
            - test_df_long: Original long format (with sample_id)
            - test_df_unique: Unique images only
            
        Raises:
            FileNotFoundError: If test.csv not found
        """
        print(f"\nLoading test data: {self.config.test_csv}")
        
        if not self.config.test_csv.exists():
            raise FileNotFoundError(f"test.csv not found: {self.config.test_csv}")
        
        test_df_long = pd.read_csv(self.config.test_csv)
        test_df_unique = test_df_long.drop_duplicates(
            subset=['image_path']
        ).reset_index(drop=True)
        
        print(f"  Long format: {len(test_df_long)} rows")
        print(f"  Unique images: {len(test_df_unique)} images\n")
        
        return test_df_long, test_df_unique


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == '__main__':
    # Initialize configuration
    config = InferenceConfig()
    config.display_info()
    
    # Run pipeline
    pipeline = InferencePipeline(config)
    pipeline.run()
    
    print("\n" + "="*70)
    print("🎊 Inference Pipeline Completed!")
    print("="*70)


