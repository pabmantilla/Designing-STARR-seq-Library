#!/usr/bin/env python3
"""
EvoAug2 Ensemble Training Script

This script trains an ensemble of 10 models using the same two-stage EvoAug2 approach
as the original script, but with different random seeds to create model diversity.

Each model in the ensemble goes through:
1. **Stage 1**: Training with evolution-inspired augmentations
2. **Stage 2**: Fine-tuning on original data to remove augmentation bias
3. **Control**: Standard training without augmentations for comparison

The ensemble provides more robust predictions by averaging across multiple models.

Usage
-----
    python ensemble_training_deepstarr.py

Outputs
-------
- 10 sets of model checkpoints (augmented, fine-tuned, control)
- Ensemble evaluation results
- Performance comparison plots
"""

import os
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from evoaug.augment import (
    RandomDeletion, RandomRC, RandomInsertion,
    RandomTranslocation, RandomMutation, RandomNoise
)
from evoaug.evoaug import RobustLoader
from evoaug_utils.model_zoo import DeepSTARRModel, DeepSTARR
from evoaug_utils import utils
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_control_model(output_dir, plots_dir):
    """Train a single control model for the entire ensemble."""
    print(f"\n{'='*60}")
    print(f"TRAINING CONTROL MODEL (No Augmentations)")
    print(f"{'='*60}")
    
    # Set seed for control model
    set_seed(42)  # Fixed seed for control model
    
    # Configuration for control model
    expt_name = 'DeepSTARR_control'
    data_path = '.'
    filepath = '/grid/wsbs/home_norepl/pmantill/train_EvoAug/deepstarr-data.h5'
    model_output_dir = os.path.join(output_dir, 'control')
    os.makedirs(model_output_dir, exist_ok=True)
    batch_size = 128
    
    # Load dataset
    print(f"Loading DeepSTARR dataset from: {filepath}")
    base_dataset = utils.H5Dataset(filepath, batch_size=batch_size, lower_case=False, transpose=False)
    print(f"Dataset loaded: {len(base_dataset)} sequences")
    print(f"Sequence length: {base_dataset.x_train.shape}")
    print(f"Number of tasks: {base_dataset.y_train.shape}")
    
    # Create FineTuneDataModule for Control training
    class FineTuneDataModule(pl.LightningDataModule):
        def __init__(self, base_dataset):
            super().__init__()
            self.base_dataset = base_dataset
            
        def train_dataloader(self):
            return self.base_dataset.train_dataloader()
            
        def val_dataloader(self):
            return self.base_dataset.val_dataloader()
            
        def test_dataloader(self):
            return self.base_dataset.test_dataloader()
    
    data_module_control = FineTuneDataModule(base_dataset)
    
    # CONTROL: Standard Training (No Augmentations)
    print(f"\n=== Control: Standard Training (No Augmentations) ===")
    
    # Check if control model already exists
    ckpt_control_path = expt_name + "_standard"
    best_control_path = os.path.join(model_output_dir, ckpt_control_path + ".ckpt")
    
    if os.path.exists(best_control_path):
        print(f"✓ Found existing control model checkpoint: {os.path.basename(best_control_path)}")
        print("Skipping control training - using existing model.")
    else:
        print("Training control model on original data without augmentations...")
        print("Control parameters: 100 epochs, learning rate 0.001, weight decay 1e-6")
        
        # Create model for control training
        deepstarr_control = DeepSTARR(2)
        model_control = DeepSTARRModel(deepstarr_control, learning_rate=0.001, weight_decay=1e-6)
        
        # Create trainer for control training
        callback_topmodel_control = pl.callbacks.ModelCheckpoint(
            monitor='val_loss',
            save_top_k=1,
            dirpath=model_output_dir,
            filename=ckpt_control_path
        )
        callback_es_control = pl.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        
        trainer_control = pl.Trainer(
            max_epochs=100,
            logger=None,
            callbacks=[callback_es_control, callback_topmodel_control],
            accelerator='auto',
            devices='auto'
        )
        
        # Train control model
        print(f"Starting control training on original, unperturbed data...")
        print(f"Control model will be saved to: {os.path.join(model_output_dir, ckpt_control_path + '.ckpt')}")
        print("Training from scratch without any augmentations for baseline comparison...")
        trainer_control.fit(model_control, datamodule=data_module_control)
        
        print("=== Control training completed ===")
    
    # Evaluate the control model
    if os.path.exists(best_control_path):
        print(f"Loading best control model from: {best_control_path}")
        deepstarr_control = DeepSTARR(2)
        model_control = DeepSTARRModel.load_from_checkpoint(best_control_path, model=deepstarr_control)
        
        # Evaluate on test set
        print("Evaluating control model on test set...")
        
        eval_trainer = pl.Trainer(
            logger=None,
            accelerator='auto',
            devices='auto'
        )
        
        eval_trainer.test(model_control, datamodule=data_module_control)
        
        # Get predictions
        pred_control = utils.get_predictions(model_control, base_dataset.x_test, batch_size=batch_size)
        results_control = utils.evaluate_model(base_dataset.y_test, pred_control, task='regression')
        
        # Print correlation metrics
        y_true_control = base_dataset.y_test
        if y_true_control is not None and pred_control is not None:
            print('Pearson r')
            vals = []
            for class_index in range(y_true_control.shape[-1]):
                vals.append(stats.pearsonr(y_true_control[:,class_index], pred_control[:,class_index])[0])
            pearson_control = np.array(vals)
            print(pearson_control)
            
            print('Spearman rho')
            vals = []
            for class_index in range(y_true_control.shape[-1]):
                vals.append(stats.spearmanr(y_true_control[:,class_index], pred_control[:,class_index])[0])
            spearman_control = np.array(vals)
            print(spearman_control)
        else:
            print("Warning: Could not compute metrics - y_true or y_score is None")
            pearson_control = np.array([])
            spearman_control = np.array([])
    
    # Return results for control model
    return {
        'seed': 'control',
        'control_path': best_control_path,
        'results_control': results_control if 'results_control' in locals() else None,
        'pearson_control': pearson_control if 'pearson_control' in locals() else None,
        'spearman_control': spearman_control if 'spearman_control' in locals() else None
    }

def train_single_model(seed, output_dir, plots_dir):
    """Train a single model with the given seed."""
    print(f"\n{'='*60}")
    print(f"TRAINING MODEL {seed} (Seed: {seed})")
    print(f"{'='*60}")
    
    # Set seed for this model
    set_seed(seed)
    
    # Configuration for this model
    expt_name = f'DeepSTARR_seed_{seed}'
    data_path = '.'
    filepath = '/grid/wsbs/home_norepl/pmantill/train_EvoAug/deepstarr-data.h5'
    model_output_dir = os.path.join(output_dir, f'seed_{seed}')
    os.makedirs(model_output_dir, exist_ok=True)
    batch_size = 128
    
    # Load dataset
    print(f"Loading DeepSTARR dataset from: {filepath}")
    base_dataset = utils.H5Dataset(filepath, batch_size=batch_size, lower_case=False, transpose=False)
    print(f"Dataset loaded: {len(base_dataset)} sequences")
    print(f"Sequence length: {base_dataset.x_train.shape}")
    print(f"Number of tasks: {base_dataset.y_train.shape}")
    
    # Create model architecture
    deepstarr = DeepSTARR(2)
    
    # Define augmentations using optimal DeepSTARR hyperparameters from hyperparameter search
    # Based on Additional file 1: Figs. S1, S3, and S4 from the paper
    augment_list = [
        #RandomDeletion(delete_min=0, delete_max=30),      # DeepSTARR optimal: delete_max = 30
        RandomTranslocation(shift_min=0, shift_max=20),   # DeepSTARR optimal: shift_max = 20
        # RandomInsertion(insert_min=0, insert_max=20),     # DeepSTARR optimal: insert_max = 20
        RandomRC(rc_prob=0.0),                           # DeepSTARR optimal: rc_prob = 0 (no reverse-complement)
        RandomMutation(mut_frac=0.05),                    # DeepSTARR optimal: mutate_frac = 0.05
        RandomNoise(noise_mean=0, noise_std=0.3),        # DeepSTARR optimal: noise_std = 0.3
    ]
    
    # Create augmented data module for Stage 1 training
    class AugmentedDataModule(pl.LightningDataModule):
        def __init__(self, base_dataset, augment_list, max_augs_per_seq, hard_aug):
            super().__init__()
            self.base_dataset = base_dataset
            self.augment_list = augment_list
            self.max_augs_per_seq = max_augs_per_seq
            self.hard_aug = hard_aug
            
        def train_dataloader(self):
            # Use RobustLoader with training dataset
            train_dataset = self.base_dataset.get_train_dataset()
            return RobustLoader(
                base_dataset=train_dataset,
                augment_list=self.augment_list,
                max_augs_per_seq=self.max_augs_per_seq,
                hard_aug=self.hard_aug,
                batch_size=self.base_dataset.batch_size,
                shuffle=True
            )
        
        def val_dataloader(self):
            # Use RobustLoader with validation dataset and disable augmentations
            val_dataset = self.base_dataset.get_val_dataset()
            loader = RobustLoader(
                base_dataset=val_dataset,
                augment_list=self.augment_list,
                max_augs_per_seq=self.max_augs_per_seq,
                hard_aug=self.hard_aug,
                batch_size=self.base_dataset.batch_size,
                shuffle=False
            )
            loader.disable_augmentations()
            return loader
        
        def test_dataloader(self):
            # Use RobustLoader with test dataset and disable augmentations
            test_dataset = self.base_dataset.get_test_dataset()
            loader = RobustLoader(
                base_dataset=test_dataset,
                augment_list=self.augment_list,
                max_augs_per_seq=self.max_augs_per_seq,
                hard_aug=self.hard_aug,
                batch_size=self.base_dataset.batch_size,
                shuffle=False
            )
            loader.disable_augmentations()
            return loader
    
    # Create augmented data module for Stage 1
    data_module = AugmentedDataModule(
        base_dataset, 
        augment_list, 
        max_augs_per_seq=2,  # DeepSTARR optimal: maximum 2 augmentations per sequence
        hard_aug=True         # DeepSTARR uses hard setting: always apply exactly 2 augmentations
    )
    
    # Create FineTuneDataModule for Stage 2 and Control training
    class FineTuneDataModule(pl.LightningDataModule):
        def __init__(self, base_dataset):
            super().__init__()
            self.base_dataset = base_dataset
            
        def train_dataloader(self):
            return self.base_dataset.train_dataloader()
            
        def val_dataloader(self):
            return self.base_dataset.val_dataloader()
            
        def test_dataloader(self):
            return self.base_dataset.test_dataloader()
    
    data_module_control = FineTuneDataModule(base_dataset)
    
    # STAGE 1: Training with Augmentations
    print(f"\n=== Stage 1: Training with EvoAug2 Augmentations (Seed {seed}) ===")
    
    # Check if augmented model already exists
    ckpt_aug_path = expt_name + "_aug"
    best_model_path = os.path.join(model_output_dir, ckpt_aug_path + ".ckpt")
    
    if os.path.exists(best_model_path):
        print(f"✓ Found existing augmented model checkpoint: {os.path.basename(best_model_path)}")
        print("Skipping Stage 1 training - using existing model.")
    else:
        print("Training DNN on sequences with EvoAug augmentations applied stochastically online...")
        print("Goal: Enhance model's ability to learn robust representations of features (e.g., motifs)")
        print("      by exposing it to expanded genetic variation while preserving motifs on average")
        
        # Create model for augmented training
        model_aug = DeepSTARRModel(deepstarr, learning_rate=0.001, weight_decay=1e-6)
        
        # Create trainer for augmented training
        callback_topmodel_aug = pl.callbacks.ModelCheckpoint(
            monitor='val_loss',
            save_top_k=1,
            dirpath=model_output_dir,
            filename=ckpt_aug_path
        )
        callback_es_aug = pl.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        
        trainer_aug = pl.Trainer(
            max_epochs=100,
            logger=None,
            callbacks=[callback_es_aug, callback_topmodel_aug],
            accelerator='auto',
            devices='auto'
        )
        
        # Train with augmentations
        print(f"Starting augmented training...")
        print(f"Augmented model will be saved to: {os.path.join(model_output_dir, ckpt_aug_path + '.ckpt')}")
        trainer_aug.fit(model_aug, datamodule=data_module)
        
        print("=== Augmented training completed ===")
    
    # Evaluate augmented model
    if os.path.exists(best_model_path):
        print(f"Loading best augmented model from: {best_model_path}")
        model_aug = DeepSTARRModel.load_from_checkpoint(best_model_path, model=deepstarr)
        
        # Evaluate on test set (no augmentations used during evaluation)
        print("Evaluating augmented model on test set (no augmentations)...")
        
        eval_trainer = pl.Trainer(
            logger=None,
            accelerator='auto',
            devices='auto'
        )
        
        eval_trainer.test(model_aug, datamodule=data_module)
        
        # Get predictions
        pred_aug = utils.get_predictions(model_aug, base_dataset.x_test, batch_size=batch_size)
        results_aug = utils.evaluate_model(base_dataset.y_test, pred_aug, task='regression')
        
        # Print correlation metrics
        y_true_aug = base_dataset.y_test
        if y_true_aug is not None and pred_aug is not None:
            print('Pearson r')
            vals = []
            for class_index in range(y_true_aug.shape[-1]):
                vals.append(stats.pearsonr(y_true_aug[:,class_index], pred_aug[:,class_index])[0])
            pearson_aug = np.array(vals)
            print(pearson_aug)
            
            print('Spearman rho')
            vals = []
            for class_index in range(y_true_aug.shape[-1]):
                vals.append(stats.spearmanr(y_true_aug[:,class_index], pred_aug[:,class_index])[0])
            spearman_r = np.array(vals)
            print(spearman_r)
        else:
            print("Warning: Could not compute metrics - y_true or y_score is None")
            pearson_aug = np.array([])
            spearman_r = np.array([])
    
    # STAGE 2: Fine-tuning on Original Data
    print(f"\n=== Stage 2: Fine-tuning on Original Data (Seed {seed}) ===")
    

    
    # Check if fine-tuned model already exists
    ckpt_finetune_path = expt_name + "_finetune"
    best_finetune_path = os.path.join(model_output_dir, ckpt_finetune_path + ".ckpt")
    
    if os.path.exists(best_finetune_path):
        print(f"✓ Found existing fine-tuned model checkpoint: {os.path.basename(best_finetune_path)}")
        print("Skipping Stage 2 fine-tuning - using existing model.")
    else:
        print("Fine-tuning the augmented model on original, unperturbed data to remove augmentation bias...")
        print("Fine-tuning parameters: 5 epochs, learning rate 0.0001, weight decay 1e-6")
        
        # Load the best augmented model for fine-tuning
        if os.path.exists(best_model_path):
            print(f"Loading best augmented model for fine-tuning from: {best_model_path}")
            model_finetune = DeepSTARRModel.load_from_checkpoint(best_model_path, model=deepstarr)
            model_finetune.learning_rate = 0.0001
            model_finetune.configure_optimizers()
            print("✓ Successfully loaded augmented model for fine-tuning")
        else:
            print("✗ ERROR: No augmented model found for fine-tuning!")
            return None
        
        # Create trainer for fine-tuning
        callback_topmodel_finetune = pl.callbacks.ModelCheckpoint(
            monitor='val_loss',
            save_top_k=1,
            dirpath=model_output_dir,
            filename=ckpt_finetune_path
        )
        
        trainer_finetune = pl.Trainer(
            max_epochs=5,
            logger=None,
            callbacks=[callback_topmodel_finetune],
            accelerator='auto',
            devices='auto'
        )
        
        # Fine-tune model on original data
        print(f"Starting fine-tuning on original, unperturbed data...")
        print(f"Fine-tuned model will be saved to: {os.path.join(model_output_dir, ckpt_finetune_path + '.ckpt')}")
        print("Goal: Remove augmentation bias and refine features towards observed biology")
        trainer_finetune.fit(model_finetune, datamodule=data_module_control)
        
        print("=== Fine-tuning on original data completed ===")
    
    # Evaluate the fine-tuned model
    if os.path.exists(best_finetune_path):
        print(f"Loading best fine-tuned model from: {best_finetune_path}")
        model_finetune = DeepSTARRModel.load_from_checkpoint(best_finetune_path, model=deepstarr)
        
        # Evaluate on test set (no augmentations used during evaluation)
        print("Evaluating fine-tuned model on test set (no augmentations)...")
        
        eval_trainer = pl.Trainer(
            logger=None,
            accelerator='auto',
            devices='auto'
        )
        
        eval_trainer.test(model_finetune, datamodule=data_module_control)
        
        # Get predictions
        pred_finetune = utils.get_predictions(model_finetune, base_dataset.x_test, batch_size=batch_size)
        results_finetune = utils.evaluate_model(base_dataset.y_test, pred_finetune, task='regression')
        
        # Print correlation metrics
        y_true_finetune = base_dataset.y_test
        if y_true_finetune is not None and pred_finetune is not None:
            print('Pearson r')
            vals = []
            for class_index in range(y_true_finetune.shape[-1]):
                vals.append(stats.pearsonr(y_true_finetune[:,class_index], pred_finetune[:,class_index])[0])
            pearson_finetune = np.array(vals)
            print(pearson_finetune)
            
            print('Spearman rho')
            vals = []
            for class_index in range(y_true_finetune.shape[-1]):
                vals.append(stats.spearmanr(y_true_finetune[:,class_index], pred_finetune[:,class_index])[0])
            spearman_finetune = np.array(vals)
            print(spearman_finetune)
        else:
            print("Warning: Could not compute metrics - y_true or y_score is None")
            pearson_finetune = np.array([])
            spearman_finetune = np.array([])
    
    # Return results for this model
    return {
        'seed': seed,
        'augmented_path': best_model_path,
        'finetuned_path': best_finetune_path,
        'results_aug': results_aug if 'results_aug' in locals() else None,
        'results_finetune': results_finetune if 'results_finetune' in locals() else None,
        'pearson_aug': pearson_aug if 'pearson_aug' in locals() else None,
        'spearman_aug': spearman_r if 'spearman_r' in locals() else None,
        'pearson_finetune': pearson_finetune if 'pearson_finetune' in locals() else None,
        'spearman_finetune': spearman_finetune if 'spearman_finetune' in locals() else None
    }

def save_performance_incrementally(result, output_dir, control_result=None):
    """Save performance data incrementally to prevent data loss."""
    csv_path = os.path.join(output_dir, 'ensemble_performance.csv')
    
    # Check if CSV already exists and is not empty
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        try:
            df = pd.read_csv(csv_path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()
    
    # Prepare data for this model
    new_data = []
    seed = result['seed']
    
    # Add augmented model results
    if result['results_aug'] is not None:
        new_data.append({
            'seed': seed,
            'model_type': 'augmented',
            'pearson_mean': np.mean(result['pearson_aug']) if 'pearson_aug' in result else np.nan,
            'pearson_std': np.std(result['pearson_aug']) if 'pearson_aug' in result else np.nan,
            'spearman_mean': np.mean(result['spearman_aug']) if 'spearman_aug' in result else np.nan,
            'spearman_std': np.std(result['spearman_aug']) if 'spearman_aug' in result else np.nan
        })
    
    # Add fine-tuned model results
    if result['results_finetune'] is not None:
        new_data.append({
            'seed': seed,
            'model_type': 'finetuned',
            'pearson_mean': np.mean(result['pearson_finetune']) if 'pearson_finetune' in result else np.nan,
            'pearson_std': np.std(result['pearson_finetune']) if 'pearson_finetune' in result else np.nan,
            'spearman_mean': np.mean(result['spearman_finetune']) if 'spearman_finetune' in result else np.nan,
            'spearman_std': np.std(result['spearman_finetune']) if 'spearman_finetune' in result else np.nan
        })
    
    # Add control model results (only if provided and not already in CSV)
    if control_result is not None and (df.empty or 'control' not in df['model_type'].values):
        new_data.append({
            'seed': 'control',
            'model_type': 'control',
            'pearson_mean': np.mean(control_result['pearson_control']) if 'pearson_control' in control_result else np.nan,
            'pearson_std': np.std(control_result['pearson_control']) if 'pearson_control' in control_result else np.nan,
            'spearman_mean': np.mean(control_result['spearman_control']) if 'spearman_control' in control_result else np.nan,
            'spearman_std': np.std(control_result['spearman_control']) if 'spearman_control' in control_result else np.nan
        })
    
    # Append new data to existing DataFrame
    if new_data:
        new_df = pd.DataFrame(new_data)
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(csv_path, index=False)
        print(f"✓ Performance data updated: {csv_path}")
    
    return df

def create_performance_csv(ensemble_results, output_dir):
    """Create CSV file with performance data from all models."""
    data = []
    
    for result in ensemble_results:
        seed = result['seed']
        
        # Add augmented model results
        if result['results_aug'] is not None:
            data.append({
                'seed': seed,
                'model_type': 'augmented',
                'pearson_mean': np.mean(result['pearson_aug']) if 'pearson_aug' in result else np.nan,
                'pearson_std': np.std(result['pearson_aug']) if 'pearson_aug' in result else np.nan,
                'spearman_mean': np.mean(result['spearman_aug']) if 'spearman_aug' in result else np.nan,
                'spearman_std': np.std(result['spearman_aug']) if 'spearman_aug' in result else np.nan
            })
        
        # Add fine-tuned model results
        if result['results_finetune'] is not None:
            data.append({
                'seed': seed,
                'model_type': 'finetuned',
                'pearson_mean': np.mean(result['pearson_finetune']) if 'pearson_finetune' in result else np.nan,
                'pearson_std': np.std(result['pearson_finetune']) if 'pearson_finetune' in result else np.nan,
                'spearman_mean': np.mean(result['spearman_finetune']) if 'spearman_finetune' in result else np.nan,
                'spearman_std': np.std(result['spearman_finetune']) if 'spearman_finetune' in result else np.nan
            })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    csv_path = os.path.join(output_dir, 'ensemble_performance.csv')
    df.to_csv(csv_path, index=False)
    print(f"✓ Performance data saved to: {csv_path}")
    
    return df

def create_performance_plots(ensemble_results, plots_dir, control_result=None):
    """Create performance comparison plots."""
    # Extract data for plotting
    seeds = []
    augmented_pearson = []
    finetuned_pearson = []
    control_pearson = []
    augmented_spearman = []
    finetuned_spearman = []
    control_spearman = []
    
    for result in ensemble_results:
        seeds.append(result['seed'])
        
        if result['pearson_aug'] is not None:
            augmented_pearson.append(np.mean(result['pearson_aug']))
            augmented_spearman.append(np.mean(result['spearman_aug']))
        else:
            augmented_pearson.append(np.nan)
            augmented_spearman.append(np.nan)
        
        if result['pearson_finetune'] is not None:
            finetuned_pearson.append(np.mean(result['pearson_finetune']))
            finetuned_spearman.append(np.mean(result['spearman_finetune']))
        else:
            finetuned_pearson.append(np.nan)
            finetuned_spearman.append(np.nan)
    
    # Add control model data if available
    if control_result and control_result['pearson_control'] is not None:
        control_pearson_val = np.mean(control_result['pearson_control'])
        control_spearman_val = np.mean(control_result['spearman_control'])
    else:
        control_pearson_val = np.nan
        control_spearman_val = np.nan
    
    # Create performance comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Pearson correlation plot
    ax1.plot(seeds, augmented_pearson, 'o-', label='Augmented', linewidth=2, markersize=8)
    ax1.plot(seeds, finetuned_pearson, 's-', label='Fine-tuned', linewidth=2, markersize=8)
    if not np.isnan(control_pearson_val):
        ax1.axhline(y=control_pearson_val, color='red', linestyle='--', linewidth=2, label='Control')
    ax1.set_xlabel('Random Seed')
    ax1.set_ylabel('Pearson Correlation')
    ax1.set_title('Pearson Correlation vs Control Across Seeds')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(seeds)
    
    # Spearman correlation plot
    ax2.plot(seeds, augmented_spearman, 'o-', label='Augmented', linewidth=2, markersize=8)
    ax2.plot(seeds, finetuned_spearman, 's-', label='Fine-tuned', linewidth=2, markersize=8)
    if not np.isnan(control_spearman_val):
        ax2.axhline(y=control_spearman_val, color='red', linestyle='--', linewidth=2, label='Control')
    ax2.set_xlabel('Random Seed')
    ax2.set_ylabel('Spearman Correlation')
    ax2.set_title('Spearman Correlation vs Control Across Seeds')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(seeds)
    
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, 'ensemble_performance_comparison.svg')
    plt.savefig(plot_path, format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Performance comparison plot saved to: {plot_path}")
    
    # Create box plot for overall comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Prepare data for box plots
    pearson_data = [augmented_pearson, finetuned_pearson]
    spearman_data = [augmented_spearman, finetuned_spearman]
    labels = ['Augmented', 'Fine-tuned']
    
    # Pearson box plot
    bp1 = ax1.boxplot(pearson_data, labels=labels, patch_artist=True)
    colors = ['lightblue', 'lightgreen']
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
    
    # Add control line if available
    if not np.isnan(control_pearson_val):
        ax1.axhline(y=control_pearson_val, color='red', linestyle='--', linewidth=2, label='Control')
        ax1.legend()
    
    ax1.set_ylabel('Pearson Correlation')
    ax1.set_title('Pearson Correlation Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Spearman box plot
    bp2 = ax2.boxplot(spearman_data, labels=labels, patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
    
    # Add control line if available
    if not np.isnan(control_spearman_val):
        ax2.axhline(y=control_spearman_val, color='red', linestyle='--', linewidth=2, label='Control')
        ax2.legend()
    
    ax2.set_ylabel('Spearman Correlation')
    ax2.set_title('Spearman Correlation Distribution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    box_plot_path = os.path.join(plots_dir, 'ensemble_performance_distribution.svg')
    plt.savefig(box_plot_path, format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Performance distribution plot saved to: {box_plot_path}")

def main():
    """Main function to train ensemble of models."""
    print("EvoAug2 Ensemble Training")
    print("=" * 50)
    print("Training 10 models with different random seeds for ensemble prediction")
    print("Training 1 control model for baseline comparison")
    
    # Configuration
    output_dir = '/grid/wsbs/home_norepl/pmantill/train_EvoAug/EvoAug_ensemble_10'
    plots_dir = '/grid/wsbs/home_norepl/pmantill/train_EvoAug/ensemble_plots'
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Define seeds for ensemble (10 models total) - multiples of 3
    seeds = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
    
    # Train control model first
    print(f"\n{'='*80}")
    print("TRAINING CONTROL MODEL")
    print(f"{'='*80}")
    
    control_result = None
    try:
        control_result = train_control_model(output_dir, plots_dir)
        if control_result:
            print(f"✓ Control model completed successfully")
            # Save control results immediately
            save_performance_incrementally({}, output_dir, control_result)
        else:
            print(f"✗ Control model failed")
    except Exception as e:
        print(f"✗ Control model failed with error: {e}")
    
    # Train ensemble models
    ensemble_results = []
    
    for i, seed in enumerate(seeds, 1):
        print(f"\n{'='*80}")
        print(f"TRAINING MODEL {i}/10 (Seed: {seed})")
        print(f"{'='*80}")
        
        try:
            result = train_single_model(seed, output_dir, plots_dir)
            if result:
                ensemble_results.append(result)
                print(f"✓ Model {i} (seed {seed}) completed successfully")
                # Save performance data immediately after each model
                save_performance_incrementally(result, output_dir, control_result)
            else:
                print(f"✗ Model {i} (seed {seed}) failed")
        except Exception as e:
            print(f"✗ Model {i} (seed {seed}) failed with error: {e}")
            continue
    
    # Print ensemble summary
    print(f"\n{'='*80}")
    print("ENSEMBLE TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"Successfully trained: {len(ensemble_results)}/10 models")
    print(f"Control model: {'✓' if control_result else '✗'}")
    print(f"Models saved to: {output_dir}")
    
    for result in ensemble_results:
        print(f"  Seed {result['seed']}:")
        print(f"    - Augmented: {os.path.basename(result['augmented_path'])}")
        print(f"    - Fine-tuned: {os.path.basename(result['finetuned_path'])}")
    
    if control_result:
        print(f"  Control: {os.path.basename(control_result['control_path'])}")
    
    # Create performance plots
    print(f"Creating performance comparison plots...")
    create_performance_plots(ensemble_results, plots_dir, control_result)
    
    print(f"\nTo use the ensemble for predictions:")
    print(f"1. Load all fine-tuned models from {output_dir}")
    print(f"2. Average their predictions for robust ensemble predictions")
    print(f"3. Compare ensemble performance against control model")
    print(f"4. Check performance data in: {os.path.join(output_dir, 'ensemble_performance.csv')}")
    print(f"5. Check performance plots in: {plots_dir}")

if __name__ == "__main__":
    main()
