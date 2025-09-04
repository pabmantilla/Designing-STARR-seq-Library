#!/usr/bin/env python3
"""
Distillation of EvoAug model ensemble into a single model using pseudo-labeling

This script loads the trained ensemble models and distills their knowledge into
a single student model using pseudo-labels generated from ensemble predictions.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from evoaug_utils.model_zoo import DeepSTARRModel, DeepSTARR
from evoaug_utils import utils
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class PseudoLabeledDataset(Dataset):
    """Dataset for pseudo-labeled data with confidence weighting."""
    def __init__(self, sequences, pseudo_labels, pseudo_uncertainties=None):
        self.sequences = sequences
        self.pseudo_labels = pseudo_labels
        self.pseudo_uncertainties = pseudo_uncertainties
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'x': self.sequences[idx],
            'y': self.pseudo_labels[idx],
            'uncertainty': self.pseudo_uncertainties[idx] if self.pseudo_uncertainties is not None else None
        }

def load_ensemble_models(output_dir, seeds=[3, 6, 9, 12, 15, 18, 21, 24, 27, 30]):
    """Load all trained ensemble models."""
    print("Loading ensemble models...")
    ensemble_models = []
    control_model = None
    
    # Load control model
    control_path = os.path.join(output_dir, 'control', 'DeepSTARR_control_standard.ckpt')
    if os.path.exists(control_path):
        print(f"Loading control model from: {control_path}")
        control_model = DeepSTARRModel.load_from_checkpoint(control_path, model=DeepSTARR(2))
        print("✓ Control model loaded")
    else:
        print("✗ Control model not found")
    
    # Load ensemble models
    for seed in seeds:
        model_path = os.path.join(output_dir, f'seed_{seed}', f'DeepSTARR_seed_{seed}_finetune.ckpt')
        if os.path.exists(model_path):
            print(f"Loading model seed {seed} from: {model_path}")
            model = DeepSTARRModel.load_from_checkpoint(model_path, model=DeepSTARR(2))
            ensemble_models.append(model)
            print(f"✓ Model seed {seed} loaded")
        else:
            print(f"✗ Model seed {seed} not found")
    
    print(f"Successfully loaded {len(ensemble_models)} ensemble models")
    return ensemble_models, control_model

def generate_pseudo_labels(ensemble_models, data_loader, device='cuda'):
    """Generate pseudo-labels using ensemble predictions."""
    print("Generating pseudo-labels from ensemble...")
    
    # Set all models to eval mode
    for model in ensemble_models:
        model.eval()
    
    pseudo_labels = []
    pseudo_uncertainties = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx % 100 == 0:
                print(f"Processing batch {batch_idx}/{len(data_loader)}")
            
            x = batch['x'].to(device)
            
            # Get predictions from all models
            model_predictions = []
            for model in ensemble_models:
                pred = model(x)
                model_predictions.append(pred)
            
            # Stack predictions: [num_models, batch_size, num_tasks]
            stacked_preds = torch.stack(model_predictions, dim=0)
            
            # Calculate ensemble mean (pseudo-label)
            ensemble_mean = stacked_preds.mean(dim=0)
            
            # Calculate ensemble uncertainty (variance across models)
            ensemble_var = stacked_preds.var(dim=0)
            ensemble_std = torch.sqrt(ensemble_var)
            
            pseudo_labels.append(ensemble_mean.cpu())
            pseudo_uncertainties.append(ensemble_std.cpu())
    
    pseudo_labels = torch.cat(pseudo_labels, dim=0)
    pseudo_uncertainties = torch.cat(pseudo_uncertainties, dim=0)
    
    print(f"Generated pseudo-labels for {len(pseudo_labels)} samples")
    print(f"Average uncertainty: {pseudo_uncertainties.mean():.4f}")
    print(f"Uncertainty std: {pseudo_uncertainties.std():.4f}")
    
    return pseudo_labels, pseudo_uncertainties

def create_psuedo_labels(ensemble_models, base_dataset, output_dir):
    """Create pseudo-labeled dataset from ensemble predictions."""
    print("Creating pseudo-labeled dataset...")
    
    # Generate pseudo-labels on training data
    train_loader = base_dataset.train_dataloader()
    pseudo_labels, pseudo_uncertainties = generate_pseudo_labels(ensemble_models, train_loader)
    
    # Create pseudo-labeled dataset (no filtering)
    pseudo_dataset = PseudoLabeledDataset(
        base_dataset.x_train,
        pseudo_labels,
        pseudo_uncertainties
    )
    
    print(f"Using all {len(pseudo_dataset)} pseudo-labels for training")
    
    # Save pseudo-labels for later use
    pseudo_data_path = os.path.join(output_dir, 'pseudo_labels.pt')
    torch.save({
        'pseudo_labels': pseudo_labels,
        'pseudo_uncertainties': pseudo_uncertainties
    }, pseudo_data_path)
    print(f"Pseudo-labels saved to: {pseudo_data_path}")
    
    return pseudo_dataset

def train_single_model(pseudo_dataset, base_dataset, output_dir, num_epochs=100, learning_rate=0.001):
    """Train a single student model using pseudo-labels."""
    print("Training student model with pseudo-labels...")
    
    # Create data loader for pseudo-labeled data
    pseudo_loader = DataLoader(
        pseudo_dataset,
        batch_size=base_dataset.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    # Create student model
    student_model = DeepSTARR(2)
    student_lightning = DeepSTARRModel(student_model, learning_rate=learning_rate, weight_decay=1e-6)
    
    # Create trainer for student model
    callback_topmodel = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        save_top_k=1,
        dirpath=output_dir,
        filename='distilled_student'
    )
    callback_es = pl.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        logger=None,
        callbacks=[callback_es, callback_topmodel],
        accelerator='auto',
        devices='auto'
    )
    
    # Custom training loop for pseudo-labeling
    print("Starting pseudo-label training...")
    
    # Set up optimizer
    optimizer = student_lightning.configure_optimizers()
    
    # Training loop
    student_lightning.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch in pseudo_loader:
            optimizer.zero_grad()
            
            # Forward pass
            x = batch['x'].to(student_lightning.device)
            y_pseudo = batch['y'].to(student_lightning.device)
            
            # Get student predictions
            student_pred = student_lightning(x)
            
            # Calculate loss (MSE between student and pseudo-labels)
            loss = F.mse_loss(student_pred, y_pseudo)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
        
        # Early stopping check
        if callback_es.stopped_epoch > 0:
            print(f"Early stopping at epoch {callback_es.stopped_epoch}")
            break
    
    print("✓ Student model training completed")
    return student_lightning

def evaluate_model(student_model, ensemble_models, control_model, base_dataset, plots_dir):
    """Evaluate the distilled model against ensemble and control."""
    print("Evaluating distilled model...")
    
    # Get test data
    test_loader = base_dataset.test_dataloader()
    y_true = base_dataset.y_test
    
    # Get ensemble predictions
    print("Getting ensemble predictions...")
    ensemble_preds = []
    for i, model in enumerate(ensemble_models):
        print(f"Evaluating ensemble model {i+1}/{len(ensemble_models)}")
        model.eval()
        with torch.no_grad():
            preds = []
            for batch in test_loader:
                pred = model(batch['x'])
                preds.append(pred)
            ensemble_preds.append(torch.cat(preds, dim=0))
    
    ensemble_avg = torch.stack(ensemble_preds).mean(dim=0)
    
    # Get student predictions
    print("Getting student model predictions...")
    student_model.eval()
    with torch.no_grad():
        student_preds = []
        for batch in test_loader:
            pred = student_model(batch['x'])
            student_preds.append(pred)
        student_preds = torch.cat(student_preds, dim=0)
    
    # Get control predictions
    control_preds = None
    if control_model is not None:
        print("Getting control model predictions...")
        control_model.eval()
        with torch.no_grad():
            control_preds = []
            for batch in test_loader:
                pred = control_model(batch['x'])
                control_preds.append(pred)
            control_preds = torch.cat(control_preds, dim=0)
    
    # Calculate metrics
    print("Calculating performance metrics...")
    
    # Pearson correlation
    ensemble_pearson = [stats.pearsonr(y_true[:, i], ensemble_avg[:, i])[0] for i in range(y_true.shape[1])]
    student_pearson = [stats.pearsonr(y_true[:, i], student_preds[:, i])[0] for i in range(y_true.shape[1])]
    
    results = {
        'ensemble_pearson': ensemble_pearson,
        'student_pearson': student_pearson,
        'performance_retention': np.mean(student_pearson) / np.mean(ensemble_pearson)
    }
    
    if control_preds is not None:
        control_pearson = [stats.pearsonr(y_true[:, i], control_preds[:, i])[0] for i in range(y_true.shape[1])]
        results['control_pearson'] = control_pearson
    
    # Print results
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    print(f"Ensemble Pearson: {np.mean(ensemble_pearson):.4f} ± {np.std(ensemble_pearson):.4f}")
    print(f"Student Pearson:  {np.mean(student_pearson):.4f} ± {np.std(student_pearson):.4f}")
    if control_preds is not None:
        print(f"Control Pearson:  {np.mean(control_pearson):.4f} ± {np.std(control_pearson):.4f}")
    print(f"Performance Retention: {results['performance_retention']:.4f}")
    print("="*60)
    
    return results

def save_model(student_model, output_dir):
    """Save the distilled student model."""
    print("Saving distilled student model...")
    
    # Save PyTorch Lightning checkpoint
    ckpt_path = os.path.join(output_dir, 'distilled_student.ckpt')
    torch.save(student_model.state_dict(), ckpt_path)
    print(f"✓ Student model saved to: {ckpt_path}")
    
    # Save model architecture and weights separately
    model_path = os.path.join(output_dir, 'distilled_student_model.pth')
    torch.save(student_model.model.state_dict(), model_path)
    print(f"✓ Model weights saved to: {model_path}")

def main():
    """Main function to distill ensemble into single model."""
    print("EvoAug Ensemble Distillation")
    print("=" * 50)
    print("Distilling 10-model ensemble into single student model using pseudo-labeling")
    
    # Configuration
    output_dir = '/grid/wsbs/home_norepl/pmantill/train_EvoAug/EvoAug_ensemble_10'
    plots_dir = '/grid/wsbs/home_norepl/pmantill/train_EvoAug/ensemble_plots'
    data_path = '/grid/wsbs/home_norepl/pmantill/train_EvoAug/deepstarr-data.h5'
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Load dataset
    print("Loading DeepSTARR dataset...")
    base_dataset = utils.H5Dataset(data_path, batch_size=128, lower_case=False, transpose=False)
    print(f"Dataset loaded: {len(base_dataset)} sequences")
    
    # Load ensemble models and control model
    ensemble_models, control_model = load_ensemble_models(output_dir)
    
    if len(ensemble_models) == 0:
        print("✗ No ensemble models found! Please train the ensemble first.")
        return
    
    # Create pseudo-labels from ensemble
    pseudo_dataset = create_psuedo_labels(ensemble_models, base_dataset, output_dir)
    
    # Train single model with pseudo-labels
    student_model = train_single_model(pseudo_dataset, base_dataset, output_dir)
    
    # Evaluate the model
    results = evaluate_model(student_model, ensemble_models, control_model, base_dataset, plots_dir)
    
    # Save the model
    save_model(student_model, output_dir)
    
    print("\n" + "="*60)
    print("DISTILLATION COMPLETED")
    print("="*60)
    print(f"Student model performance retention: {results['performance_retention']:.3f}")
    print(f"Models saved to: {output_dir}")
    print(f"Plots saved to: {plots_dir}")
    print("="*60)

if __name__ == "__main__":
    main()