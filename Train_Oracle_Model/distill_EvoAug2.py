#!/usr/bin/env python3
"""
EvoAug2 Uncertainty-Aware Distillation

Distills an ensemble of 10 teacher models into a single student model that learns
both mean predictions and epistemic uncertainty from the ensemble.

The student model has:
- Mean head: 2 outputs (dev, hk) - targets ensemble mean predictions
- Epistemic uncertainty head: 2 outputs (dev, hk) - targets ensemble std deviation
- Uses softplus activation for uncertainty outputs to ensure non-negative values

Loss function: L = L_mean + λ * L_uncertainty where λ=1
- L_mean = MSE between predicted mean and teacher ensemble mean
- L_uncertainty = MSE between predicted uncertainty and teacher ensemble std
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
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_float32_matmul_precision('medium')
plt.style.use('default')
sns.set_palette("husl")

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class UncertaintyAwareStudent(nn.Module):
    """Student model with separate heads for mean and epistemic uncertainty prediction.
    
    Uses the exact same backbone architecture as DeepSTARR but with modified output heads.
    Only predicts epistemic uncertainty (model-to-model variability), not aleatoric uncertainty.
    """
    
    def __init__(self, teacher_model, num_tasks=2):
        super().__init__()
        self.num_tasks = num_tasks
        
        # Create a new DeepSTARR model with the same architecture as teacher
        self.backbone = DeepSTARR(2)
        
        # Copy all weights from teacher except the final layer (fc7)
        teacher_state = teacher_model.model.state_dict()
        backbone_state = self.backbone.state_dict()
        
        # Copy all layers except the final linear layer (fc7)
        for name, param in teacher_state.items():
            if name in backbone_state and name not in ['fc7.weight', 'fc7.bias']:
                backbone_state[name] = param
        
        self.backbone.load_state_dict(backbone_state)
        
        # Remove the final layer (fc7) to get just the backbone
        delattr(self.backbone, 'fc7')
        
        # Mean prediction head (same as teacher: 2 outputs)
        self.mean_head = nn.Linear(256, num_tasks)
        
        # Epistemic uncertainty head (2 outputs, constrained to be non-negative)
        self.uncertainty_head = nn.Linear(256, num_tasks)
    
    def forward(self, x):
        # Extract features using backbone (same as teacher)
        # Go through all layers except the final fc7
        cnn = torch.conv1d(x, self.backbone.conv1_filters, stride=1, padding="same")
        cnn = self.backbone.batchnorm1(cnn)
        cnn = self.backbone.activation1(cnn)
        cnn = self.backbone.maxpool1(cnn)

        cnn = torch.conv1d(cnn, self.backbone.conv2_filters, stride=1, padding="same")
        cnn = self.backbone.batchnorm2(cnn)
        cnn = self.backbone.activation(cnn)
        cnn = self.backbone.maxpool2(cnn)

        cnn = torch.conv1d(cnn, self.backbone.conv3_filters, stride=1, padding="same")
        cnn = self.backbone.batchnorm3(cnn)
        cnn = self.backbone.activation(cnn)
        cnn = self.backbone.maxpool3(cnn)

        cnn = torch.conv1d(cnn, self.backbone.conv4_filters, stride=1, padding="same")
        cnn = self.backbone.batchnorm4(cnn)
        cnn = self.backbone.activation(cnn)
        cnn = self.backbone.maxpool4(cnn)

        cnn = self.backbone.flatten(cnn)
        cnn = self.backbone.fc5(cnn)
        cnn = self.backbone.batchnorm5(cnn)
        cnn = self.backbone.activation(cnn)
        cnn = self.backbone.dropout4(cnn)

        cnn = self.backbone.fc6(cnn)
        cnn = self.backbone.batchnorm6(cnn)
        cnn = self.backbone.activation(cnn)
        cnn = self.backbone.dropout4(cnn)
        
        # Now we have the features (256-dimensional)
        features = cnn
        
        # Mean predictions (no activation - raw outputs)
        mean_pred = self.mean_head(features)
        
        # Epistemic uncertainty predictions (softplus to ensure non-negative)
        uncertainty_pred = F.softplus(self.uncertainty_head(features))
        
        return {
            'mean': mean_pred,
            'uncertainty': uncertainty_pred
        }

class UncertaintyAwareLightningModel(pl.LightningModule):
    """PyTorch Lightning wrapper for uncertainty-aware student model."""
    
    def __init__(self, student_model, learning_rate=0.001, weight_decay=1e-6, 
                 lambda_uncertainty=1.0):
        super().__init__()
        self.student_model = student_model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lambda_uncertainty = lambda_uncertainty
        
    def forward(self, x):
        return self.student_model(x)
    
    def training_step(self, batch, batch_idx):
        x = batch['x']
        y_mean = batch['y_mean']  # Teacher ensemble mean
        y_uncertainty = batch['y_uncertainty']  # Teacher ensemble std
        
        # Get student predictions
        outputs = self(x)
        mean_pred = outputs['mean']  # Predicted mean activity
        uncertainty_pred = outputs['uncertainty']  # Predicted epistemic uncertainty
        
        # Multi-task loss: L = L_mean + λ * L_uncertainty
        # L_mean = MSE between predicted mean and teacher ensemble mean
        loss_mean = F.mse_loss(mean_pred, y_mean)
        
        # L_uncertainty = MSE between predicted uncertainty and teacher ensemble std
        loss_uncertainty = F.mse_loss(uncertainty_pred, y_uncertainty)
        
        # Total loss: L = L_mean + λ * L_uncertainty
        total_loss = loss_mean + self.lambda_uncertainty * loss_uncertainty
        
        # Log individual losses
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_loss_mean', loss_mean, on_step=True, on_epoch=True)
        self.log('train_loss_uncertainty', loss_uncertainty, on_step=True, on_epoch=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[0]  # Ground truth validation data
        y_true = batch[1]
        
        # Get student predictions
        outputs = self(x)
        mean_pred = outputs['mean']
        
        # Use mean predictions for validation (standard MSE loss)
        loss = F.mse_loss(mean_pred, y_true)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        return optimizer

class UncertaintyDataset(Dataset):
    """Dataset for uncertainty-aware distillation with pseudo-labels."""
    
    def __init__(self, sequences, mean_labels, uncertainty_labels):
        self.sequences = sequences
        self.mean_labels = mean_labels
        self.uncertainty_labels = uncertainty_labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'x': self.sequences[idx],
            'y_mean': self.mean_labels[idx],
            'y_uncertainty': self.uncertainty_labels[idx]
        }

def load_ensemble_models(output_dir, seeds=[3, 6, 9, 12, 15, 18, 21, 24, 27, 30]):
    """Load all trained ensemble models."""
    print("Loading ensemble models...")
    ensemble_models = []
    
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
    return ensemble_models

def generate_uncertainty_pseudo_labels(ensemble_models, base_dataset, device='cuda'):
    """Generate pseudo-labels with mean and epistemic uncertainty from ensemble.
    
    Uses the exact same prediction method as the training script (utils.get_predictions)
    to ensure consistency with how models were evaluated during training.
    """
    print("Generating uncertainty-aware pseudo-labels from ensemble...")
    print("Using same prediction method as training script (utils.get_predictions)")
    
    # Set all models to eval mode
    for model in ensemble_models:
        model.eval()
    
    # Get predictions from each model using the same method as training script
    model_predictions = []
    for i, model in enumerate(ensemble_models):
        print(f"Getting predictions from model {i+1}/{len(ensemble_models)}...")
        # Use utils.get_predictions exactly like in training script
        pred = utils.get_predictions(model, base_dataset.x_train, batch_size=128)
        model_predictions.append(torch.tensor(pred))
        
        # Debug: print first model predictions
        if i == 0:
            print(f"Debug - Model {i} prediction shape: {pred.shape}")
            print(f"Debug - Model {i} prediction range: [{pred.min():.4f}, {pred.max():.4f}]")
            print(f"Debug - Model {i} prediction mean: {pred.mean():.4f}")
    
    # Stack predictions: [num_models, num_samples, num_tasks]
    stacked_preds = torch.stack(model_predictions, dim=0)
    
    # Calculate ensemble mean (pseudo-label for mean head)
    ensemble_mean = stacked_preds.mean(dim=0)
    
    # Calculate ensemble standard deviation (pseudo-label for uncertainty head)
    ensemble_std = stacked_preds.std(dim=0)
    
    # Debug: print ensemble stats
    print(f"Debug - Stacked predictions shape: {stacked_preds.shape}")
    print(f"Debug - Ensemble mean range: [{ensemble_mean.min():.4f}, {ensemble_mean.max():.4f}]")
    print(f"Debug - Ensemble std range: [{ensemble_std.min():.4f}, {ensemble_std.max():.4f}]")
    print(f"Debug - Individual model predictions range:")
    for i, pred in enumerate(model_predictions):
        print(f"  Model {i}: [{pred.min():.4f}, {pred.max():.4f}]")
    
    print(f"Generated pseudo-labels for {len(ensemble_mean)} samples")
    print(f"Mean prediction stats:")
    print(f"  Mean: {ensemble_mean.mean():.4f}")
    print(f"  Std: {ensemble_mean.std():.4f}")
    print(f"Uncertainty prediction stats:")
    print(f"  Mean: {ensemble_std.mean():.4f}")
    print(f"  Std: {ensemble_std.std():.4f}")
    
    return ensemble_mean, ensemble_std

def create_uncertainty_dataset(base_dataset, output_dir, ensemble_models=None):
    """Create uncertainty-aware pseudo-labeled dataset."""
    print("Creating uncertainty-aware pseudo-labeled dataset...")
    
    # Check if pseudo-labels already exist
    pseudo_data_path = os.path.join(output_dir, 'uncertainty_pseudo_labels.pt')
    
    if os.path.exists(pseudo_data_path):
        print(f"✓ Found existing uncertainty pseudo-labels at: {pseudo_data_path}")
        print("Loading existing pseudo-labels...")
        
        # Load existing pseudo-labels
        pseudo_data = torch.load(pseudo_data_path, map_location='cpu')
        mean_predictions = pseudo_data['mean_predictions']
        uncertainty_predictions = pseudo_data['uncertainty_predictions']
        
        print(f"✓ Loaded pseudo-labels for {len(mean_predictions)} samples")
    else:
        print("No existing pseudo-labels found. Generating new ones...")
        
        if ensemble_models is None or len(ensemble_models) == 0:
            print("✗ No ensemble models available for pseudo-label generation!")
            return None
        
        # Generate pseudo-labels on training data using same method as training script
        mean_predictions, uncertainty_predictions = generate_uncertainty_pseudo_labels(
            ensemble_models, base_dataset, device
        )
        
        # Save pseudo-labels for later use
        torch.save({
            'mean_predictions': mean_predictions,
            'uncertainty_predictions': uncertainty_predictions
        }, pseudo_data_path)
        print(f"✓ Pseudo-labels saved to: {pseudo_data_path}")
    
    # Create uncertainty dataset
    uncertainty_dataset = UncertaintyDataset(
        base_dataset.x_train,
        mean_predictions,
        uncertainty_predictions
    )
    
    print(f"Using all {len(uncertainty_dataset)} pseudo-labels for training")
    return uncertainty_dataset

class UncertaintyDataModule(pl.LightningDataModule):
    """DataModule for uncertainty-aware distillation."""
    
    def __init__(self, uncertainty_dataset, base_dataset):
        super().__init__()
        self.uncertainty_dataset = uncertainty_dataset
        self.base_dataset = base_dataset
        
    def train_dataloader(self):
        return DataLoader(
            self.uncertainty_dataset,
            batch_size=self.base_dataset.batch_size,
            shuffle=True,
            num_workers=0
        )
    
    def val_dataloader(self):
        # Use ground truth for validation
        return self.base_dataset.val_dataloader()
    
    def test_dataloader(self):
        return self.base_dataset.test_dataloader()

def train_uncertainty_student(uncertainty_dataset, base_dataset, output_dir, 
                            ensemble_models, num_epochs=100, learning_rate=0.001):
    """Train uncertainty-aware student model using exact same hyperparameters as ensemble.
    
    Uses multi-task loss: L = L_mean + λ * L_uncertainty where λ=1
    - L_mean = MSE between predicted mean and teacher ensemble mean
    - L_uncertainty = MSE between predicted uncertainty and teacher ensemble std
    """
    print("Training uncertainty-aware student model...")
    
    # Check if student model already exists
    student_checkpoint_path = os.path.join(output_dir, 'uncertainty_student.ckpt')
    
    if os.path.exists(student_checkpoint_path):
        print(f"✓ Found existing student model: {os.path.basename(student_checkpoint_path)}")
        print("Loading existing student model...")
        
        # Create model architecture using first teacher model
        teacher_model = ensemble_models[0]
        student_model = UncertaintyAwareStudent(teacher_model, num_tasks=2)
        lightning_model = UncertaintyAwareLightningModel(
            student_model, 
            learning_rate=learning_rate,
            weight_decay=1e-6,  # Same as ensemble models
            lambda_uncertainty=1.0  # λ = 1 as specified
        )
        
        # Load checkpoint
        lightning_model = UncertaintyAwareLightningModel.load_from_checkpoint(
            student_checkpoint_path, 
            student_model=student_model
        )
        print("✓ Student model loaded successfully")
        return lightning_model
    
    print("No existing student model found. Starting training...")
    
    # Create data module
    data_module = UncertaintyDataModule(uncertainty_dataset, base_dataset)
    
    # Create student model using first teacher model for backbone initialization
    teacher_model = ensemble_models[0]
    student_model = UncertaintyAwareStudent(teacher_model, num_tasks=2)
    lightning_model = UncertaintyAwareLightningModel(
        student_model, 
        learning_rate=learning_rate,
        weight_decay=1e-6,  # Same as ensemble models
        lambda_uncertainty=1.0  # λ = 1 as specified
    )
    
    # Create trainer with exact same hyperparameters as ensemble models
    callback_topmodel = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        save_top_k=1,
        dirpath=output_dir,
        filename='uncertainty_student'
    )
    callback_es = pl.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        logger=None,
        callbacks=[callback_es, callback_topmodel],
        accelerator='auto',
        devices='auto'
    )
    
    # Train model
    print("Starting uncertainty-aware training...")
    print(f"Using exact same hyperparameters as ensemble models:")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Weight decay: 1e-6")
    print(f"  - Max epochs: {num_epochs}")
    print(f"  - Early stopping patience: 10")
    print(f"  - Loss function: L = L_mean + λ * L_uncertainty (λ=1)")
    trainer.fit(lightning_model, datamodule=data_module)
    
    print("✓ Student model training completed")
    return lightning_model

def evaluate_uncertainty_student(student_model, base_dataset, plots_dir):
    """Evaluate the uncertainty-aware student model."""
    print("Evaluating uncertainty-aware student model...")
    
    # Get test data
    test_loader = base_dataset.test_dataloader()
    y_true = base_dataset.y_test
    
    # Get student predictions
    student_model.eval()
    student_device = next(student_model.parameters()).device
    
    with torch.no_grad():
        mean_preds = []
        uncertainty_preds = []
        
        for batch in test_loader:
            x = batch[0].to(student_device)
            outputs = student_model(x)
            mean_preds.append(outputs['mean'].cpu())
            uncertainty_preds.append(outputs['uncertainty'].cpu())
        
        mean_preds = torch.cat(mean_preds, dim=0)
        uncertainty_preds = torch.cat(uncertainty_preds, dim=0)
    
    # Calculate metrics
    print("Calculating performance metrics...")
    
    # Pearson correlation for mean predictions
    mean_pearson = [stats.pearsonr(y_true[:, i], mean_preds[:, i])[0] for i in range(y_true.shape[1])]
    
    # Spearman correlation for mean predictions
    mean_spearman = [stats.spearmanr(y_true[:, i], mean_preds[:, i])[0] for i in range(y_true.shape[1])]
    
    # Print results
    print("\n" + "="*60)
    print("UNCERTAINTY-AWARE STUDENT MODEL EVALUATION")
    print("="*60)
    print(f"Mean Prediction Pearson: {np.mean(mean_pearson):.4f} ± {np.std(mean_pearson):.4f}")
    print(f"Mean Prediction Spearman: {np.mean(mean_spearman):.4f} ± {np.std(mean_spearman):.4f}")
    print(f"Uncertainty Mean: {uncertainty_preds.mean():.4f}")
    print(f"Uncertainty Std: {uncertainty_preds.std():.4f}")
    print("="*60)
    
    return {
        'mean_pearson': mean_pearson,
        'mean_spearman': mean_spearman,
        'uncertainty_mean': uncertainty_preds.mean().item(),
        'uncertainty_std': uncertainty_preds.std().item()
    }

def save_student_performance_to_csv(results, output_dir):
    """Save student model performance to the ensemble performance CSV."""
    import pandas as pd
    
    csv_path = os.path.join(output_dir, 'ensemble_performance.csv')
    
    # Check if CSV already exists
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()
    
    # Prepare student model data
    student_data = {
        'seed': 'student',
        'model_type': 'uncertainty_student',
        'pearson_mean': np.mean(results['mean_pearson']),
        'pearson_std': np.std(results['mean_pearson']),
        'spearman_mean': np.mean(results['mean_spearman']),
        'spearman_std': np.std(results['mean_spearman'])
    }
    
    # Add student data to DataFrame
    new_df = pd.DataFrame([student_data])
    df = pd.concat([df, new_df], ignore_index=True)
    
    # Save updated CSV
    df.to_csv(csv_path, index=False)
    print(f"✓ Student model performance saved to: {csv_path}")
    
    return df

def create_plots_directory():
    """Create plots directory if it doesn't exist."""
    plots_dir = "Distillation_plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        print(f"Created plots directory: {plots_dir}")
    return plots_dir

def main():
    """Main function for uncertainty-aware distillation."""
    print("EvoAug2 Uncertainty-Aware Distillation")
    print("=" * 60)
    print("Distilling 10-model ensemble into uncertainty-aware student model")
    
    # Configuration
    expt_name = 'DeepSTARR_Uncertainty'
    data_path = '/grid/wsbs/home_norepl/pmantill/train_EvoAug/deepstarr-data.h5'
    output_dir = '/grid/wsbs/home_norepl/pmantill/train_EvoAug/EvoAug_ensemble_10'
    batch_size = 128
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = create_plots_directory()
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        print("Please download the DeepSTARR data first:")
        print("wget https://zenodo.org/record/7265991/files/DeepSTARR_data.h5")
        return
    
    # Load dataset
    print("Loading DeepSTARR dataset...")
    base_dataset = utils.H5Dataset(data_path, batch_size=batch_size, lower_case=False, transpose=False)
    print(f"Dataset loaded: {len(base_dataset)} sequences")
    
    # Load ensemble models
    ensemble_models = load_ensemble_models(output_dir)
    
    if len(ensemble_models) == 0:
        print("✗ No ensemble models found! Please train the ensemble first.")
        return
    
    # Create uncertainty-aware pseudo-labeled dataset
    uncertainty_dataset = create_uncertainty_dataset(base_dataset, output_dir, ensemble_models)
    
    if uncertainty_dataset is None:
        print("✗ Failed to create uncertainty dataset!")
        return
    
    # Train uncertainty-aware student model with exact same hyperparameters as ensemble
    student_model = train_uncertainty_student(
        uncertainty_dataset, 
        base_dataset, 
        output_dir, 
        ensemble_models,
        num_epochs=100,  # Same as ensemble Stage 1
        learning_rate=0.001  # Same as ensemble Stage 1
    )
    
    # Evaluate the model
    results = evaluate_uncertainty_student(student_model, base_dataset, plots_dir)
    
    # Save student performance to ensemble CSV
    save_student_performance_to_csv(results, output_dir)
    
    print("\n" + "="*60)
    print("UNCERTAINTY-AWARE DISTILLATION COMPLETED")
    print("="*60)
    print(f"Student model Pearson performance: {np.mean(results['mean_pearson']):.4f}")
    print(f"Student model Spearman performance: {np.mean(results['mean_spearman']):.4f}")
    print(f"Uncertainty prediction mean: {results['uncertainty_mean']:.4f}")
    print(f"Models saved to: {output_dir}")
    print(f"Plots saved to: {plots_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
