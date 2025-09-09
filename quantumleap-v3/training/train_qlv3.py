#!/usr/bin/env python3
"""
QuantumLeap v3 Training Script
PyTorch Lightning training pipeline with multi-task loss and W&B logging
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb
import yaml
import argparse
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import math

# Import our model architecture
import sys
sys.path.append('../models')
from qlv3_architecture import create_qlv3_model


class SquatDataset(torch.utils.data.Dataset):
    """
    Dataset loader for synthetic squat data
    """
    
    def __init__(
        self, 
        h5_path: str, 
        metadata_path: str,
        sequence_length: int = 100,
        split: str = 'train',
        train_ratio: float = 0.8,
        val_ratio: float = 0.1
    ):
        self.h5_path = h5_path
        self.sequence_length = sequence_length
        
        # Load metadata
        import json
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Create train/val/test splits
        total_samples = len(self.metadata)
        train_end = int(total_samples * train_ratio)
        val_end = int(total_samples * (train_ratio + val_ratio))
        
        if split == 'train':
            self.indices = list(range(0, train_end))
        elif split == 'val':
            self.indices = list(range(train_end, val_end))
        else:  # test
            self.indices = list(range(val_end, total_samples))
        
        print(f"{split} dataset: {len(self.indices)} samples")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        sample_idx = self.indices[idx]
        
        with h5py.File(self.h5_path, 'r') as h5f:
            # Load sensor data
            phone_imu = h5f['phone_imu'][sample_idx]  # (timesteps, 6)
            watch_imu = h5f['watch_imu'][sample_idx]  # (timesteps, 6)
            barometer = h5f['barometer'][sample_idx]  # (timesteps,)
            joint_positions = h5f['joint_positions'][sample_idx]  # (timesteps, 4)
            fatigue_labels = h5f['fatigue_labels'][sample_idx]  # (timesteps,)
        
        # Combine multi-modal input
        barometer = barometer.reshape(-1, 1)  # (timesteps, 1)
        sensor_data = np.concatenate([phone_imu, watch_imu, barometer], axis=1)  # (timesteps, 13)
        
        # Truncate or pad to sequence_length
        if sensor_data.shape[0] > self.sequence_length:
            # Random crop for training, center crop for val/test
            if sample_idx in self.indices and len(self.indices) == len(range(0, int(len(self.metadata) * 0.8))):  # train
                start_idx = np.random.randint(0, sensor_data.shape[0] - self.sequence_length + 1)
            else:
                start_idx = (sensor_data.shape[0] - self.sequence_length) // 2
            
            sensor_data = sensor_data[start_idx:start_idx + self.sequence_length]
            joint_positions = joint_positions[start_idx:start_idx + self.sequence_length]
            fatigue_labels = fatigue_labels[start_idx:start_idx + self.sequence_length]
        else:
            # Pad with zeros
            pad_length = self.sequence_length - sensor_data.shape[0]
            sensor_data = np.pad(sensor_data, ((0, pad_length), (0, 0)), mode='constant')
            joint_positions = np.pad(joint_positions, ((0, pad_length), (0, 0)), mode='constant')
            fatigue_labels = np.pad(fatigue_labels, (0, pad_length), mode='constant')
        
        # Get form errors from metadata
        form_errors = self.metadata[sample_idx]['form_errors']
        
        # Create form error labels (binary for each timestep)
        form_error_labels = {
            'knee_valgus': float('knee_valgus' in form_errors),
            'forward_lean': float('forward_lean' in form_errors),
            'insufficient_depth': float('insufficient_depth' in form_errors)
        }
        
        # Exercise type (just squats for now)
        exercise_label = 0  # squat = 0
        
        # Create rep detection labels (simplified - peak detection)
        rep_labels = self._create_rep_labels(joint_positions)
        
        return {
            'sensor_data': torch.FloatTensor(sensor_data),
            'joint_positions': torch.FloatTensor(joint_positions),
            'fatigue_labels': torch.FloatTensor(fatigue_labels),
            'form_error_labels': form_error_labels,
            'exercise_label': torch.LongTensor([exercise_label]),
            'rep_labels': torch.FloatTensor(rep_labels)
        }
    
    def _create_rep_labels(self, joint_positions: np.ndarray) -> np.ndarray:
        """Create rep detection labels from joint positions"""
        # Use knee angle as proxy for rep detection
        knee_angles = joint_positions[:, 2]  # Right knee
        
        # Find local minima (bottom of squat)
        rep_labels = np.zeros_like(knee_angles)
        
        # Simple peak detection
        for i in range(1, len(knee_angles) - 1):
            if knee_angles[i] < knee_angles[i-1] and knee_angles[i] < knee_angles[i+1]:
                if knee_angles[i] < -60:  # Significant squat depth
                    rep_labels[i] = 1.0
        
        return rep_labels


class QLv3LightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for QuantumLeap v3 training
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        
        # Create model
        self.model = create_qlv3_model(config)
        
        # Loss weights
        self.loss_weights = config['loss_weights']
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
    
    def forward(self, x, mask=None):
        return self.model(x, mask)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config['training']['max_epochs'],
            eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, 'val')
    
    def _shared_step(self, batch, stage):
        sensor_data = batch['sensor_data']
        joint_positions = batch['joint_positions']
        fatigue_labels = batch['fatigue_labels']
        form_error_labels = batch['form_error_labels']
        exercise_labels = batch['exercise_label'].squeeze()
        rep_labels = batch['rep_labels']
        
        # Forward pass
        predictions, aux_info = self.model(sensor_data)
        
        # Calculate losses
        losses = self._calculate_losses(predictions, aux_info, batch)
        
        # Total loss
        total_loss = (
            self.loss_weights['pose_regression'] * losses['pose_loss'] +
            self.loss_weights['classification'] * losses['classification_loss'] +
            self.loss_weights['vq_commitment'] * losses['vq_loss']
        )
        
        # Log metrics
        self.log(f'{stage}_loss', total_loss, prog_bar=True)
        self.log(f'{stage}_pose_loss', losses['pose_loss'])
        self.log(f'{stage}_classification_loss', losses['classification_loss'])
        self.log(f'{stage}_vq_loss', losses['vq_loss'])
        self.log(f'{stage}_vq_perplexity', aux_info['vq_perplexity'])
        
        # Additional metrics
        if stage == 'val':
            # Calculate pose estimation accuracy (MPJPE)
            pose_error = torch.mean(torch.abs(predictions['pose_mean'] - joint_positions))
            self.log('val_mpjpe', pose_error)
            
            # Form error detection accuracy
            for error_type in form_error_labels:
                pred_probs = predictions['form_errors'][error_type].mean(dim=1)  # Average over time
                true_labels = torch.tensor([form_error_labels[error_type][i] for i in range(len(form_error_labels[error_type]))]).float().to(self.device)
                
                # Binary accuracy
                pred_binary = (pred_probs > 0.5).float()
                accuracy = (pred_binary == true_labels).float().mean()
                self.log(f'val_{error_type}_accuracy', accuracy)
        
        return total_loss
    
    def _calculate_losses(self, predictions, aux_info, batch):
        """Calculate all loss components"""
        
        # 1. Pose regression loss (negative log-likelihood of Gaussian)
        pose_mean = predictions['pose_mean']
        pose_std = predictions['pose_std']
        joint_positions = batch['joint_positions']
        
        # Negative log-likelihood
        pose_loss = 0.5 * torch.mean(
            torch.log(2 * math.pi * pose_std**2) + 
            ((joint_positions - pose_mean) / pose_std)**2
        )
        
        # 2. Classification losses
        exercise_loss = F.cross_entropy(
            predictions['exercise_logits'].mean(dim=1),  # Average over time
            batch['exercise_label'].squeeze()
        )
        
        # Form error detection losses
        form_error_losses = []
        for error_type, pred_probs in predictions['form_errors'].items():
            true_labels = torch.tensor([
                batch['form_error_labels'][error_type][i] 
                for i in range(len(batch['form_error_labels'][error_type]))
            ]).float().to(self.device)
            
            # Binary cross-entropy over time-averaged predictions
            pred_avg = pred_probs.mean(dim=1)
            form_loss = F.binary_cross_entropy(pred_avg, true_labels)
            form_error_losses.append(form_loss)
        
        classification_loss = exercise_loss + torch.stack(form_error_losses).mean()
        
        # 3. VQ commitment loss
        vq_loss = aux_info['vq_commit_loss']
        
        # 4. Cognitive state loss (fatigue prediction)
        fatigue_pred = predictions['cognitive_state'][:, :, 0]  # First dimension is fatigue
        fatigue_true = batch['fatigue_labels']
        cognitive_loss = F.mse_loss(fatigue_pred, fatigue_true)
        
        # 5. Rep detection loss
        rep_pred = predictions['rep_probability'].squeeze(-1)
        rep_true = batch['rep_labels']
        rep_loss = F.binary_cross_entropy(rep_pred, rep_true)
        
        return {
            'pose_loss': pose_loss,
            'classification_loss': classification_loss + cognitive_loss + rep_loss,
            'vq_loss': vq_loss
        }


def create_data_loaders(config: Dict[str, Any], data_path: str, metadata_path: str):
    """Create train/val/test data loaders"""
    
    # Create datasets
    train_dataset = SquatDataset(
        data_path, metadata_path, 
        sequence_length=config['data']['sequence_length'],
        split='train'
    )
    
    val_dataset = SquatDataset(
        data_path, metadata_path,
        sequence_length=config['data']['sequence_length'], 
        split='val'
    )
    
    test_dataset = SquatDataset(
        data_path, metadata_path,
        sequence_length=config['data']['sequence_length'],
        split='test'
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description="Train QuantumLeap v3 model")
    parser.add_argument("--config", required=True, help="Training configuration file")
    parser.add_argument("--data", required=True, help="Path to HDF5 dataset")
    parser.add_argument("--metadata", required=True, help="Path to metadata JSON")
    parser.add_argument("--resume", help="Resume from checkpoint")
    parser.add_argument("--test-only", action="store_true", help="Only run testing")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize wandb logger
    wandb_logger = WandbLogger(
        project=config['logging']['project'],
        entity=config['logging']['entity'],
        name=f"{config['model']['name']}_run",
        config=config
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        config, args.data, args.metadata
    )
    
    # Create model
    model = QLv3LightningModule(config)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            filename='qlv3-{epoch:02d}-{val_loss:.3f}'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min'
        ),
        LearningRateMonitor(logging_interval='step')
    ]
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        logger=wandb_logger,
        callbacks=callbacks,
        gradient_clip_val=config['training']['gradient_clip_val'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision=16,  # Mixed precision for efficiency
        log_every_n_steps=config['logging']['log_every_n_steps']
    )
    
    if args.test_only:
        # Load checkpoint and test
        if args.resume:
            model = QLv3LightningModule.load_from_checkpoint(args.resume, config=config)
        trainer.test(model, test_loader)
    else:
        # Train model
        trainer.fit(
            model, 
            train_loader, 
            val_loader,
            ckpt_path=args.resume if args.resume else None
        )
        
        # Test best model
        trainer.test(ckpt_path='best', dataloaders=test_loader)
    
    # Log model size
    model_stats = model.model.get_model_size()
    wandb.log(model_stats)
    
    print(f"Training complete! Model size: {model_stats['model_size_mb']:.1f} MB")


if __name__ == "__main__":
    main()
