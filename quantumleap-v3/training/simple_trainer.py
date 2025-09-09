#!/usr/bin/env python3
"""
Simplified QuantumLeap v3 Training Script
Fast training pipeline without external dependencies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SquatDataset(Dataset):
    """Dataset loader for synthetic squat data"""
    
    def __init__(self, h5_path, split='train', train_ratio=0.8, val_ratio=0.1):
        self.h5_path = h5_path
        
        with h5py.File(h5_path, 'r') as f:
            self.num_samples = f['phone_imu'].shape[0]
            self.sequence_length = f['phone_imu'].shape[1]
            
            # Load metadata
            metadata_str = f.attrs['metadata']
            self.metadata = json.loads(metadata_str)
        
        # Create splits
        train_end = int(train_ratio * self.num_samples)
        val_end = int((train_ratio + val_ratio) * self.num_samples)
        
        if split == 'train':
            self.indices = list(range(0, train_end))
        elif split == 'val':
            self.indices = list(range(train_end, val_end))
        else:  # test
            self.indices = list(range(val_end, self.num_samples))
        
        logger.info(f"{split} split: {len(self.indices)} samples")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        
        with h5py.File(self.h5_path, 'r') as f:
            # Load sensor data
            phone_imu = f['phone_imu'][real_idx]  # (seq_len, 6)
            watch_imu = f['watch_imu'][real_idx]  # (seq_len, 6)
            barometer = f['barometer'][real_idx]  # (seq_len,)
            
            # Load labels
            rep_detection = f['rep_detection'][real_idx]  # (seq_len,)
            exercise_class = f['exercise_class'][real_idx]  # scalar
            form_quality = f['form_quality'][real_idx]  # scalar
            cognitive_state = f['cognitive_state'][real_idx]  # (2,)
        
        # Combine inputs: phone (6) + watch (6) + barometer (1) = 13 channels
        barometer_expanded = barometer[:, np.newaxis]  # (seq_len, 1)
        inputs = np.concatenate([phone_imu, watch_imu, barometer_expanded], axis=1)
        
        # Convert to tensors
        inputs = torch.FloatTensor(inputs).transpose(0, 1)  # (13, seq_len)
        rep_detection = torch.FloatTensor(rep_detection)
        exercise_class = torch.LongTensor([int(exercise_class)])
        form_quality = torch.FloatTensor([form_quality])
        cognitive_state = torch.FloatTensor(cognitive_state)
        
        return {
            'inputs': inputs,
            'rep_detection': rep_detection,
            'exercise_class': exercise_class,
            'form_quality': form_quality,
            'cognitive_state': cognitive_state
        }

class SimpleQuantumLeapV3(nn.Module):
    """Simplified QuantumLeap v3 model for fast training"""
    
    def __init__(self, input_channels=13, hidden_dim=128, num_layers=4):
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        
        # 1D CNN feature extractor
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            hidden_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=0.1,
            bidirectional=True
        )
        
        lstm_output_dim = hidden_dim * 2  # bidirectional
        
        # Multi-task heads
        self.rep_head = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.exercise_head = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)  # squat vs other
        )
        
        self.quality_head = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.cognitive_head = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2),  # fatigue, focus
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        # x: (batch, channels, seq_len)
        
        # CNN feature extraction
        conv_out = self.conv_layers(x)  # (batch, hidden_dim, seq_len)
        
        # Transpose for LSTM: (batch, seq_len, hidden_dim)
        lstm_in = conv_out.transpose(1, 2)
        
        # LSTM temporal modeling
        lstm_out, _ = self.lstm(lstm_in)  # (batch, seq_len, hidden_dim*2)
        
        # Multi-task outputs
        rep_detection = self.rep_head(lstm_out).squeeze(-1)  # (batch, seq_len)
        
        # Use mean pooling for sequence-level predictions
        pooled = lstm_out.mean(dim=1)  # (batch, hidden_dim*2)
        
        exercise_class = self.exercise_head(pooled)  # (batch, 2)
        form_quality = self.quality_head(pooled).squeeze(-1)  # (batch,)
        cognitive_state = self.cognitive_head(pooled)  # (batch, 2)
        
        return rep_detection, exercise_class, form_quality, cognitive_state

class Trainer:
    """Simple training loop"""
    
    def __init__(self, model, train_loader, val_loader, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Optimizer
        self.optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Loss functions
        self.rep_criterion = nn.BCELoss()
        self.exercise_criterion = nn.CrossEntropyLoss()
        self.quality_criterion = nn.MSELoss()
        self.cognitive_criterion = nn.MSELoss()
        
        # Loss weights
        self.loss_weights = {
            'rep': 2.0,
            'exercise': 0.5,
            'quality': 1.0,
            'cognitive': 0.5
        }
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            # Move to device
            inputs = batch['inputs'].to(self.device)
            rep_labels = batch['rep_detection'].to(self.device)
            exercise_labels = batch['exercise_class'].squeeze().to(self.device)
            quality_labels = batch['form_quality'].squeeze().to(self.device)
            cognitive_labels = batch['cognitive_state'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Calculate losses
            rep_loss = self.rep_criterion(outputs['rep_detection'], rep_labels)
            exercise_loss = self.exercise_criterion(outputs['exercise_class'], exercise_labels)
            quality_loss = self.quality_criterion(outputs['form_quality'], quality_labels)
            cognitive_loss = self.cognitive_criterion(outputs['cognitive_state'], cognitive_labels)
            
            # Weighted total loss
            total_batch_loss = (
                self.loss_weights['rep'] * rep_loss +
                self.loss_weights['exercise'] * exercise_loss +
                self.loss_weights['quality'] * quality_loss +
                self.loss_weights['cognitive'] * cognitive_loss
            )
            
            # Backward pass
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += total_batch_loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{total_batch_loss.item():.4f}'})
        
        return total_loss / num_batches
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                inputs = batch['inputs'].to(self.device)
                rep_labels = batch['rep_detection'].to(self.device)
                exercise_labels = batch['exercise_class'].squeeze().to(self.device)
                quality_labels = batch['form_quality'].squeeze().to(self.device)
                cognitive_labels = batch['cognitive_state'].to(self.device)
                
                outputs = self.model(inputs)
                
                rep_loss = self.rep_criterion(outputs['rep_detection'], rep_labels)
                exercise_loss = self.exercise_criterion(outputs['exercise_class'], exercise_labels)
                quality_loss = self.quality_criterion(outputs['form_quality'], quality_labels)
                cognitive_loss = self.cognitive_criterion(outputs['cognitive_state'], cognitive_labels)
                
                total_batch_loss = (
                    self.loss_weights['rep'] * rep_loss +
                    self.loss_weights['exercise'] * exercise_loss +
                    self.loss_weights['quality'] * quality_loss +
                    self.loss_weights['cognitive'] * cognitive_loss
                )
                
                total_loss += total_batch_loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, num_epochs=20):
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 5
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
                logger.info("New best model saved!")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping after {epoch+1} epochs")
                    break
        
        logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../data/synthetic_squat_dataset.h5")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Create datasets
    train_dataset = SquatDataset(args.data, split='train')
    val_dataset = SquatDataset(args.data, split='val')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0  # Avoid multiprocessing issues
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    model = SimpleQuantumLeapV3(input_channels=13, hidden_dim=128, num_layers=4)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, device)
    
    # Train
    trainer.train(num_epochs=args.epochs)
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
