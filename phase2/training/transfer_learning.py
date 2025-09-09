#!/usr/bin/env python3
"""
Transfer Learning Pipeline for Real Human Data
Fine-tunes synthetic-trained model on real human squat data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import h5py
from pathlib import Path
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Import the model architecture
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from quantumleap-v3.training.simple_trainer import QuantumLeapV3Model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HumanSquatDataset(Dataset):
    """Dataset for real human squat data"""
    
    def __init__(self, human_data_dir, sequence_length=200, transform=None):
        self.human_data_dir = Path(human_data_dir)
        self.sequence_length = sequence_length
        self.transform = transform
        
        # Load all human sessions
        self.samples = self._load_human_samples()
        logger.info(f"Loaded {len(self.samples)} human samples")
    
    def _load_human_samples(self):
        """Load and process human session data"""
        
        samples = []
        
        for session_file in self.human_data_dir.glob("human_squat_*.json"):
            try:
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                
                # Extract sensor data
                phone_imu = np.array(session_data['sensor_data']['phone_imu'])
                watch_imu = np.array(session_data['sensor_data']['watch_imu'])
                barometer = np.array(session_data['sensor_data']['barometer'])
                
                # Create sequences
                session_samples = self._create_sequences(phone_imu, watch_imu, barometer, session_data)
                samples.extend(session_samples)
                
            except Exception as e:
                logger.error(f"Error loading {session_file}: {e}")
        
        return samples
    
    def _create_sequences(self, phone_imu, watch_imu, barometer, session_data):
        """Create training sequences from session data"""
        
        if len(phone_imu) < self.sequence_length:
            return []
        
        samples = []
        rep_timestamps = session_data['analysis'].get('rep_timestamps', [])
        
        # Create overlapping sequences
        step_size = self.sequence_length // 4  # 75% overlap
        
        for start_idx in range(0, len(phone_imu) - self.sequence_length, step_size):
            end_idx = start_idx + self.sequence_length
            
            # Extract sequence data
            phone_seq = phone_imu[start_idx:end_idx]
            watch_seq = watch_imu[start_idx:end_idx] if len(watch_imu) > end_idx else np.zeros((self.sequence_length, 6))
            baro_seq = barometer[start_idx:end_idx] if len(barometer) > end_idx else np.zeros(self.sequence_length)
            
            # Create labels for this sequence
            labels = self._create_sequence_labels(start_idx, end_idx, rep_timestamps, session_data)
            
            sample = {
                'phone_imu': phone_seq,
                'watch_imu': watch_seq,
                'barometer': baro_seq,
                'labels': labels
            }
            
            samples.append(sample)
        
        return samples
    
    def _create_sequence_labels(self, start_idx, end_idx, rep_timestamps, session_data):
        """Create labels for sequence"""
        
        # Count reps in this sequence (simplified)
        sequence_duration = (end_idx - start_idx) / 100.0  # Assuming 100Hz
        sequence_start_time = start_idx / 100.0
        
        reps_in_sequence = sum(1 for t in rep_timestamps 
                              if sequence_start_time <= t <= sequence_start_time + sequence_duration)
        
        # Create multi-task labels
        labels = {
            'rep_count': min(reps_in_sequence, 5),  # Cap at 5 reps
            'exercise_type': 0,  # 0 = squat
            'form_quality': session_data['analysis']['quality_metrics'].get('quality_score', 0.8),
            'cognitive_state': 0.7  # Default moderate focus
        }
        
        return labels
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Convert to tensors
        phone_imu = torch.FloatTensor(sample['phone_imu']).transpose(0, 1)  # (6, seq_len)
        watch_imu = torch.FloatTensor(sample['watch_imu']).transpose(0, 1)  # (6, seq_len)
        barometer = torch.FloatTensor(sample['barometer']).unsqueeze(0)  # (1, seq_len)
        
        # Combine all sensors
        x = torch.cat([phone_imu, watch_imu, barometer], dim=0)  # (13, seq_len)
        
        # Labels
        labels = sample['labels']
        rep_count = torch.LongTensor([labels['rep_count']])
        exercise_type = torch.LongTensor([labels['exercise_type']])
        form_quality = torch.FloatTensor([labels['form_quality']])
        cognitive_state = torch.FloatTensor([labels['cognitive_state']])
        
        if self.transform:
            x = self.transform(x)
        
        return x, (rep_count, exercise_type, form_quality, cognitive_state)

class TransferLearningTrainer:
    """Transfer learning trainer for human data fine-tuning"""
    
    def __init__(self, pretrained_model_path, device='cpu'):
        self.device = torch.device(device)
        self.pretrained_model_path = pretrained_model_path
        
        # Load pretrained model
        self.model = self._load_pretrained_model()
        self.model.to(self.device)
        
        # Setup for transfer learning
        self._setup_transfer_learning()
        
        logger.info(f"Transfer learning trainer initialized on {device}")
    
    def _load_pretrained_model(self):
        """Load pretrained model from synthetic training"""
        
        # Create model architecture
        model = QuantumLeapV3Model(
            input_channels=13,  # phone(6) + watch(6) + barometer(1)
            sequence_length=200,
            num_classes=6,  # rep count classes (0-5)
            hidden_dim=128
        )
        
        # Load pretrained weights
        if Path(self.pretrained_model_path).exists():
            checkpoint = torch.load(self.pretrained_model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded pretrained model from {self.pretrained_model_path}")
        else:
            logger.warning(f"Pretrained model not found: {self.pretrained_model_path}")
            logger.info("Starting with randomly initialized weights")
        
        return model
    
    def _setup_transfer_learning(self):
        """Setup model for transfer learning"""
        
        # Freeze early layers (feature extraction)
        for name, param in self.model.named_parameters():
            if 'cnn_extractor' in name:
                param.requires_grad = False  # Freeze CNN layers
            elif 'lstm' in name:
                param.requires_grad = True   # Fine-tune LSTM
            else:
                param.requires_grad = True   # Fine-tune output layers
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    def train(self, human_data_dir, epochs=20, batch_size=16, learning_rate=1e-4):
        """Train model on human data"""
        
        # Create dataset
        dataset = HumanSquatDataset(human_data_dir)
        
        if len(dataset) == 0:
            logger.error("No human data found for training")
            return None
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup optimizer and loss
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), 
                              lr=learning_rate)
        
        # Multiple loss functions for multi-task learning
        rep_criterion = nn.CrossEntropyLoss()
        exercise_criterion = nn.CrossEntropyLoss()
        quality_criterion = nn.MSELoss()
        cognitive_criterion = nn.MSELoss()
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        logger.info(f"Starting transfer learning training...")
        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
                data = data.to(self.device)
                rep_targets, exercise_targets, quality_targets, cognitive_targets = targets
                
                rep_targets = rep_targets.squeeze().to(self.device)
                exercise_targets = exercise_targets.squeeze().to(self.device)
                quality_targets = quality_targets.squeeze().to(self.device)
                cognitive_targets = cognitive_targets.squeeze().to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(data)
                rep_pred, exercise_pred, quality_pred, cognitive_pred = outputs
                
                # Calculate losses
                rep_loss = rep_criterion(rep_pred, rep_targets)
                exercise_loss = exercise_criterion(exercise_pred, exercise_targets)
                quality_loss = quality_criterion(quality_pred, quality_targets)
                cognitive_loss = cognitive_criterion(cognitive_pred, cognitive_targets)
                
                # Combined loss
                total_loss = rep_loss + exercise_loss + 0.5 * quality_loss + 0.5 * cognitive_loss
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                train_loss += total_loss.item()
                
                # Calculate accuracy (rep count)
                _, predicted = torch.max(rep_pred.data, 1)
                train_total += rep_targets.size(0)
                train_correct += (predicted == rep_targets).sum().item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, targets in val_loader:
                    data = data.to(self.device)
                    rep_targets, exercise_targets, quality_targets, cognitive_targets = targets
                    
                    rep_targets = rep_targets.squeeze().to(self.device)
                    exercise_targets = exercise_targets.squeeze().to(self.device)
                    quality_targets = quality_targets.squeeze().to(self.device)
                    cognitive_targets = cognitive_targets.squeeze().to(self.device)
                    
                    outputs = self.model(data)
                    rep_pred, exercise_pred, quality_pred, cognitive_pred = outputs
                    
                    # Calculate losses
                    rep_loss = rep_criterion(rep_pred, rep_targets)
                    exercise_loss = exercise_criterion(exercise_pred, exercise_targets)
                    quality_loss = quality_criterion(quality_pred, quality_targets)
                    cognitive_loss = cognitive_criterion(cognitive_pred, cognitive_targets)
                    
                    total_loss = rep_loss + exercise_loss + 0.5 * quality_loss + 0.5 * cognitive_loss
                    val_loss += total_loss.item()
                    
                    # Calculate accuracy
                    _, predicted = torch.max(rep_pred.data, 1)
                    val_total += rep_targets.size(0)
                    val_correct += (predicted == rep_targets).sum().item()
            
            # Calculate metrics
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            
            # Store history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            logger.info(f"Epoch {epoch+1}/{epochs}")
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Early stopping
            if epoch > 5 and val_loss > min(history['val_loss'][:-1]):
                logger.info("Early stopping triggered")
                break
        
        return history
    
    def save_model(self, output_path):
        """Save fine-tuned model"""
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_channels': 13,
                'sequence_length': 200,
                'num_classes': 6,
                'hidden_dim': 128
            }
        }, output_path)
        
        logger.info(f"Fine-tuned model saved to {output_path}")
    
    def evaluate_on_human_data(self, human_data_dir):
        """Evaluate model performance on human test data"""
        
        # Create test dataset
        test_dataset = HumanSquatDataset(human_data_dir)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in test_loader:
                data = data.to(self.device)
                rep_targets = targets[0].squeeze().to(self.device)
                
                outputs = self.model(data)
                rep_pred = outputs[0]
                
                _, predicted = torch.max(rep_pred.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(rep_targets.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        report = classification_report(all_targets, all_predictions)
        
        logger.info(f"Human data evaluation accuracy: {accuracy:.3f}")
        logger.info(f"Classification report:\n{report}")
        
        return accuracy, report

def main():
    """Main transfer learning interface"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Transfer Learning for Human Data")
    parser.add_argument("--pretrained", required=True, help="Path to pretrained model")
    parser.add_argument("--human_data", required=True, help="Path to human data directory")
    parser.add_argument("--output", default="./human_finetuned_model.pth", help="Output model path")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", default="cpu", help="Training device")
    
    args = parser.parse_args()
    
    print("🔄 Transfer Learning for Human Data")
    print(f"Pretrained model: {args.pretrained}")
    print(f"Human data: {args.human_data}")
    print(f"Output: {args.output}")
    
    # Create trainer
    trainer = TransferLearningTrainer(args.pretrained, args.device)
    
    # Train on human data
    print("\n🚀 Starting transfer learning...")
    history = trainer.train(
        human_data_dir=args.human_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    
    if history:
        # Save fine-tuned model
        trainer.save_model(args.output)
        
        # Evaluate performance
        print("\n📊 Evaluating on human data...")
        accuracy, report = trainer.evaluate_on_human_data(args.human_data)
        
        print(f"\n✅ Transfer learning complete!")
        print(f"Final accuracy on human data: {accuracy:.3f}")
        print(f"Model saved to: {args.output}")
    else:
        print("❌ Transfer learning failed - check human data availability")

if __name__ == "__main__":
    main()
