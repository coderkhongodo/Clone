#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PhoBERT Training với VnCoreNLP Preprocessed Data
Modified version để sử dụng dữ liệu đã được xử lý bằng VnCoreNLP
"""

import os
import json
import pickle
import argparse
import random
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from tqdm import tqdm
from collections import Counter

# Import classes từ script gốc
import sys
sys.path.append('scripts/phobert')
from train_phobert import (
    FocalLoss, LabelSmoothingCrossEntropy, WeightedFocalLoss, F1Loss,
    PhoBERTClassifier, Trainer, set_seed
)

class VnCoreNLPClickbaitDataset(Dataset):
    """Dataset cho PhoBERT với dữ liệu đã xử lý bằng VnCoreNLP"""
    
    def __init__(self, data_path, balance_data=False, balance_strategy='oversample'):
        """
        Load dữ liệu đã preprocessing bằng VnCoreNLP
        
        Args:
            data_path: Path to pickle file đã xử lý
            balance_data: Whether to balance the dataset
            balance_strategy: 'oversample' hoặc 'undersample'
        """
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        self.input_ids = self.data['input_ids']
        self.attention_mask = self.data['attention_mask']
        
        # Labels đã được encode sẵn
        raw_labels = self.data['labels']
        if isinstance(raw_labels, torch.Tensor):
            self.labels = raw_labels.tolist()
        else:
            self.labels = list(raw_labels)
        
        # Convert to integers
        self.labels = [int(label) for label in self.labels]
        
        original_size = len(self.input_ids)
        print(f"📊 STATS: Dataset từ {data_path}: {original_size} samples")
        
        # In thống kê phân bố class
        label_counts = Counter(self.labels)
        print(f"📊 STATS: Phân bố class:")
        
        total_samples = len(self.labels)
        for label in [0, 1]:
            count = label_counts.get(label, 0)
            percentage = (count / total_samples) * 100 if total_samples > 0 else 0
            class_name = 'clickbait' if label == 0 else 'non-clickbait'
            print(f"   {class_name}: {count} ({percentage:.1f}%)")
        
        # Data balancing nếu được yêu cầu
        if balance_data and 'train' in data_path:
            print(f"⚖️ Áp dụng data balancing: {balance_strategy}")
            self._balance_dataset(balance_strategy)
            
            new_size = len(self.input_ids)
            print(f"📊 STATS: Sau balancing: {new_size} samples")
            
            # In lại thống kê sau balancing
            label_counts = Counter(self.labels)
            print(f"📊 STATS: Phân bố class sau balancing:")
            total_samples = len(self.labels)
            for label in [0, 1]:
                count = label_counts.get(label, 0)
                percentage = (count / total_samples) * 100 if total_samples > 0 else 0
                class_name = 'clickbait' if label == 0 else 'non-clickbait'
                print(f"   {class_name}: {count} ({percentage:.1f}%)")
    
    def _balance_dataset(self, strategy='oversample'):
        """Balance dataset using oversample hoặc undersample"""
        labels_array = np.array(self.labels)
        
        # Tìm class counts
        unique_labels, counts = np.unique(labels_array, return_counts=True)
        
        if strategy == 'oversample':
            # Oversample minority class
            max_count = max(counts)
            
            all_indices = []
            for label in unique_labels:
                label_indices = np.where(labels_array == label)[0]
                
                if len(label_indices) < max_count:
                    # Oversample
                    additional_needed = max_count - len(label_indices)
                    additional_indices = np.random.choice(
                        label_indices, 
                        size=additional_needed, 
                        replace=True
                    )
                    combined_indices = np.concatenate([label_indices, additional_indices])
                else:
                    combined_indices = label_indices
                
                all_indices.extend(combined_indices)
            
            # Shuffle indices
            all_indices = np.array(all_indices)
            np.random.shuffle(all_indices)
            
        elif strategy == 'undersample':
            # Undersample majority class
            min_count = min(counts)
            
            all_indices = []
            for label in unique_labels:
                label_indices = np.where(labels_array == label)[0]
                
                if len(label_indices) > min_count:
                    # Undersample
                    selected_indices = np.random.choice(
                        label_indices, 
                        size=min_count, 
                        replace=False
                    )
                else:
                    selected_indices = label_indices
                
                all_indices.extend(selected_indices)
            
            # Shuffle indices
            all_indices = np.array(all_indices)
            np.random.shuffle(all_indices)
        
        # Apply resampling
        self.input_ids = self.input_ids[all_indices]
        self.attention_mask = self.attention_mask[all_indices]
        self.labels = [self.labels[i] for i in all_indices]
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def save_test_results(trainer, test_loader, output_dir, config):
    """
    Lưu kết quả test vào file JSON với 4 chữ số thập phân
    """
    import torch
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
    
    # Set model to eval mode
    trainer.model.eval()
    
    device = next(trainer.model.parameters()).device
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    print("🧪 Đang chạy test...")
    
    # Evaluate on test set
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = trainer.model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=-1)
            _, predicted = torch.max(logits.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, zero_division=0
    )
    
    # Overall metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted', zero_division=0
    )
    
    # Detailed classification report
    class_names = ['clickbait', 'non-clickbait']
    report = classification_report(
        all_labels, all_predictions, 
        target_names=class_names, 
        output_dict=True,
        zero_division=0
    )
    
    # Create results dictionary with 4 decimal places
    results = {
        "model_info": {
            "model_name": config.get("model_name", ""),
            "data_source": "VnCoreNLP_preprocessed",
            "total_params": config.get("total_params", 0),
            "trainable_params": config.get("trainable_params", 0)
        },
        "test_metrics": {
            "accuracy": round(accuracy, 4),
            "macro_avg": {
                "precision": round(precision_macro, 4),
                "recall": round(recall_macro, 4),
                "f1_score": round(f1_macro, 4)
            },
            "weighted_avg": {
                "precision": round(precision_weighted, 4),
                "recall": round(recall_weighted, 4),
                "f1_score": round(f1_weighted, 4)
            }
        },
        "per_class_metrics": {},
        "confusion_matrix": {
            "labels": class_names,
            "matrix": []
        },
        "sample_counts": {
            "total_test_samples": len(all_labels),
            "clickbait": int(support[0]),
            "non_clickbait": int(support[1])
        }
    }
    
    # Per-class metrics with 4 decimal places
    for i, class_name in enumerate(class_names):
        results["per_class_metrics"][class_name] = {
            "precision": round(precision[i], 4),
            "recall": round(recall[i], 4),
            "f1_score": round(f1[i], 4),
            "support": int(support[i])
        }
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_predictions)
    results["confusion_matrix"]["matrix"] = cm.tolist()
    
    # Save to JSON file
    import json
    os.makedirs(output_dir, exist_ok=True)
    
    results_file = os.path.join(output_dir, 'test_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save detailed classification report
    report_file = os.path.join(output_dir, 'classification_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("VnCoreNLP PhoBERT Test Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {config.get('model_name', '')}\n")
        f.write(f"Data: VnCoreNLP preprocessed\n")
        f.write(f"Test Samples: {len(all_labels)}\n\n")
        
        f.write("Classification Report:\n")
        f.write(classification_report(
            all_labels, all_predictions, 
            target_names=class_names,
            digits=4,
            zero_division=0
        ))
        
        f.write(f"\n\nConfusion Matrix:\n")
        f.write(f"             Predicted\n")
        f.write(f"Actual    {class_names[0]:<12} {class_names[1]:<12}\n")
        for i, actual_class in enumerate(class_names):
            f.write(f"{actual_class:<9} {cm[i][0]:<12} {cm[i][1]:<12}\n")
    
    print(f"✅ Đã lưu kết quả test:")
    print(f"   📋 JSON: {results_file}")
    print(f"   📄 Report: {report_file}")
    print(f"🎯 Test Accuracy: {accuracy:.4f}")
    print(f"🎯 Test F1-Score: {f1_weighted:.4f}")

def main():
    parser = argparse.ArgumentParser(description="PhoBERT Training với VnCoreNLP Data")
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='vinai/phobert-base',
                       choices=['vinai/phobert-base', 'vinai/phobert-large'],
                       help='PhoBERT model name')
    parser.add_argument('--data_dir', type=str, default='data-vncorenlp-v2',
                       help='Directory chứa dữ liệu đã xử lý bằng VnCoreNLP')
    parser.add_argument('--output_dir', type=str, default='checkpoints-vncorenlp',
                       help='Output directory for checkpoints')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=15,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.15,
                       help='Warmup ratio')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['linear', 'cosine', 'cosine_restarts'],
                       help='Learning rate scheduler')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                       help='Gradient accumulation steps')
    
    # Model specific
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--num_classes', type=int, default=2,
                       help='Number of classes')
    parser.add_argument('--num_freeze_layers', type=int, default=6,
                       help='Number of layers to freeze (0 = no freezing)')
    
    # Loss function
    parser.add_argument('--loss_type', type=str, default='f1',
                       choices=['cross_entropy', 'focal', 'weighted_focal', 'f1', 'label_smoothing'],
                       help='Loss function type')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                       help='Label smoothing factor')
    
    # Model architecture
    parser.add_argument('--pooling_strategy', type=str, default='cls_mean',
                       choices=['cls', 'cls_mean', 'attention'],
                       help='Pooling strategy for sentence representation')
    
    # Training options
    parser.add_argument('--use_amp', action='store_true',
                       help='Use mixed precision training')
    parser.add_argument('--use_ema', action='store_true', default=True,
                       help='Use Exponential Moving Average')
    parser.add_argument('--ema_decay', type=float, default=0.9999,
                       help='EMA decay rate')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience')
    
    # Advanced optimizations
    parser.add_argument('--progressive_unfreezing', action='store_true',
                       help='Progressive unfreezing of layers during training')
    parser.add_argument('--lr_restarts', type=str, default='',
                       help='Epochs for learning rate restarts (comma-separated: 3,6,9)')
    parser.add_argument('--stochastic_depth', type=float, default=0.0,
                       help='Stochastic depth rate (0.0 = disabled)')
    parser.add_argument('--rdrop_alpha', type=float, default=0.0,
                       help='R-Drop regularization alpha (0.0 = disabled)')
    parser.add_argument('--swa_start', type=int, default=-1,
                       help='Epoch to start Stochastic Weight Averaging (-1 = disabled)')
    parser.add_argument('--swa_lr', type=float, default=1e-5,
                       help='Learning rate for SWA')
    parser.add_argument('--layer_lr_decay', type=float, default=1.0,
                       help='Layer-wise learning rate decay (1.0 = disabled)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Focal loss gamma parameter')
    parser.add_argument('--adversarial_training', action='store_true',
                       help='Enable Fast Gradient Method (FGM) adversarial training')
    
    # Data balancing arguments
    parser.add_argument('--balance_data', action='store_true', default=True,
                       help='Enable data balancing for training set')
    parser.add_argument('--balance_strategy', type=str, default='oversample',
                       choices=['oversample', 'undersample'],
                       help='Data balancing strategy')
    
    # CosineAnnealingWarmRestarts parameters
    parser.add_argument('--cosine_t0', type=int, default=None,
                       help='T_0 for CosineAnnealingWarmRestarts (default: 3 epochs)')
    parser.add_argument('--cosine_t_mult', type=int, default=2,
                       help='T_mult for CosineAnnealingWarmRestarts')
    parser.add_argument('--eta_min', type=float, default=1e-8,
                       help='Minimum learning rate for cosine scheduler')
    
    # Additional optimizations
    parser.add_argument('--optimizer_type', type=str, default='adamw',
                       choices=['adamw', 'adafactor'],
                       help='Optimizer type')
    parser.add_argument('--gradient_clipping', type=float, default=1.0,
                       help='Gradient clipping value (0.0 = disabled)')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Logging
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use wandb for logging')
    parser.add_argument('--wandb_project', type=str, default='phobert-vncorenlp',
                       help='Wandb project name')
    parser.add_argument('--run_name', type=str, default=None,
                       help='Run name for wandb')
    
    args = parser.parse_args()
    
    # Set run name if not provided
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = args.model_name.split('/')[-1]
        balance_suffix = "_balanced" if args.balance_data else ""
        args.run_name = f"{model_short}_vncorenlp{balance_suffix}_{timestamp}"
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine data subdirectory based on model (updated for VnCoreNLP data)
    if 'base' in args.model_name:
        data_subdir = 'phobert-base'
    else:
        data_subdir = 'phobert-large'
    
    data_path = os.path.join(args.data_dir, data_subdir)
    
    print("=" * 70)
    print("🚀 PHOBERT TRAINING VỚI VNCORENLP DATA")
    print("=" * 70)
    print(f"📂 Data directory: {data_path}")
    print(f"🤖 Model: {args.model_name}")
    print(f"💾 Output: {args.output_dir}")
    print(f"🔧 VnCoreNLP preprocessed data")
    print("=" * 70)
    
    # Kiểm tra data có tồn tại không
    train_path = os.path.join(data_path, 'train_processed.pkl')
    val_path = os.path.join(data_path, 'val_processed.pkl')
    test_path = os.path.join(data_path, 'test_processed.pkl')
    
    if not os.path.exists(train_path):
        print(f"❌ Không tìm thấy training data: {train_path}")
        print("💡 Hãy chạy script apply_vncorenlp_to_phobert.py trước")
        return
    
    # Load datasets với VnCoreNLP data
    print("📂 Loading datasets với VnCoreNLP preprocessed data...")
    train_dataset = VnCoreNLPClickbaitDataset(
        train_path,
        balance_data=args.balance_data,
        balance_strategy=args.balance_strategy
    )
    val_dataset = VnCoreNLPClickbaitDataset(val_path)
    
    test_dataset = None
    if os.path.exists(test_path):
        test_dataset = VnCoreNLPClickbaitDataset(test_path)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    # Initialize model
    model = PhoBERTClassifier(
        model_name=args.model_name,
        num_classes=args.num_classes,
        dropout_rate=args.dropout_rate,
        num_freeze_layers=args.num_freeze_layers,
        pooling_strategy=args.pooling_strategy
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 STATS: Total parameters: {total_params:,}")
    print(f"📊 STATS: Trainable parameters: {trainable_params:,}")
    print(f"📊 STATS: Training samples: {len(train_dataset)}")
    print(f"📊 STATS: Validation samples: {len(val_dataset)}")
    if test_dataset:
        print(f"📊 STATS: Test samples: {len(test_dataset)}")
    
    # Save config
    config = vars(args)
    config.update({
        'total_params': total_params,
        'trainable_params': trainable_params,
        'data_source': 'VnCoreNLP_preprocessed',
        'vncorenlp_version': '1.2',
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'test_samples': len(test_dataset) if test_dataset else 0
    })
    
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize trainer
    trainer = Trainer(args, model, train_loader, val_loader, test_loader)
    
    # Override trainer's data path để tránh lỗi path
    if hasattr(trainer, 'args'):
        trainer.args.data_dir = args.data_dir
    
    # Start training
    print("\n🚀 Bắt đầu training...")
    trainer.train()
    
    # Lưu kết quả test vào file
    if test_loader:
        print("\n📊 Đang lưu kết quả test...")
        save_test_results(trainer, test_loader, args.output_dir, config)
    
    print("\n" + "=" * 70)
    print("✅ HOÀN THÀNH TRAINING!")
    print(f"📁 Checkpoints được lưu tại: {args.output_dir}")
    print(f"🎯 Model đã được train với dữ liệu VnCoreNLP")
    if test_loader:
        print(f"📋 Kết quả test được lưu tại: {args.output_dir}/test_results.json")
    print("=" * 70)

if __name__ == "__main__":
    main() 