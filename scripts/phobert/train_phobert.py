#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

# Optional: Wandb for logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WARNING: Wandb không có sẵn - sẽ không log metrics online")

class FocalLoss(nn.Module):
    """Focal Loss để xử lý class imbalance"""
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross entropy loss"""
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        
    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
                 self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss

class WeightedFocalLoss(nn.Module):
    """Weighted Focal Loss với class weights"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class F1Loss(nn.Module):
    """F1 Loss - directly optimize F1 score"""
    def __init__(self, epsilon=1e-7):
        super(F1Loss, self).__init__()
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true):
        # Convert to probabilities
        y_pred = F.softmax(y_pred, dim=1)
        
        # Convert labels to one-hot
        y_true_onehot = F.one_hot(y_true, num_classes=y_pred.size(1)).float()
        
        # Calculate TP, FP, FN for each class
        tp = (y_pred * y_true_onehot).sum(dim=0)
        fp = (y_pred * (1 - y_true_onehot)).sum(dim=0)
        fn = ((1 - y_pred) * y_true_onehot).sum(dim=0)
        
        # Calculate precision and recall
        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)
        
        # Calculate F1 score
        f1 = 2 * precision * recall / (precision + recall + self.epsilon)
        
        # Return negative F1 (since we want to minimize loss)
        return 1 - f1.mean()

class ClickbaitDataset(Dataset):
    """Dataset cho PhoBERT clickbait classification với data balancing"""
    
    def __init__(self, data_path, balance_data=False, balance_strategy='oversample'):
        """
        Load preprocessed data từ pickle file với tùy chọn data balancing
        
        Args:
            data_path: Path to pickle file
            balance_data: Whether to balance the dataset
            balance_strategy: 'oversample', 'undersample', or 'smote'
        """
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        self.input_ids = self.data['input_ids']
        self.attention_mask = self.data['attention_mask'] 
        
        # Ensure labels are integers from the start
        raw_labels = self.data['labels']
        if isinstance(raw_labels, torch.Tensor):
            self.labels = raw_labels.tolist()
        else:
            self.labels = list(raw_labels)
        
        # Convert to integers if they aren't already
        self.labels = [int(label) for label in self.labels]
        
        original_size = len(self.input_ids)
        print(f"STATS: Original dataset: {original_size} samples from {data_path}")
        
        # Print class distribution
        label_counts = Counter(self.labels)
        print(f"STATS: Class distribution:")
        
        total_samples = len(self.labels)
        for label in [0, 1]:  # Ensure consistent order
            count = label_counts.get(label, 0)
            percentage = (count / total_samples) * 100 if total_samples > 0 else 0
            class_name = 'clickbait' if label == 0 else 'non-clickbait'  # Fix: 0=clickbait, 1=non-clickbait
            print(f"   {class_name}: {count} ({percentage:.1f}%)")
        
        # Apply data balancing if requested
        if balance_data and 'train' in data_path:
            print(f"PROCESS: Applying data balancing strategy: {balance_strategy}")
            self._balance_dataset(balance_strategy)
            
            # Print new distribution
            balanced_label_counts = Counter(self.labels)
            total_balanced = len(self.labels)
            print(f"STATS: Balanced class distribution:")
            for label in [0, 1]:  # Ensure consistent order
                count = balanced_label_counts.get(label, 0)
                percentage = (count / total_balanced) * 100 if total_balanced > 0 else 0
                class_name = 'clickbait' if label == 0 else 'non-clickbait'  # Fix: 0=clickbait, 1=non-clickbait
                print(f"   {class_name}: {count} ({percentage:.1f}%)")
            
            print(f"METRICS: Dataset size increased from {original_size} to {len(self.labels)} samples")
    
    def _balance_dataset(self, strategy='oversample'):
        """Balance dataset using specified strategy"""
        # Convert to lists for easier manipulation
        input_ids_list = [tensor.clone() for tensor in self.input_ids]
        attention_mask_list = [tensor.clone() for tensor in self.attention_mask]
        labels_list = list(self.labels)  # Already integers
        
        # Separate by class
        clickbait_indices = [i for i, label in enumerate(labels_list) if label == 0]  # Fix: 0=clickbait
        non_clickbait_indices = [i for i, label in enumerate(labels_list) if label == 1]  # Fix: 1=non-clickbait
        
        n_clickbait = len(clickbait_indices)
        n_non_clickbait = len(non_clickbait_indices)
        
        print(f"STATS: Before balancing - Clickbait: {n_clickbait}, Non-clickbait: {n_non_clickbait}")
        
        if strategy == 'oversample':
            # Oversample minority class
            if n_clickbait < n_non_clickbait:
                # Oversample clickbait
                target_size = n_non_clickbait
                minority_indices = clickbait_indices
                print(f"PROCESS: Oversampling clickbait from {n_clickbait} to {target_size}")
            else:
                # Oversample non-clickbait
                target_size = n_clickbait
                minority_indices = non_clickbait_indices
                print(f"PROCESS: Oversampling non-clickbait from {n_non_clickbait} to {target_size}")
            
            # Calculate how many samples to add
            samples_to_add = target_size - len(minority_indices)
            
            # Randomly sample with replacement from minority class
            additional_indices = np.random.choice(minority_indices, size=samples_to_add, replace=True)
            
            # Add the additional samples
            for idx in additional_indices:
                input_ids_list.append(input_ids_list[idx].clone())
                attention_mask_list.append(attention_mask_list[idx].clone())
                labels_list.append(labels_list[idx])
        
        elif strategy == 'undersample':
            # Undersample majority class
            if n_clickbait > n_non_clickbait:
                # Undersample clickbait
                target_size = n_non_clickbait
                majority_indices = clickbait_indices
                minority_indices = non_clickbait_indices
            else:
                # Undersample non-clickbait
                target_size = n_clickbait
                majority_indices = non_clickbait_indices
                minority_indices = clickbait_indices
            
            # Randomly sample from majority class
            selected_majority = np.random.choice(majority_indices, size=target_size, replace=False)
            
            # Keep all minority + selected majority
            selected_indices = list(minority_indices) + list(selected_majority)
            
            # Update lists
            input_ids_list = [input_ids_list[i] for i in selected_indices]
            attention_mask_list = [attention_mask_list[i] for i in selected_indices]
            labels_list = [labels_list[i] for i in selected_indices]
        
        # Update instance variables
        self.input_ids = input_ids_list
        self.attention_mask = attention_mask_list
        self.labels = labels_list
        
        # Shuffle the dataset
        combined = list(zip(self.input_ids, self.attention_mask, self.labels))
        random.shuffle(combined)
        self.input_ids, self.attention_mask, self.labels = zip(*combined)
        
        # Convert to lists again for easier access
        self.input_ids = list(self.input_ids)
        self.attention_mask = list(self.attention_mask)
        self.labels = list(self.labels)
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

class PhoBERTClassifier(nn.Module):
    """PhoBERT model cho clickbait classification"""
    
    def __init__(self, model_name, num_classes=2, dropout_rate=0.1, num_freeze_layers=0, pooling_strategy='cls_mean'):
        super(PhoBERTClassifier, self).__init__()
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.phobert = AutoModel.from_pretrained(model_name)
        self.pooling_strategy = pooling_strategy
        
        # Freeze layers nếu cần
        if num_freeze_layers > 0:
            self.freeze_layers(num_freeze_layers)
        
        # Tính input size cho classifier dựa trên pooling strategy
        if pooling_strategy == 'cls_mean':
            classifier_input_size = self.config.hidden_size * 2  # CLS + Mean pooling
        elif pooling_strategy == 'attention':
            classifier_input_size = self.config.hidden_size
            # Attention weights cho attention pooling
            self.attention_weights = nn.Linear(self.config.hidden_size, 1)
        else:  # 'cls'
            classifier_input_size = self.config.hidden_size
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # Multi-layer classifier với batch normalization
        self.hidden1 = nn.Linear(classifier_input_size, self.config.hidden_size)
        self.bn1 = nn.BatchNorm1d(self.config.hidden_size)
        self.hidden2 = nn.Linear(self.config.hidden_size, self.config.hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(self.config.hidden_size // 2)
        self.classifier = nn.Linear(self.config.hidden_size // 2, num_classes)
        
        # Initialize weights
        self._init_weights()
        
        print(f"PACKAGE: Model initialized: {model_name}")
        print(f"CONFIG: Hidden size: {self.config.hidden_size}")
        print(f"TARGET: Num classes: {num_classes}")
        print(f"FREEZE: Frozen layers: {num_freeze_layers}")
        print(f"TARGET: Pooling strategy: {pooling_strategy}")
    
    def _init_weights(self):
        """Initialize classifier weights"""
        for module in [self.hidden1, self.hidden2, self.classifier]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def freeze_layers(self, num_freeze_layers):
        """Freeze early transformer layers"""
        for param in self.phobert.embeddings.parameters():
            param.requires_grad = False
            
        for i in range(num_freeze_layers):
            for param in self.phobert.encoder.layer[i].parameters():
                param.requires_grad = False
        
        print(f"FREEZE: Frozen {num_freeze_layers} layers")
    
    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Different pooling strategies
        if self.pooling_strategy == 'cls':
            # Standard CLS token
            pooled_output = outputs.last_hidden_state[:, 0]
        
        elif self.pooling_strategy == 'cls_mean':
            # Concatenate CLS token and mean pooling
            cls_output = outputs.last_hidden_state[:, 0]
            
            # Mean pooling with attention mask
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_output = sum_embeddings / sum_mask
            
            pooled_output = torch.cat([cls_output, mean_output], dim=1)
        
        elif self.pooling_strategy == 'attention':
            # Attention-based pooling
            token_embeddings = outputs.last_hidden_state
            
            # Compute attention weights
            attention_weights = self.attention_weights(token_embeddings)
            attention_weights = F.softmax(attention_weights, dim=1)
            
            # Apply attention mask
            attention_mask_expanded = attention_mask.unsqueeze(-1)
            attention_weights = attention_weights * attention_mask_expanded
            
            # Weighted sum
            pooled_output = torch.sum(attention_weights * token_embeddings, dim=1)
        
        pooled_output = self.dropout(pooled_output)
        
        # Multi-layer classifier với residual connections
        hidden1 = F.relu(self.bn1(self.hidden1(pooled_output)))
        hidden1 = self.dropout(hidden1)
        
        hidden2 = F.relu(self.bn2(self.hidden2(hidden1)))
        hidden2 = self.dropout(hidden2)
        
        # Residual connection nếu dimensions match
        if hidden1.size(1) == hidden2.size(1):
            hidden2 = hidden2 + hidden1
        
        logits = self.classifier(hidden2)
        
        return logits

class Trainer:
    """Trainer class cho PhoBERT"""
    
    def __init__(self, args, model, train_loader, val_loader, test_loader=None):
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"CONFIG: Device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            print(f"SAVE: GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            print(f"🔥 Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
        
        self.model.to(self.device)
        
        # Gradient accumulation
        self.gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
        print(f"PROCESS: Gradient accumulation steps: {self.gradient_accumulation_steps}")
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            eps=1e-8
        )
        
        # Scheduler
        total_steps = len(train_loader) * args.epochs
        warmup_steps = int(total_steps * args.warmup_ratio)
        
        if args.scheduler == 'linear':
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        elif args.scheduler == 'cosine':
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        elif args.scheduler == 'cosine_restarts':
            # CosineAnnealingWarmRestarts with periodic restarts
            t0 = args.cosine_t0 if hasattr(args, 'cosine_t0') and args.cosine_t0 is not None else len(train_loader) * 3
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=t0,  # Restart every 3 epochs by default
                T_mult=args.cosine_t_mult if hasattr(args, 'cosine_t_mult') else 2,
                eta_min=args.eta_min if hasattr(args, 'eta_min') else 1e-8
            )
            print(f"PROCESS: Using CosineAnnealingWarmRestarts scheduler with T_0={t0}")
        
        # Stochastic Weight Averaging (SWA)
        self.use_swa = hasattr(args, 'swa_start') and args.swa_start > 0
        if self.use_swa:
            from torch.optim.swa_utils import AveragedModel, SWALR
            self.swa_model = AveragedModel(self.model)
            self.swa_scheduler = SWALR(self.optimizer, swa_lr=args.swa_lr if hasattr(args, 'swa_lr') else 1e-5)
            print(f"PROCESS: SWA will start at epoch {args.swa_start}")
        
        # Progressive unfreezing setup
        self.progressive_unfreezing = args.progressive_unfreezing
        if self.progressive_unfreezing:
            self.total_transformer_layers = 12 if 'base' in args.model_name else 24
            self.unfreeze_schedule = self._create_unfreeze_schedule()
            print(f"🔓 Progressive unfreezing enabled for {self.total_transformer_layers} layers")
        
        # Mixed precision
        self.use_amp = args.use_amp and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = GradScaler()
            print("⚡ Using mixed precision training")
        
        # Loss function selection với class weights
        loss_type = getattr(args, 'loss_type', 'cross_entropy')
        
        # Calculate class weights from training data
        self.class_weights = self.calculate_class_weights()
        
        if loss_type == 'focal':
            self.criterion = FocalLoss(alpha=1.0, gamma=2.0)
            print("TARGET: Using Focal Loss")
        elif loss_type == 'weighted_focal':
            self.criterion = WeightedFocalLoss(alpha=self.class_weights, gamma=2.0)
            print("TARGET: Using Weighted Focal Loss with class weights")
        elif loss_type == 'f1':
            self.criterion = F1Loss()
            print("TARGET: Using F1 Loss - directly optimizing F1 score")
        elif loss_type == 'label_smoothing':
            smoothing = getattr(args, 'label_smoothing', 0.1)
            self.criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
            print(f"TARGET: Using Label Smoothing Loss (smoothing={smoothing})")
        else:
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
            print("TARGET: Using Weighted Cross Entropy Loss")
        
        print(f"STATS: Class weights: {self.class_weights}")
        
        # Early stopping
        self.best_val_f1 = 0
        self.patience_counter = 0
        
        # Exponential Moving Average
        self.use_ema = getattr(args, 'use_ema', True)
        if self.use_ema:
            self.ema_model = self.create_ema_model()
            self.ema_decay = getattr(args, 'ema_decay', 0.9999)
            print(f"METRICS: Using EMA with decay: {self.ema_decay}")
        
        # Learning rate scheduling với restart
        self.restart_epochs = getattr(args, 'restart_epochs', [])
        self.current_restart = 0
    
    def calculate_class_weights(self):
        """Calculate class weights based on training data distribution"""
        # Count samples per class
        class_counts = torch.zeros(2)  # Assuming 2 classes
        total_samples = 0
        
        for batch in self.train_loader:
            labels = batch['labels']
            for label in labels:
                class_counts[label.item()] += 1
                total_samples += 1
        
        # Calculate inverse frequency weights
        class_weights = total_samples / (2.0 * class_counts)
        
        # Normalize weights
        class_weights = class_weights / class_weights.sum() * 2.0
        
        return class_weights.to(self.device)
    
    def create_ema_model(self):
        """Create EMA model"""
        ema_model = type(self.model)(
            self.args.model_name,
            self.args.num_classes,
            self.args.dropout_rate,
            self.args.num_freeze_layers,
            getattr(self.args, 'pooling_strategy', 'cls_mean')
        )
        ema_model.load_state_dict(self.model.state_dict())
        ema_model.to(self.device)  # Move EMA model to same device as main model
        for param in ema_model.parameters():
            param.requires_grad = False
        return ema_model
    
    def update_ema(self):
        """Update EMA model"""
        if not self.use_ema:
            return
        
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(model_param.data, alpha=1 - self.ema_decay)
    
    def _create_unfreeze_schedule(self):
        """Create a list of layers to unfreeze based on progressive unfreezing"""
        unfreeze_schedule = []
        if self.progressive_unfreezing:
            # Determine the number of layers to unfreeze
            num_layers_to_unfreeze = int(self.total_transformer_layers * self.args.layer_lr_decay)
            
            # Ensure we don't unfreeze more layers than available
            num_layers_to_unfreeze = min(num_layers_to_unfreeze, self.total_transformer_layers)
            
            # Create a list of layer indices to unfreeze
            unfreeze_schedule = [i for i in range(num_layers_to_unfreeze)]
            
            # Add a placeholder for the last layer if it's not in the schedule
            if self.total_transformer_layers - 1 not in unfreeze_schedule:
                unfreeze_schedule.append(self.total_transformer_layers - 1)
            
            # Ensure the schedule is sorted and unique
            unfreeze_schedule = sorted(list(set(unfreeze_schedule)))
            
            print(f"🔓 Unfreezing schedule: {unfreeze_schedule}")
        return unfreeze_schedule
    
    def train_epoch(self):
        """Train một epoch với gradient accumulation"""
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(pbar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            if self.use_amp:
                with autocast():
                    logits = self.model(input_ids, attention_mask)
                    loss = self.criterion(logits, labels)
                    # Scale loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    # Update EMA model
                    self.update_ema()

                    # Stochastic Weight Averaging (SWA)
                    if self.use_swa and self.swa_scheduler.epoch > 0:
                        self.swa_model.update_parameters(self.model)
                        self.swa_scheduler.step()

                    # Progressive unfreezing
                    if self.progressive_unfreezing and self.unfreeze_schedule:
                        current_epoch = self.swa_scheduler.epoch + 1 # Use SWA epoch for unfreezing
                        if current_epoch in self.unfreeze_schedule:
                            for name, param in self.model.named_parameters():
                                if name in self.phobert.state_dict() and param.requires_grad:
                                    param.requires_grad = True
                                    print(f"🔓 Unfreezing layer: {name}")
                                elif name.startswith('phobert.encoder.layer.'):
                                    layer_idx = int(name.split('.')[3])
                                    if layer_idx in self.unfreeze_schedule:
                                        for sub_name, sub_param in self.phobert.encoder.layer[layer_idx].named_parameters():
                                            if sub_name in self.phobert.state_dict() and sub_param.requires_grad:
                                                sub_param.requires_grad = True
                                                print(f"🔓 Unfreezing layer: {name}.{sub_name}")
                        # Remove the unfreezed layer from the schedule after it's processed
                        self.unfreeze_schedule = [e for e in self.unfreeze_schedule if e != current_epoch]
                        if not self.unfreeze_schedule:
                            print("🔓 All transformer layers unfreezed.")
            else:
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
                
                loss.backward()
                
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    # Update EMA model
                    self.update_ema()

                    # Stochastic Weight Averaging (SWA)
                    if self.use_swa and self.swa_scheduler.epoch > 0:
                        self.swa_model.update_parameters(self.model)
                        self.swa_scheduler.step()

                    # Progressive unfreezing
                    if self.progressive_unfreezing and self.unfreeze_schedule:
                        current_epoch = self.swa_scheduler.epoch + 1 # Use SWA epoch for unfreezing
                        if current_epoch in self.unfreeze_schedule:
                            for name, param in self.model.named_parameters():
                                if name in self.phobert.state_dict() and param.requires_grad:
                                    param.requires_grad = True
                                    print(f"🔓 Unfreezing layer: {name}")
                                elif name.startswith('phobert.encoder.layer.'):
                                    layer_idx = int(name.split('.')[3])
                                    if layer_idx in self.unfreeze_schedule:
                                        for sub_name, sub_param in self.phobert.encoder.layer[layer_idx].named_parameters():
                                            if sub_name in self.phobert.state_dict() and sub_param.requires_grad:
                                                sub_param.requires_grad = True
                                                print(f"🔓 Unfreezing layer: {name}.{sub_name}")
                        # Remove the unfreezed layer from the schedule after it's processed
                        self.unfreeze_schedule = [e for e in self.unfreeze_schedule if e != current_epoch]
                        if not self.unfreeze_schedule:
                            print("🔓 All transformer layers unfreezed.")
            
            # Statistics (use original loss for logging)
            total_loss += loss.item() * self.gradient_accumulation_steps
            _, predicted = torch.max(logits.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
            
            # Update progress bar
            current_acc = correct_predictions / total_predictions
            pbar.set_postfix({
                'loss': f'{loss.item() * self.gradient_accumulation_steps:.4f}',
                'acc': f'{current_acc:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def evaluate(self, data_loader, split_name="Validation"):
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Evaluating {split_name}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                if self.use_amp:
                    with autocast():
                        logits = self.model(input_ids, attention_mask)
                        loss = self.criterion(logits, labels)
                else:
                    logits = self.model(input_ids, attention_mask)
                    loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                
                _, predicted = torch.max(logits.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': all_predictions,
            'labels': all_labels
        }
    
    def evaluate_ema(self, data_loader, split_name="EMA Validation"):
        """Evaluate EMA model"""
        if not self.use_ema:
            return None
        
        self.ema_model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Evaluating {split_name}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                if self.use_amp:
                    with autocast():
                        logits = self.ema_model(input_ids, attention_mask)
                        loss = self.criterion(logits, labels)
                else:
                    logits = self.ema_model(input_ids, attention_mask)
                    loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                
                _, predicted = torch.max(logits.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': all_predictions,
            'labels': all_labels
        }
    
    def save_checkpoint(self, epoch, metrics, is_best=False, ema_metrics=None):
        """Save model checkpoint including EMA model"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'args': vars(self.args)
        }
        
        # Add EMA model if available
        if self.use_ema:
            checkpoint['ema_model_state_dict'] = self.ema_model.state_dict()
            if ema_metrics:
                checkpoint['ema_metrics'] = ema_metrics
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.args.output_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model (choose between regular and EMA based on F1)
        if is_best:
            best_path = os.path.join(self.args.output_dir, 'best_model.pth')
            
            # Determine which model performed better
            if ema_metrics and ema_metrics['f1'] > metrics['f1']:
                checkpoint['best_model_type'] = 'ema'
                best_f1 = ema_metrics['f1']
                print(f"SAVE: Saved best EMA model with F1: {best_f1:.4f}")
            else:
                checkpoint['best_model_type'] = 'regular'
                best_f1 = metrics['f1']
                print(f"SAVE: Saved best regular model with F1: {best_f1:.4f}")
            
            torch.save(checkpoint, best_path)
    
    def train(self):
        """Main training loop"""
        print("START: Starting training...")
        
        for epoch in range(self.args.epochs):
            print(f"\n📅 Epoch {epoch + 1}/{self.args.epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate with both regular model and EMA model
            val_metrics = self.evaluate(self.val_loader, "Validation")
            
            # Evaluate EMA model if available
            ema_metrics = None
            if self.use_ema:
                ema_metrics = self.evaluate_ema(self.val_loader, "Validation EMA")
            
            # Print metrics
            print(f"STATS: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"STATS: Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"STATS: Val F1: {val_metrics['f1']:.4f}, Val Precision: {val_metrics['precision']:.4f}, Val Recall: {val_metrics['recall']:.4f}")
            
            if ema_metrics:
                print(f"STATS: EMA Val F1: {ema_metrics['f1']:.4f}, EMA Val Acc: {ema_metrics['accuracy']:.4f}")
            
            # Log to wandb
            if WANDB_AVAILABLE and self.args.use_wandb:
                log_dict = {
                    'epoch': epoch + 1,
                    'train/loss': train_loss,
                    'train/accuracy': train_acc,
                    'val/loss': val_metrics['loss'],
                    'val/accuracy': val_metrics['accuracy'],
                    'val/f1': val_metrics['f1'],
                    'val/precision': val_metrics['precision'],
                    'val/recall': val_metrics['recall'],
                    'learning_rate': self.scheduler.get_last_lr()[0]
                }
                
                if ema_metrics:
                    log_dict.update({
                        'val_ema/accuracy': ema_metrics['accuracy'],
                        'val_ema/f1': ema_metrics['f1'],
                        'val_ema/precision': ema_metrics['precision'],
                        'val_ema/recall': ema_metrics['recall']
                    })
                
                wandb.log(log_dict)
            
            # Save checkpoint - prioritize F1 score over accuracy
            current_f1 = ema_metrics['f1'] if ema_metrics and ema_metrics['f1'] > val_metrics['f1'] else val_metrics['f1']
            is_best = current_f1 > self.best_val_f1
            
            if is_best:
                self.best_val_f1 = current_f1
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            self.save_checkpoint(epoch, val_metrics, is_best, ema_metrics)
            
            # Early stopping
            if self.patience_counter >= self.args.patience:
                print(f"⏹️ Early stopping at epoch {epoch + 1}")
                break
        
        # Test evaluation
        if self.test_loader:
            print("\nTEST: Testing on test set...")
            test_metrics = self.evaluate(self.test_loader, "Test")
            print(f"STATS: Test Acc: {test_metrics['accuracy']:.4f}")
            print(f"STATS: Test F1: {test_metrics['f1']:.4f}")
            
            # Classification report - use the same data path as the model
            if 'base' in self.args.model_name:
                data_subdir = 'phobert-base-v2'
            else:
                data_subdir = 'phobert-large-v2'
            data_path = os.path.join(self.args.data_dir, data_subdir, 'train_processed.pkl')
            with open(data_path, 'rb') as f:
                sample_data = pickle.load(f)
            label_mapping = sample_data['label_mapping']
            label_names = list(label_mapping.keys())
            
            report = classification_report(
                test_metrics['labels'], 
                test_metrics['predictions'],
                target_names=label_names
            )
            print("\nREPORT: Classification Report:")
            print(report)
            
            # Save report
            with open(os.path.join(self.args.output_dir, 'test_report.txt'), 'w') as f:
                f.write(report)
        
        print("SUCCESS: Training completed!")

def set_seed(seed):
    """Set random seed cho reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="PhoBERT Training for Clickbait Detection")
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='vinai/phobert-base',
                       choices=['vinai/phobert-base', 'vinai/phobert-large'],
                       help='PhoBERT model name')
    parser.add_argument('--data_dir', type=str, default='data-bert-v2',
                       help='Directory containing preprocessed data')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
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
    parser.add_argument('--wandb_project', type=str, default='phobert-clickbait',
                       help='Wandb project name')
    parser.add_argument('--run_name', type=str, default=None,
                       help='Run name for wandb')
    
    args = parser.parse_args()
    
    # Set run name if not provided
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = args.model_name.split('/')[-1]
        balance_suffix = "_balanced" if args.balance_data else ""
        args.run_name = f"{model_short}{balance_suffix}_{timestamp}"
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine data subdirectory based on model
    if 'base' in args.model_name:
        data_subdir = 'phobert-base-v2'
    else:
        data_subdir = 'phobert-large-v2'
    
    data_path = os.path.join(args.data_dir, data_subdir)
    
    # Load datasets với data balancing
    print("LOAD: Loading datasets...")
    train_dataset = ClickbaitDataset(
        os.path.join(data_path, 'train_processed.pkl'),
        balance_data=args.balance_data,
        balance_strategy=args.balance_strategy
    )
    val_dataset = ClickbaitDataset(os.path.join(data_path, 'val_processed.pkl'))
    
    test_dataset = None
    test_path = os.path.join(data_path, 'test_processed.pkl')
    if os.path.exists(test_path):
        test_dataset = ClickbaitDataset(test_path)
    
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
    print(f"STATS: Total parameters: {total_params:,}")
    print(f"STATS: Trainable parameters: {trainable_params:,}")
    
    # Save config
    config = vars(args)
    config['total_params'] = total_params
    config['trainable_params'] = trainable_params
    
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize trainer
    trainer = Trainer(args, model, train_loader, val_loader, test_loader)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main() 