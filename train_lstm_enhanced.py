#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced LSTM với pre-trained embeddings cho clickbait detection
Cải thiện từ version cũ với Word2Vec/FastText và regularization tốt hơn
"""

import os
import re
import json
import pickle
import numpy as np
import pandas as pd
from collections import Counter
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import clip_grad_norm_

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Vietnamese processing
try:
    from pyvi import ViTokenizer
    PYVI_AVAILABLE = True
    print("✅ pyvi available")
except ImportError:
    PYVI_AVAILABLE = False
    print("⚠️ pyvi not available - using basic tokenization")

# Pre-trained embeddings
try:
    import gensim
    from gensim.models import Word2Vec, FastText
    GENSIM_AVAILABLE = True
    print("✅ gensim available for pre-trained embeddings")
except ImportError:
    GENSIM_AVAILABLE = False
    print("⚠️ gensim not available - using random embeddings")

try:
    import fasttext
    FASTTEXT_AVAILABLE = True
    print("✅ fasttext available")
except ImportError:
    FASTTEXT_AVAILABLE = False
    print("⚠️ fasttext not available")

class EnhancedVietnameseTextProcessor:
    """Enhanced text processor với hỗ trợ pre-trained embeddings"""
    
    def __init__(self, max_vocab_size=10000, min_word_freq=2, max_length=100,
                 embedding_type='random', embedding_path=None, embedding_dim=300):
        self.max_vocab_size = max_vocab_size
        self.min_word_freq = min_word_freq
        self.max_length = max_length
        self.embedding_type = embedding_type  # 'random', 'word2vec', 'fasttext'
        self.embedding_path = embedding_path
        self.embedding_dim = embedding_dim
        
        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.START_TOKEN = '<START>'
        self.END_TOKEN = '<END>'
        
        # Vocabulary
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
        
        # Pre-trained embeddings
        self.pretrained_model = None
        self.embedding_matrix = None
        
        print(f"🔧 Enhanced Vietnamese Text Processor")
        print(f"   Max vocab: {max_vocab_size}")
        print(f"   Min freq: {min_word_freq}")
        print(f"   Max length: {max_length}")
        print(f"   Embedding type: {embedding_type}")
        print(f"   Embedding dim: {embedding_dim}")
    
    def load_pretrained_embeddings(self):
        """Load pre-trained embeddings"""
        if self.embedding_type == 'random':
            print("🎲 Using random embeddings")
            return
            
        if not self.embedding_path or not os.path.exists(self.embedding_path):
            print(f"⚠️ Embedding path not found: {self.embedding_path}")
            print("🎲 Falling back to random embeddings")
            self.embedding_type = 'random'
            return
        
        try:
            if self.embedding_type == 'word2vec':
                print(f"📥 Loading Word2Vec from {self.embedding_path}")
                self.pretrained_model = gensim.models.KeyedVectors.load_word2vec_format(
                    self.embedding_path, binary=True
                )
                self.embedding_dim = self.pretrained_model.vector_size
                
            elif self.embedding_type == 'fasttext':
                print(f"📥 Loading FastText from {self.embedding_path}")
                if FASTTEXT_AVAILABLE:
                    self.pretrained_model = fasttext.load_model(self.embedding_path)
                    self.embedding_dim = self.pretrained_model.get_dimension()
                else:
                    # Try with gensim
                    self.pretrained_model = gensim.models.FastText.load(self.embedding_path)
                    self.embedding_dim = self.pretrained_model.vector_size
                    
            print(f"✅ Loaded embeddings: dimension = {self.embedding_dim}")
            
        except Exception as e:
            print(f"❌ Error loading embeddings: {e}")
            print("🎲 Falling back to random embeddings")
            self.embedding_type = 'random'
            self.pretrained_model = None
    
    def clean_text(self, text):
        """Làm sạch text cơ bản"""
        text = text.lower()
        text = re.sub(r'[^\w\s\u00C0-\u024F\u1E00-\u1EFF]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def word_segment(self, text):
        """Word segmentation với pyvi"""
        if not PYVI_AVAILABLE:
            return text.split()
        
        try:
            segmented = ViTokenizer.tokenize(text)
            return segmented.split()
        except Exception as e:
            return text.split()
    
    def preprocess_text(self, text):
        """Pipeline preprocessing hoàn chỉnh"""
        text = self.clean_text(text)
        words = self.word_segment(text)
        words = [word for word in words if len(word) > 1]
        return words
    
    def get_word_vector(self, word):
        """Lấy vector từ pre-trained model"""
        if self.pretrained_model is None:
            return None
            
        try:
            if self.embedding_type == 'word2vec':
                if word in self.pretrained_model:
                    return self.pretrained_model[word]
            elif self.embedding_type == 'fasttext':
                if hasattr(self.pretrained_model, 'get_word_vector'):
                    return self.pretrained_model.get_word_vector(word)
                elif hasattr(self.pretrained_model, 'wv') and word in self.pretrained_model.wv:
                    return self.pretrained_model.wv[word]
        except:
            pass
        
        return None
    
    def build_vocabulary(self, texts):
        """Xây dựng vocabulary và embedding matrix"""
        print("🔄 Building vocabulary...")
        
        # Load pre-trained embeddings first
        self.load_pretrained_embeddings()
        
        # Count words
        word_counts = Counter()
        for text in tqdm(texts, desc="Processing texts"):
            words = self.preprocess_text(text)
            word_counts.update(words)
        
        # Filter by frequency
        filtered_words = [word for word, count in word_counts.items() 
                         if count >= self.min_word_freq]
        
        # Sort by frequency
        filtered_words = sorted(filtered_words, 
                               key=lambda x: word_counts[x], reverse=True)
        
        # Limit vocabulary size
        if len(filtered_words) > self.max_vocab_size - 4:
            filtered_words = filtered_words[:self.max_vocab_size - 4]
        
        # Build word2idx mapping
        self.word2idx = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.START_TOKEN: 2,
            self.END_TOKEN: 3
        }
        
        for i, word in enumerate(filtered_words):
            self.word2idx[word] = i + 4
        
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        
        # Build embedding matrix
        self.build_embedding_matrix()
        
        print(f"✅ Vocabulary built: {self.vocab_size} words")
        print(f"   Embedding matrix shape: {self.embedding_matrix.shape}")
    
    def build_embedding_matrix(self):
        """Xây dựng embedding matrix từ pre-trained vectors"""
        print("🔄 Building embedding matrix...")
        
        # Initialize embedding matrix
        self.embedding_matrix = np.random.normal(
            0, 0.1, (self.vocab_size, self.embedding_dim)
        )
        
        # Set PAD token to zeros
        self.embedding_matrix[0] = np.zeros(self.embedding_dim)
        
        if self.embedding_type == 'random':
            print("🎲 Using random embedding matrix")
            return
        
        # Fill with pre-trained vectors
        found_count = 0
        for word, idx in self.word2idx.items():
            if word in [self.PAD_TOKEN, self.UNK_TOKEN, self.START_TOKEN, self.END_TOKEN]:
                continue
                
            vector = self.get_word_vector(word)
            if vector is not None:
                self.embedding_matrix[idx] = vector
                found_count += 1
        
        coverage = found_count / (self.vocab_size - 4) * 100
        print(f"✅ Embedding coverage: {coverage:.1f}% ({found_count}/{self.vocab_size-4})")
    
    def text_to_sequence(self, text):
        """Chuyển text thành sequence của indices"""
        words = self.preprocess_text(text)
        sequence = [self.word2idx.get(word, self.word2idx[self.UNK_TOKEN]) 
                   for word in words]
        sequence = [self.word2idx[self.START_TOKEN]] + sequence + [self.word2idx[self.END_TOKEN]]
        
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
        
        return sequence
    
    def texts_to_sequences(self, texts):
        """Chuyển danh sách texts thành sequences"""
        sequences = []
        for text in tqdm(texts, desc="Converting texts to sequences"):
            seq = self.text_to_sequence(text)
            sequences.append(seq)
        return sequences

class ClickbaitDataset(Dataset):
    """Dataset class cho LSTM training"""
    
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sequence, label

def collate_fn(batch):
    """Custom collate function để pad sequences"""
    sequences, labels = zip(*batch)
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return sequences_padded, labels

class EnhancedLSTM(nn.Module):
    """Enhanced LSTM với pre-trained embeddings và better regularization"""
    
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=256, 
                 num_layers=2, dropout=0.5, num_classes=2, 
                 embedding_matrix=None, freeze_embeddings=False):
        super(EnhancedLSTM, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Load pre-trained embeddings if available
        if embedding_matrix is not None:
            print("🔄 Loading pre-trained embedding weights...")
            self.embedding.weight.data.copy_(torch.FloatTensor(embedding_matrix))
            
            if freeze_embeddings:
                print("🧊 Freezing embedding layer")
                self.embedding.weight.requires_grad = False
        
        # LSTM layers với layer normalization
        self.lstm1 = nn.LSTM(
            embedding_dim, hidden_dim,
            num_layers=1, dropout=0, bidirectional=True,
            batch_first=True
        )
        
        self.lstm2 = nn.LSTM(
            hidden_dim * 2, hidden_dim,
            num_layers=1, dropout=0, bidirectional=True,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_dim * 2)
        self.layer_norm2 = nn.LayerNorm(hidden_dim * 2)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout * 0.5)  # Lighter dropout for final layer
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=8,
            dropout=dropout * 0.3,
            batch_first=True
        )
        
        # Classification head với multiple layers
        # Input dimension: hidden_dim * 2 (from LSTM) * 2 (mean + max pooling) = hidden_dim * 4
        classifier_input_dim = hidden_dim * 4
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
        print(f"📦 Enhanced LSTM Model:")
        print(f"   Vocab size: {vocab_size}")
        print(f"   Embedding dim: {embedding_dim}")
        print(f"   Hidden dim: {hidden_dim}")
        print(f"   Num layers: {num_layers}")
        print(f"   Dropout: {dropout}")
        print(f"   Pre-trained embeddings: {embedding_matrix is not None}")
        print(f"   Attention: MultiheadAttention")
    
    def _init_weights(self):
        """Initialize weights"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x, attention_mask=None):
        batch_size, seq_len = x.size()
        
        # Embedding
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        embedded = self.dropout1(embedded)
        
        # First LSTM layer
        lstm1_out, _ = self.lstm1(embedded)
        lstm1_out = self.layer_norm1(lstm1_out)
        lstm1_out = self.dropout2(lstm1_out)
        
        # Second LSTM layer
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = self.layer_norm2(lstm2_out)
        
        # Self-attention
        if attention_mask is not None:
            # Convert padding mask to attention mask
            attn_mask = (attention_mask == 0)
        else:
            attn_mask = None
            
        attn_out, _ = self.attention(
            lstm2_out, lstm2_out, lstm2_out,
            key_padding_mask=attn_mask
        )
        
        # Global max pooling và mean pooling
        if attention_mask is not None:
            # Mask cho pooling
            mask = attention_mask.unsqueeze(-1).float()
            attn_out = attn_out * mask
            
            # Mean pooling với mask
            lengths = attention_mask.sum(dim=1, keepdim=True).float()
            mean_pooled = attn_out.sum(dim=1) / lengths
            
            # Max pooling với mask  
            attn_out_masked = attn_out.masked_fill(attention_mask.unsqueeze(-1) == 0, -float('inf'))
            max_pooled, _ = attn_out_masked.max(dim=1)
        else:
            mean_pooled = attn_out.mean(dim=1)
            max_pooled, _ = attn_out.max(dim=1)
        
        # Combine pooled features
        combined = torch.cat([mean_pooled, max_pooled], dim=1)  # [batch_size, hidden_dim * 2 * 2]
        combined = self.dropout3(combined)
        
        # Classification
        output = self.classifier(combined)
        
        return output

class EnhancedLSTMTrainer:
    """Enhanced LSTM Trainer với advanced features"""
    
    def __init__(self, model, train_loader, val_loader, test_loader=None, 
                 learning_rate=0.001, device='cuda', scheduler_type='cosine'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Optimizer với weight decay
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Loss function với label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Learning rate scheduler
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=10, T_mult=2
            )
        elif scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=3
            )
        else:
            self.scheduler = None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        
        print(f"🚀 Enhanced LSTM Trainer initialized")
        print(f"   Device: {device}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Scheduler: {scheduler_type}")
    
    def create_attention_mask(self, sequences):
        """Tạo attention mask cho sequences"""
        return (sequences != 0).long()
    
    def train_epoch(self):
        """Train một epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (sequences, labels) in enumerate(self.train_loader):
            sequences, labels = sequences.to(self.device), labels.to(self.device)
            attention_mask = self.create_attention_mask(sequences)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(sequences, attention_mask)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update scheduler
            if self.scheduler and isinstance(self.scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts):
                self.scheduler.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 50 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'   Batch {batch_idx}/{len(self.train_loader)}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Acc: {100.*correct/total:.2f}%, '
                      f'LR: {current_lr:.6f}')
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, data_loader, split_name="Validation"):
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for sequences, labels in data_loader:
                sequences, labels = sequences.to(self.device), labels.to(self.device)
                attention_mask = self.create_attention_mask(sequences)
                
                outputs = self.model(sequences, attention_mask)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        print(f"\n📊 {split_name} Results:")
        print(f"   Loss: {avg_loss:.4f}")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        
        # Detailed metrics breakdown for test evaluation
        if split_name == "Test":
            from sklearn.metrics import classification_report
            report = classification_report(
                all_labels, all_predictions,
                target_names=['non-clickbait', 'clickbait'],
                output_dict=True
            )
            
            # Quick metrics summary
            clickbait_precision = report['clickbait']['precision']
            clickbait_f1 = report['clickbait']['f1-score']
            non_clickbait_precision = report['non-clickbait']['precision']
            non_clickbait_f1 = report['non-clickbait']['f1-score']
            macro_f1 = report['macro avg']['f1-score']
            weighted_f1 = report['weighted avg']['f1-score']
            
            print(f"   📋 Breakdown:")
            print(f"     Non-clickbait: Precision={non_clickbait_precision:.4f}, F1={non_clickbait_f1:.4f}")
            print(f"     Clickbait: Precision={clickbait_precision:.4f}, F1={clickbait_f1:.4f}")
            print(f"     Macro F1: {macro_f1:.4f}")
            print(f"     Weighted F1: {weighted_f1:.4f}")
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': all_predictions,
            'labels': all_labels
        }
    
    def train(self, num_epochs, save_dir="models_lstm_enhanced"):
        """Training loop chính với early stopping"""
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_f1 = 0
        patience = 7  # Increased patience
        patience_counter = 0
        
        print(f"\n🚀 Starting enhanced training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            print(f"\n📅 Epoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            print(f"\n🔄 Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
            
            # Validate
            val_results = self.evaluate(self.val_loader, "Validation")
            self.val_losses.append(val_results['loss'])
            self.val_accuracies.append(val_results['accuracy'] * 100)
            
            # Update scheduler
            if self.scheduler and isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_results['loss'])
            
            # Early stopping check
            if val_results['f1'] > best_val_f1:
                best_val_f1 = val_results['f1']
                patience_counter = 0
                
                # Save best model
                torch.save(self.model.state_dict(), 
                          os.path.join(save_dir, 'best_model_enhanced.pth'))
                print(f"💾 Saved best model! F1: {best_val_f1:.4f}")
                
            else:
                patience_counter += 1
                print(f"⏳ Patience: {patience_counter}/{patience}")
                
                if patience_counter >= patience:
                    print(f"🛑 Early stopping triggered!")
                    break
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates,
            'best_val_f1': best_val_f1
        }
        
        with open(os.path.join(save_dir, 'training_history_enhanced.json'), 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\n✅ Training completed!")
        print(f"📊 Best validation F1: {best_val_f1:.4f}")
        
        # Test evaluation nếu có
        if self.test_loader:
            print("\n🔄 Evaluating on test set...")
            
            # Load best model
            self.model.load_state_dict(
                torch.load(os.path.join(save_dir, 'best_model_enhanced.pth'))
            )
            
            test_results = self.evaluate(self.test_loader, "Test")
            
            # Classification report for detailed metrics
            report = classification_report(
                test_results['labels'], 
                test_results['predictions'],
                target_names=['non-clickbait', 'clickbait'],
                output_dict=True
            )
            
            # Save test results with detailed metrics
            detailed_results = test_results.copy()
            detailed_results['detailed_report'] = report
            
            with open(os.path.join(save_dir, 'test_results_enhanced.json'), 'w') as f:
                # Convert numpy arrays and numpy types to Python native types for JSON serialization
                serializable_results = {}
                for key, value in detailed_results.items():
                    if isinstance(value, np.ndarray):
                        serializable_results[key] = value.tolist()
                    elif isinstance(value, (np.integer, np.floating)):
                        serializable_results[key] = value.item()
                    elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], (np.integer, np.floating)):
                        serializable_results[key] = [v.item() if isinstance(v, (np.integer, np.floating)) else v for v in value]
                    elif isinstance(value, dict):
                        # Handle nested dictionaries (like classification report)
                        serializable_results[key] = value
                    else:
                        serializable_results[key] = value
                json.dump(serializable_results, f, indent=2)
            
            # Classification report
            report = classification_report(
                test_results['labels'], 
                test_results['predictions'],
                target_names=['non-clickbait', 'clickbait'],
                output_dict=True
            )
            
            print("\n📋 Detailed Test Results:")
            print(classification_report(
                test_results['labels'], 
                test_results['predictions'],
                target_names=['non-clickbait', 'clickbait']
            ))
            
            # Enhanced metrics table
            print("\n" + "="*90)
            print("📊 COMPREHENSIVE EVALUATION METRICS")
            print("="*90)
            
            # Extract metrics
            clickbait_precision = report['clickbait']['precision']
            clickbait_f1 = report['clickbait']['f1-score']
            non_clickbait_precision = report['non-clickbait']['precision']
            non_clickbait_f1 = report['non-clickbait']['f1-score']
            macro_f1 = report['macro avg']['f1-score']
            weighted_f1 = report['weighted avg']['f1-score']
            accuracy = report['accuracy']
            
            # Create formatted table
            print(f"{'Model':<20} {'Clickbait':<20} {'Non-Clickbait':<20} {'Macro F1':<10} {'Weighted F1':<12} {'Accuracy':<10}")
            print(f"{'':<20} {'Precision':<10} {'F1':<10} {'Precision':<10} {'F1':<10} {'':<10} {'':<12} {'':<10}")
            print("-" * 95)
            
            # Create data row with proper alignment
            model_name = "Enhanced LSTM"
            data_row = (f"{model_name:<20} "
                       f"{clickbait_precision:<10.4f} {clickbait_f1:<10.4f} "
                       f"{non_clickbait_precision:<10.4f} {non_clickbait_f1:<10.4f} "
                       f"{macro_f1:<10.4f} {weighted_f1:<12.4f} {accuracy:<10.4f}")
            print(data_row)
            print("="*95)
        
        return history

def load_data(data_dir):
    """Load training data"""
    data = {}
    
    for split in ['train', 'val', 'test']:
        # Handle both flat structure (data/) and nested structure (simple_dataset/)
        file_path_flat = os.path.join(data_dir, f'{split}.csv')
        file_path_nested = os.path.join(data_dir, split, f'{split}.csv')
        
        file_path = None
        if os.path.exists(file_path_flat):
            file_path = file_path_flat
        elif os.path.exists(file_path_nested):
            file_path = file_path_nested
        
        if file_path:
            print(f"📥 Loading {split} data from {file_path}")
            df = pd.read_csv(file_path)
            
            # Handle different column names
            if 'text' in df.columns:
                texts = df['text'].tolist()
            elif 'title' in df.columns:
                texts = df['title'].tolist()
            else:
                print(f"⚠️ Không tìm thấy cột text hoặc title trong {file_path}")
                continue
            
            labels = df['label'].tolist()
            
            # Convert labels to integers
            label_map = {'non-clickbait': 0, 'clickbait': 1}
            labels = [label_map.get(label, 0) for label in labels]
            
            data[split] = (texts, labels)
            print(f"   {split}: {len(texts)} samples")
            print(f"   Clickbait: {sum(labels)}, Non-clickbait: {len(labels) - sum(labels)}")
        else:
            print(f"⚠️ Không tìm thấy {split} data trong {data_dir}")
    
    return data

def main():
    parser = argparse.ArgumentParser(description="Enhanced LSTM with Pre-trained Embeddings")
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='simple_dataset',
                       help='Directory containing train/val/test data')
    parser.add_argument('--output_dir', type=str, default='models_lstm_enhanced',
                       help='Output directory for models and results')
    
    # Model arguments
    parser.add_argument('--embedding_dim', type=int, default=300,
                       help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate')
    
    # Embedding arguments
    parser.add_argument('--embedding_type', type=str, default='random',
                       choices=['random', 'word2vec', 'fasttext'],
                       help='Type of embeddings to use')
    parser.add_argument('--embedding_path', type=str, default=None,
                       help='Path to pre-trained embedding file')
    parser.add_argument('--freeze_embeddings', action='store_true',
                       help='Freeze embedding layer during training')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'plateau', 'none'],
                       help='Learning rate scheduler')
    
    # Text processing arguments
    parser.add_argument('--max_vocab_size', type=int, default=15000,
                       help='Maximum vocabulary size')
    parser.add_argument('--max_length', type=int, default=150,
                       help='Maximum sequence length')
    parser.add_argument('--min_word_freq', type=int, default=2,
                       help='Minimum word frequency')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    print("=== ENHANCED LSTM WITH PRE-TRAINED EMBEDDINGS ===")
    print(f"📁 Data directory: {args.data_dir}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🔧 Embedding type: {args.embedding_type}")
    if args.embedding_path:
        print(f"📥 Embedding path: {args.embedding_path}")
    print()
    
    # Load data
    data = load_data(args.data_dir)
    
    if 'train' not in data:
        print("❌ Training data not found!")
        return
    
    train_texts, train_labels = data['train']
    
    # Initialize text processor
    processor = EnhancedVietnameseTextProcessor(
        max_vocab_size=args.max_vocab_size,
        min_word_freq=args.min_word_freq,
        max_length=args.max_length,
        embedding_type=args.embedding_type,
        embedding_path=args.embedding_path,
        embedding_dim=args.embedding_dim
    )
    
    # Build vocabulary
    processor.build_vocabulary(train_texts)
    
    # Convert texts to sequences
    print("🔄 Converting texts to sequences...")
    train_sequences = processor.texts_to_sequences(train_texts)
    
    # Prepare validation data
    if 'val' in data:
        val_texts, val_labels = data['val']
        val_sequences = processor.texts_to_sequences(val_texts)
    else:
        print("⚠️ No validation data found")
        val_sequences, val_labels = None, None
    
    # Prepare test data
    test_sequences, test_labels = None, None
    if 'test' in data:
        test_texts, test_labels = data['test']
        test_sequences = processor.texts_to_sequences(test_texts)
    
    # Create datasets
    train_dataset = ClickbaitDataset(train_sequences, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, collate_fn=collate_fn)
    
    val_loader = None
    if val_sequences:
        val_dataset = ClickbaitDataset(val_sequences, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                               shuffle=False, collate_fn=collate_fn)
    
    test_loader = None
    if test_sequences:
        test_dataset = ClickbaitDataset(test_sequences, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                                shuffle=False, collate_fn=collate_fn)
    
    # Create model
    model = EnhancedLSTM(
        vocab_size=processor.vocab_size,
        embedding_dim=processor.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_classes=2,
        embedding_matrix=processor.embedding_matrix,
        freeze_embeddings=args.freeze_embeddings
    )
    
    # Create trainer
    trainer = EnhancedLSTMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=args.learning_rate,
        device=device,
        scheduler_type=args.scheduler
    )
    
    # Train model
    history = trainer.train(args.num_epochs, args.output_dir)
    
    # Save processor
    processor_path = os.path.join(args.output_dir, 'text_processor_enhanced.pkl')
    with open(processor_path, 'wb') as f:
        pickle.dump(processor, f)
    
    print(f"\n✅ Enhanced LSTM training completed!")
    print(f"📁 Models saved in: {args.output_dir}")

if __name__ == "__main__":
    main() 