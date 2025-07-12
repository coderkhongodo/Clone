#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Demo script thử nghiệm XLM-RoBERTa cho clickbait detection
Dataset: simple_dataset (title classification)
"""

import os
import pandas as pd
import torch
import numpy as np
from typing import List, Dict
import argparse
from tqdm import tqdm
import json
from datetime import datetime

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, precision_score, recall_score, f1_score
)
import seaborn as sns
import matplotlib.pyplot as plt

class XLMRobertaDataset(torch.utils.data.Dataset):
    """Dataset class cho XLM-RoBERTa"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class XLMRobertaClickbaitClassifier:
    """XLM-RoBERTa classifier cho clickbait detection"""
    
    def __init__(self, model_name: str = 'xlm-roberta-base', max_length: int = 256):
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_names = ['non-clickbait', 'clickbait']
        
        print(f"🚀 Khởi tạo XLM-RoBERTa Classifier")
        print(f"   Model: {model_name}")
        print(f"   Device: {self.device}")
        print(f"   Max length: {max_length}")
        
        # Load tokenizer và model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2
        )
        self.model.to(self.device)
        
    def load_data(self, data_path: str) -> Dict:
        """Load dữ liệu từ simple_dataset"""
        print(f"📂 Loading data từ {data_path}")
        
        # Load train, val, test
        train_df = pd.read_csv(os.path.join(data_path, 'train/train.csv'))
        val_df = pd.read_csv(os.path.join(data_path, 'val/val.csv'))
        test_df = pd.read_csv(os.path.join(data_path, 'test/test.csv'))
        
        # Convert labels
        label_map = {'non-clickbait': 0, 'clickbait': 1}
        
        data = {
            'train': {
                'texts': train_df['title'].tolist(),
                'labels': [label_map[label] for label in train_df['label']]
            },
            'val': {
                'texts': val_df['title'].tolist(),
                'labels': [label_map[label] for label in val_df['label']]
            },
            'test': {
                'texts': test_df['title'].tolist(),
                'labels': [label_map[label] for label in test_df['label']]
            }
        }
        
        print(f"✅ Data loaded successfully:")
        print(f"   Train: {len(data['train']['texts'])} samples")
        print(f"   Val: {len(data['val']['texts'])} samples")  
        print(f"   Test: {len(data['test']['texts'])} samples")
        
        # Hiển thị phân bố labels
        for split in ['train', 'val', 'test']:
            labels = data[split]['labels']
            clickbait_count = sum(labels)
            total = len(labels)
            print(f"   {split.capitalize()}: {clickbait_count}/{total} clickbait ({clickbait_count/total*100:.1f}%)")
        
        return data
    
    def create_datasets(self, data: Dict):
        """Tạo PyTorch datasets"""
        train_dataset = XLMRobertaDataset(
            data['train']['texts'], 
            data['train']['labels'], 
            self.tokenizer, 
            self.max_length
        )
        
        val_dataset = XLMRobertaDataset(
            data['val']['texts'], 
            data['val']['labels'], 
            self.tokenizer, 
            self.max_length
        )
        
        test_dataset = XLMRobertaDataset(
            data['test']['texts'], 
            data['test']['labels'], 
            self.tokenizer, 
            self.max_length
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute metrics cho Trainer"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train(self, train_dataset, val_dataset, output_dir: str = './xlm_roberta_results', 
              num_epochs: int = 3, batch_size: int = 16, learning_rate: float = 2e-5):
        """Train model"""
        print(f"\n🔄 Bắt đầu training...")
        print(f"   Epochs: {num_epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {learning_rate}")
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=50,
            eval_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=200,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            report_to=None  # Disable wandb
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
        )
        
        # Train
        trainer.train()
        
        # Save model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"✅ Training completed! Model saved to {output_dir}")
        
        return trainer
    
    def evaluate(self, test_dataset, output_dir: str = './xlm_roberta_results'):
        """Evaluate model trên test set với chi tiết đầy đủ"""
        print(f"\n📊 Evaluating trên test set...")
        
        # Load best model
        self.model = AutoModelForSequenceClassification.from_pretrained(output_dir)
        self.model.to(self.device)
        self.model.eval()
        
        # Predict
        predictions = []
        true_labels = []
        prediction_probs = []
        
        dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                
                preds = torch.argmax(outputs.logits, dim=-1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                prediction_probs.extend(probabilities.cpu().numpy())
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(true_labels, predictions)
        
        # Overall metrics (weighted average)
        precision_weighted = precision_score(true_labels, predictions, average='weighted')
        recall_weighted = recall_score(true_labels, predictions, average='weighted')
        f1_weighted = f1_score(true_labels, predictions, average='weighted')
        
        # Per-class metrics
        precision_per_class = precision_score(true_labels, predictions, average=None)
        recall_per_class = recall_score(true_labels, predictions, average=None)
        f1_per_class = f1_score(true_labels, predictions, average=None)
        
        # Macro average
        precision_macro = precision_score(true_labels, predictions, average='macro')
        recall_macro = recall_score(true_labels, predictions, average='macro')
        f1_macro = f1_score(true_labels, predictions, average='macro')
        
        # Support (số samples thực tế cho mỗi class)
        unique, counts = np.unique(true_labels, return_counts=True)
        support_per_class = counts.tolist()
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # Classification report
        class_report = classification_report(
            true_labels, predictions, 
            target_names=self.label_names,
            output_dict=True
        )
        
        # Detailed results structure
        results = {
            "accuracy": accuracy,
            "precision": precision_weighted,
            "recall": recall_weighted,
            "f1": f1_weighted,
            "precision_per_class": precision_per_class.tolist(),
            "recall_per_class": recall_per_class.tolist(),
            "f1_per_class": f1_per_class.tolist(),
            "support_per_class": support_per_class,
            "confusion_matrix": cm.tolist(),
            "classification_report": {
                self.label_names[0]: {
                    "precision": float(precision_per_class[0]),
                    "recall": float(recall_per_class[0]),
                    "f1-score": float(f1_per_class[0]),
                    "support": float(support_per_class[0])
                },
                self.label_names[1]: {
                    "precision": float(precision_per_class[1]),
                    "recall": float(recall_per_class[1]),
                    "f1-score": float(f1_per_class[1]),
                    "support": float(support_per_class[1])
                },
                "accuracy": accuracy,
                "macro avg": {
                    "precision": precision_macro,
                    "recall": recall_macro,
                    "f1-score": f1_macro,
                    "support": float(sum(support_per_class))
                },
                "weighted avg": {
                    "precision": precision_weighted,
                    "recall": recall_weighted,
                    "f1-score": f1_weighted,
                    "support": float(sum(support_per_class))
                }
            },
            "label_mapping": {
                self.label_names[0]: 0,
                self.label_names[1]: 1
            },
            "model_info": {
                "model_name": self.model_name,
                "model_path": f"{output_dir}/model.safetensors",
                "test_samples": len(true_labels),
                "device": str(self.device),
                "max_length": self.max_length
            }
        }
        
        # Print detailed results
        print(f"\n📈 DETAILED TEST RESULTS:")
        print(f"=" * 60)
        print(f"📊 Overall Metrics:")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Precision (weighted): {precision_weighted:.4f}")
        print(f"   Recall (weighted): {recall_weighted:.4f}")
        print(f"   F1-score (weighted): {f1_weighted:.4f}")
        
        print(f"\n🎯 Per-Class Metrics:")
        for i, label in enumerate(self.label_names):
            print(f"   {label.upper()}:")
            print(f"     Precision: {precision_per_class[i]:.4f}")
            print(f"     Recall: {recall_per_class[i]:.4f}")
            print(f"     F1-score: {f1_per_class[i]:.4f}")
            print(f"     Support: {support_per_class[i]} samples")
        
        print(f"\n📈 Macro Average:")
        print(f"   Precision: {precision_macro:.4f}")
        print(f"   Recall: {recall_macro:.4f}")
        print(f"   F1-score: {f1_macro:.4f}")
        
        print(f"\n🔢 Confusion Matrix:")
        print(f"   Predicted →")
        print(f"   Actual ↓     {self.label_names[0]:<15} {self.label_names[1]:<15}")
        print(f"   {self.label_names[0]:<12} {cm[0][0]:<15} {cm[0][1]:<15}")
        print(f"   {self.label_names[1]:<12} {cm[1][0]:<15} {cm[1][1]:<15}")
        
        # Calculate additional insights
        true_positives = cm[1][1]
        false_positives = cm[0][1]
        false_negatives = cm[1][0]
        true_negatives = cm[0][0]
        
        print(f"\n🔍 Detailed Analysis:")
        print(f"   True Positives (Correct clickbait): {true_positives}")
        print(f"   False Positives (Wrong clickbait): {false_positives}")
        print(f"   False Negatives (Missed clickbait): {false_negatives}")
        print(f"   True Negatives (Correct non-clickbait): {true_negatives}")
        
        # Error analysis
        clickbait_recall = recall_per_class[1] if len(recall_per_class) > 1 else recall_per_class[0]
        nonclickbait_recall = recall_per_class[0] if len(recall_per_class) > 1 else recall_per_class[1]
        
        print(f"\n📋 Performance Summary:")
        if clickbait_recall > 0.8:
            print(f"   ✅ Excellent clickbait detection: {clickbait_recall:.1%} recall")
        elif clickbait_recall > 0.7:
            print(f"   ✅ Good clickbait detection: {clickbait_recall:.1%} recall")
        else:
            print(f"   ⚠️ Room for improvement in clickbait detection: {clickbait_recall:.1%} recall")
            
        if nonclickbait_recall > 0.8:
            print(f"   ✅ Excellent non-clickbait detection: {nonclickbait_recall:.1%} recall")
        elif nonclickbait_recall > 0.7:
            print(f"   ✅ Good non-clickbait detection: {nonclickbait_recall:.1%} recall")
        else:
            print(f"   ⚠️ Room for improvement in non-clickbait detection: {nonclickbait_recall:.1%} recall")
        
        print(f"\n💾 Model Info:")
        print(f"   Model: {self.model_name}")
        print(f"   Test samples: {len(true_labels)}")
        print(f"   Device: {self.device}")
        print(f"   Max length: {self.max_length}")
        
        # Save results
        with open(f'{output_dir}/test_results_detailed.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_dir}/test_results_detailed.json")
        
        # Plot confusion matrix
        self.plot_confusion_matrix(true_labels, predictions, output_dir)
        
        return results
    
    def plot_confusion_matrix(self, true_labels, predictions, output_dir):
        """Vẽ confusion matrix chi tiết với percentages"""
        cm = confusion_matrix(true_labels, predictions)
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        plt.figure(figsize=(10, 8))
        
        # Create annotations with both counts and percentages
        annotations = []
        for i in range(cm.shape[0]):
            row = []
            for j in range(cm.shape[1]):
                count = cm[i, j]
                percent = cm_percent[i, j]
                row.append(f'{count}\n({percent:.1f}%)')
            annotations.append(row)
        
        # Plot heatmap
        sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', 
                   xticklabels=[f'{label}\n(Predicted)' for label in self.label_names], 
                   yticklabels=[f'{label}\n(Actual)' for label in self.label_names],
                   cbar_kws={'label': 'Number of Samples'})
        
        plt.title('Confusion Matrix - XLM-RoBERTa Clickbait Detection\n' + 
                 f'Total Samples: {cm.sum()}, Accuracy: {accuracy_score(true_labels, predictions):.3f}', 
                 fontsize=14, pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # Add performance metrics text
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='weighted')
        recall = recall_score(true_labels, predictions, average='weighted')
        f1 = f1_score(true_labels, predictions, average='weighted')
        
        metrics_text = f'Accuracy: {accuracy:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}'
        plt.figtext(0.5, 0.02, metrics_text, ha='center', fontsize=10, 
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/confusion_matrix_detailed.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create a simple version
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.label_names, yticklabels=self.label_names)
        plt.title('Confusion Matrix - XLM-RoBERTa')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=300)
        plt.close()
        
        print(f"✅ Confusion matrices saved:")
        print(f"   📊 Detailed: {output_dir}/confusion_matrix_detailed.png")
        print(f"   📊 Simple: {output_dir}/confusion_matrix.png")
    
    def predict_single(self, text: str, output_dir: str = './xlm_roberta_results') -> Dict:
        """Predict cho một text"""
        # Load model nếu chưa load
        if not hasattr(self, 'model') or self.model is None:
            self.model = AutoModelForSequenceClassification.from_pretrained(output_dir)
            self.model.to(self.device)
            self.model.eval()
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(outputs.logits, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        result = {
            'text': text,
            'predicted_label': self.label_names[predicted_class],
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': {
                'non-clickbait': probabilities[0][0].item(),
                'clickbait': probabilities[0][1].item()
            }
        }
        
        return result
    
    def generate_evaluation_report(self, results: Dict, output_dir: str):
        """Tạo báo cáo đánh giá chi tiết dạng text"""
        report_path = f"{output_dir}/evaluation_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("📊 XLM-RoBERTa CLICKBAIT DETECTION - EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Model info
            f.write("🤖 MODEL INFORMATION:\n")
            f.write("-"*40 + "\n")
            model_info = results['model_info']
            f.write(f"Model Name: {model_info['model_name']}\n")
            f.write(f"Test Samples: {model_info['test_samples']}\n")
            f.write(f"Device: {model_info['device']}\n")
            f.write(f"Max Length: {model_info['max_length']}\n\n")
            
            # Overall performance
            f.write("📈 OVERALL PERFORMANCE:\n")
            f.write("-"*40 + "\n")
            f.write(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)\n")
            f.write(f"Precision (weighted): {results['precision']:.4f}\n")
            f.write(f"Recall (weighted): {results['recall']:.4f}\n")
            f.write(f"F1-score (weighted): {results['f1']:.4f}\n\n")
            
            # Per-class performance
            f.write("🎯 PER-CLASS PERFORMANCE:\n")
            f.write("-"*40 + "\n")
            class_report = results['classification_report']
            for label in self.label_names:
                if label in class_report:
                    f.write(f"{label.upper()}:\n")
                    f.write(f"  Precision: {class_report[label]['precision']:.4f}\n")
                    f.write(f"  Recall: {class_report[label]['recall']:.4f}\n")
                    f.write(f"  F1-score: {class_report[label]['f1-score']:.4f}\n")
                    f.write(f"  Support: {int(class_report[label]['support'])} samples\n\n")
            
            # Macro and weighted averages
            f.write("📊 AVERAGE METRICS:\n")
            f.write("-"*40 + "\n")
            f.write("Macro Average:\n")
            f.write(f"  Precision: {class_report['macro avg']['precision']:.4f}\n")
            f.write(f"  Recall: {class_report['macro avg']['recall']:.4f}\n")
            f.write(f"  F1-score: {class_report['macro avg']['f1-score']:.4f}\n\n")
            
            f.write("Weighted Average:\n")
            f.write(f"  Precision: {class_report['weighted avg']['precision']:.4f}\n")
            f.write(f"  Recall: {class_report['weighted avg']['recall']:.4f}\n")
            f.write(f"  F1-score: {class_report['weighted avg']['f1-score']:.4f}\n\n")
            
            # Confusion matrix
            f.write("🔢 CONFUSION MATRIX:\n")
            f.write("-"*40 + "\n")
            cm = results['confusion_matrix']
            f.write(f"Predicted →     {self.label_names[0]:<15} {self.label_names[1]:<15}\n")
            f.write(f"Actual ↓\n")
            f.write(f"{self.label_names[0]:<15} {cm[0][0]:<15} {cm[0][1]:<15}\n")
            f.write(f"{self.label_names[1]:<15} {cm[1][0]:<15} {cm[1][1]:<15}\n\n")
            
            # Error analysis
            f.write("🔍 ERROR ANALYSIS:\n")
            f.write("-"*40 + "\n")
            true_positives = cm[1][1]
            false_positives = cm[0][1]
            false_negatives = cm[1][0]
            true_negatives = cm[0][0]
            
            f.write(f"True Positives (Correct clickbait predictions): {true_positives}\n")
            f.write(f"False Positives (Incorrect clickbait predictions): {false_positives}\n")
            f.write(f"False Negatives (Missed clickbait): {false_negatives}\n")
            f.write(f"True Negatives (Correct non-clickbait predictions): {true_negatives}\n\n")
            
            # Performance insights
            f.write("💡 PERFORMANCE INSIGHTS:\n")
            f.write("-"*40 + "\n")
            
            clickbait_precision = results['precision_per_class'][1] if len(results['precision_per_class']) > 1 else 0
            clickbait_recall = results['recall_per_class'][1] if len(results['recall_per_class']) > 1 else 0
            
            if clickbait_recall > 0.8:
                f.write("✅ Excellent clickbait detection capability\n")
            elif clickbait_recall > 0.7:
                f.write("✅ Good clickbait detection capability\n")
            else:
                f.write("⚠️ Room for improvement in clickbait detection\n")
            
            if clickbait_precision > 0.8:
                f.write("✅ High precision in clickbait predictions (low false positives)\n")
            elif clickbait_precision > 0.7:
                f.write("✅ Good precision in clickbait predictions\n")
            else:
                f.write("⚠️ Consider reducing false positive rate\n")
            
            # Recommendations
            f.write("\n📋 RECOMMENDATIONS:\n")
            f.write("-"*40 + "\n")
            if results['accuracy'] > 0.8:
                f.write("🎯 Model performance is excellent and ready for production\n")
            elif results['accuracy'] > 0.7:
                f.write("🎯 Model performance is good, consider fine-tuning for improvement\n")
            else:
                f.write("🎯 Model needs more training or data augmentation\n")
            
            if false_positives > false_negatives:
                f.write("⚠️ Consider increasing prediction threshold to reduce false positives\n")
            elif false_negatives > false_positives:
                f.write("⚠️ Consider decreasing prediction threshold to catch more clickbait\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("End of Report\n")
            f.write("="*80 + "\n")
        
        print(f"📄 Evaluation report saved to: {report_path}")

def demo_training(args):
    """Demo training XLM-RoBERTa"""
    print("="*60)
    print("🚀 DEMO TRAINING XLM-RoBERTa")
    print("="*60)
    
    # Initialize classifier
    classifier = XLMRobertaClickbaitClassifier(
        model_name=args.model_name,
        max_length=args.max_length
    )
    
    # Load data
    data = classifier.load_data(args.data_path)
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = classifier.create_datasets(data)
    
    # Train
    trainer = classifier.train(
        train_dataset, 
        val_dataset, 
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # Evaluate
    results = classifier.evaluate(test_dataset, args.output_dir)
    
    # Generate comprehensive report
    classifier.generate_evaluation_report(results, args.output_dir)
    
    print("\n" + "="*60)
    print("✅ DEMO TRAINING HOÀN THÀNH!")
    print("="*60)
    print(f"📁 Results saved in: {args.output_dir}/")
    print(f"   📊 JSON: test_results_detailed.json")
    print(f"   📄 Report: evaluation_report.txt")
    print(f"   📈 Charts: confusion_matrix_detailed.png")
    print(f"   🤖 Model: model.safetensors")
    
    return classifier, results

def demo_interactive(args):
    """Demo interactive prediction"""
    print("="*60)
    print("🎮 DEMO INTERACTIVE - XLM-RoBERTa")
    print("="*60)
    
    classifier = XLMRobertaClickbaitClassifier()
    
    print("Nhập text để phân loại clickbait (gõ 'quit' để thoát)")
    print()
    
    while True:
        text = input("📝 Nhập text: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("👋 Tạm biệt!")
            break
        
        if not text:
            continue
        
        try:
            result = classifier.predict_single(text, args.output_dir)
            
            print(f"\n📊 Kết quả:")
            print(f"   Label: {result['predicted_label'].upper()}")
            print(f"   Confidence: {result['confidence']:.3f}")
            print(f"   Probabilities:")
            print(f"     Non-clickbait: {result['probabilities']['non-clickbait']:.3f}")
            print(f"     Clickbait: {result['probabilities']['clickbait']:.3f}")
            
            # Interpretation
            if result['confidence'] > 0.8:
                conf_level = "RẤT TIN CẬY"
            elif result['confidence'] > 0.6:
                conf_level = "TIN CẬY"
            else:
                conf_level = "KHÔNG CHẮC CHẮN"
            
            print(f"   Đánh giá: {conf_level}")
            print()
            
        except Exception as e:
            print(f"❌ Lỗi: {e}")
            print("Hãy đảm bảo model đã được train trước!")

def demo_predefined_examples(args):
    """Demo với examples có sẵn"""
    print("="*60)
    print("📋 DEMO VỚI EXAMPLES CÓ SẴN")
    print("="*60)
    
    classifier = XLMRobertaClickbaitClassifier()
    
    examples = [
        "5 bí mật gây sốc mà bạn chưa bao giờ biết về smartphone",
        "Nghiên cứu mới về tác động của AI đến giáo dục",
        "Cô gái 20 tuổi kiếm được 100 triệu/tháng bằng cách này",
        "Chính phủ công bố chính sách mới về thuế môi trường",
        "Bạn sẽ không tin những gì xảy ra khi cô ấy mở cửa",
        "Báo cáo tài chính quý 3 của các doanh nghiệp lớn"
    ]
    
    print("🔍 Đang phân tích các examples...")
    results = []
    
    for i, text in enumerate(examples, 1):
        try:
            result = classifier.predict_single(text, args.output_dir)
            results.append(result)
            
            print(f"\n{i}. [{result['predicted_label'].upper()}] (confidence: {result['confidence']:.3f})")
            print(f"   Text: {text}")
            
        except Exception as e:
            print(f"❌ Lỗi cho example {i}: {e}")
    
            # Tổng kết
        if results:
            clickbait_count = sum(1 for r in results if r['predicted_class'] == 1)
            high_conf_clickbait = sum(1 for r in results if r['predicted_class'] == 1 and r['confidence'] > 0.8)
            high_conf_nonclickbait = sum(1 for r in results if r['predicted_class'] == 0 and r['confidence'] > 0.8)
            avg_confidence = sum(r['confidence'] for r in results) / len(results)
            
            print(f"\n📊 Tổng kết chi tiết:")
            print(f"   Total examples: {len(results)}")
            print(f"   Clickbait predictions: {clickbait_count}/{len(results)} ({clickbait_count/len(results)*100:.1f}%)")
            print(f"   Non-clickbait predictions: {len(results)-clickbait_count}/{len(results)} ({(len(results)-clickbait_count)/len(results)*100:.1f}%)")
            print(f"   Average confidence: {avg_confidence:.3f}")
            print(f"   High confidence (>0.8):")
            print(f"     Clickbait: {high_conf_clickbait}/{clickbait_count} ({high_conf_clickbait/max(clickbait_count,1)*100:.1f}%)")
            print(f"     Non-clickbait: {high_conf_nonclickbait}/{len(results)-clickbait_count} ({high_conf_nonclickbait/max(len(results)-clickbait_count,1)*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='XLM-RoBERTa Clickbait Detection Demo')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'interactive', 'examples'],
                       help='Demo mode')
    parser.add_argument('--data_path', type=str, default='simple_dataset',
                       help='Path to dataset')
    parser.add_argument('--model_name', type=str, default='xlm-roberta-base',
                       help='XLM-RoBERTa model name')
    parser.add_argument('--output_dir', type=str, default='./xlm_roberta_results',
                       help='Output directory')
    parser.add_argument('--max_length', type=int, default=256,
                       help='Max sequence length')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    print(f"🎯 Mode: {args.mode}")
    print(f"📂 Data path: {args.data_path}")
    print(f"🤖 Model: {args.model_name}")
    print()
    
    if args.mode == 'train':
        demo_training(args)
    elif args.mode == 'interactive':
        demo_interactive(args)
    elif args.mode == 'examples':
        demo_predefined_examples(args)

if __name__ == "__main__":
    main() 