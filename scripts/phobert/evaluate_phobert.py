#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pickle
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

from train_phobert import PhoBERTClassifier, ClickbaitDataset

def load_model(checkpoint_path, model_name, num_classes=2):
    """Load trained model từ checkpoint"""
    import torch.serialization
    import numpy as np
    
    # Fix PyTorch 2.6 weights_only issue - try weights_only=False first
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print("SUCCESS: Loaded checkpoint with weights_only=False")
    except Exception as e:
        print(f"WARNING: Loading with weights_only=False failed: {e}")
        try:
            # Add more numpy types to safe globals
            safe_globals = [
                torch.storage._LegacyStorage,
                np.core.multiarray.scalar,
                np.ndarray,
                np.dtype,
                np.core.multiarray._reconstruct,
                np.dtypes.Float64DType,  # Add this specifically
                np.dtypes.Int64DType,
                np.dtypes.Float32DType,
                np.dtypes.Int32DType,
                type(np.dtype('float64')),  # Add dtype types
                type(np.dtype('int64')),
                type(np.dtype('float32')),
                type(np.dtype('int32'))
            ]
            
            torch.serialization.add_safe_globals(safe_globals)
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            print("SUCCESS: Loaded checkpoint with safe_globals and weights_only=True")
        except Exception as e2:
            print(f"WARNING: Loading with safe_globals failed: {e2}")
            print("CONFIG: Trying final fallback without weights_only argument...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
            print("SUCCESS: Loaded checkpoint with fallback method")
    
    # Get config from checkpoint if available
    if 'args' in checkpoint:
        args = checkpoint['args']
        dropout_rate = args.get('dropout_rate', 0.3)
        num_freeze_layers = args.get('num_freeze_layers', 0)
    else:
        dropout_rate = 0.3
        num_freeze_layers = 0
    
    model = PhoBERTClassifier(
        model_name=model_name,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        num_freeze_layers=num_freeze_layers
    )
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model, checkpoint

def evaluate_model(model, data_loader, device, label_mapping):
    """Evaluate model và trả về detailed metrics"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=-1)
            _, predicted = torch.max(logits.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Tính metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted'
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        all_labels, all_predictions, average=None
    )
    
    # Classification report
    label_names = list(label_mapping.keys())
    report = classification_report(
        all_labels, all_predictions,
        target_names=label_names,
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'support_per_class': support,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities,
        'confusion_matrix': cm,
        'classification_report': report,
        'label_names': label_names
    }
    
    return results

def plot_confusion_matrix(cm, label_names, title="Confusion Matrix", save_path=None):
    """Vẽ confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"STATS: Confusion matrix saved: {save_path}")
    
    plt.show()

def plot_metrics_comparison(results_dict, save_path=None):
    """So sánh metrics giữa các models"""
    models = list(results_dict.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(models))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [results_dict[model][metric] for model in models]
        ax.bar(x + i * width, values, width, label=metric.title())
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Thêm values trên bars
    for i, metric in enumerate(metrics):
        values = [results_dict[model][metric] for model in models]
        for j, v in enumerate(values):
            ax.text(j + i * width, v + 0.001, f'{v:.3f}', 
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"STATS: Metrics comparison saved: {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Evaluate PhoBERT models")
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model_name', type=str, required=True,
                       help='PhoBERT model name')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to test data (pickle file)')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"CONFIG: Device: {device}")
    
    # Load model
    print("PACKAGE: Loading model...")
    model, checkpoint = load_model(args.model_path, args.model_name)
    model.to(device)
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    # Load test data
    print("LOAD: Loading test data...")
    test_dataset = ClickbaitDataset(args.data_path)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Get label mapping
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)
    label_mapping = data['label_mapping']
    
    print(f"STATS: Label mapping: {label_mapping}")
    print(f"STATS: Test samples: {len(test_dataset)}")
    
    # Evaluate
    print("TEST: Evaluating model...")
    results = evaluate_model(model, test_loader, device, label_mapping)
    
    # Print results
    print("\n" + "="*60)
    print("REPORT: EVALUATION RESULTS")
    print("="*60)
    print("\nTARGET: MAIN PERFORMANCE METRICS:")
    print("-" * 40)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-Score:  {results['f1']:.4f}")
    print("-" * 40)
    
    # Summary table format
    print(f"\nSTATS: SUMMARY TABLE:")
    print(f"{'Metric':<12} {'Score':<8}")
    print("-" * 20)
    print(f"{'Accuracy':<12} {results['accuracy']:.4f}")
    print(f"{'Precision':<12} {results['precision']:.4f}")
    print(f"{'Recall':<12} {results['recall']:.4f}")
    print(f"{'F1-Score':<12} {results['f1']:.4f}")
    print("-" * 20)
    
    print(f"\nSTATS: PER-CLASS METRICS:")
    print("-" * 50)
    for i, label in enumerate(results['label_names']):
        print(f"\n{label.upper()}:")
        print(f"  Accuracy:  N/A     (Overall metric)")
        print(f"  Precision: {results['precision_per_class'][i]:.4f}")
        print(f"  Recall:    {results['recall_per_class'][i]:.4f}")
        print(f"  F1-Score:  {results['f1_per_class'][i]:.4f}")
        print(f"  Support:   {results['support_per_class'][i]} samples")
    print("-" * 50)
    
    # Classification report
    print(f"\nREPORT: Detailed Classification Report:")
    report_str = classification_report(
        results['labels'], 
        results['predictions'],
        target_names=results['label_names']
    )
    print(report_str)
    
    # Save results
    output_file = os.path.join(args.output_dir, 'evaluation_results.json')
    
    # Convert numpy arrays to lists for JSON serialization với formatting
    results_to_save = {
        'main_metrics': {
            'accuracy': round(float(results['accuracy']), 4),
            'precision': round(float(results['precision']), 4),
            'recall': round(float(results['recall']), 4),
            'f1_score': round(float(results['f1']), 4)
        },
        'detailed_metrics': {
            'precision_per_class': [round(float(x), 4) for x in results['precision_per_class']],
            'recall_per_class': [round(float(x), 4) for x in results['recall_per_class']],
            'f1_per_class': [round(float(x), 4) for x in results['f1_per_class']],
            'support_per_class': results['support_per_class'].tolist()
        },
        'confusion_matrix': results['confusion_matrix'].tolist(),
        'classification_report': results['classification_report'],
        'label_mapping': label_mapping,
        'model_info': {
            'model_name': args.model_name,
            'model_path': args.model_path,
            'test_samples': len(test_dataset)
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, indent=2, ensure_ascii=False)
    
    print(f"\nSAVE: Results saved to: {output_file}")
    
    # Save confusion matrix plot
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(
        results['confusion_matrix'], 
        results['label_names'],
        title=f"Confusion Matrix - {args.model_name}",
        save_path=cm_path
    )
    
    # Save detailed report
    report_path = os.path.join(args.output_dir, 'classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Test samples: {len(test_dataset)}\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n\n")
        f.write(report_str)
    
    print(f"REPORT: Report saved to: {report_path}")
    
    # Final summary với 4 metrics chính
    print("\n" + "="*60)
    print("BEST: FINAL SUMMARY - 4 MAIN METRICS")
    print("="*60)
    print(f"TARGET: Accuracy:  {results['accuracy']:.4f}")
    print(f"TARGET: Precision: {results['precision']:.4f}")
    print(f"TARGET: Recall:    {results['recall']:.4f}")
    print(f"TARGET: F1-Score:  {results['f1']:.4f}")
    print("="*60)
    print("SUCCESS: Evaluation completed!")

if __name__ == "__main__":
    main() 