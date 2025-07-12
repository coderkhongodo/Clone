#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script để hiển thị kết quả comprehensive từ các model đã train
"""

import os
import json
import argparse
from typing import List, Dict

def load_model_results(model_dir: str) -> Dict:
    """Load kết quả từ model directory"""
    results = {}
    
    # Load test results
    test_file = os.path.join(model_dir, 'test_results_enhanced.json')
    if os.path.exists(test_file):
        with open(test_file, 'r') as f:
            test_results = json.load(f)
        results['test'] = test_results
    
    # Load training history
    history_file = os.path.join(model_dir, 'training_history_enhanced.json')
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
        results['history'] = history
    
    return results

def extract_comprehensive_metrics(results: Dict) -> Dict:
    """Extract comprehensive metrics từ results"""
    if 'test' not in results:
        return None
    
    test_results = results['test']
    
    # Extract basic metrics
    accuracy = test_results.get('accuracy', 0)
    precision = test_results.get('precision', 0)
    recall = test_results.get('recall', 0)
    f1 = test_results.get('f1', 0)
    
    # Extract detailed report if available
    if 'detailed_report' in test_results:
        report = test_results['detailed_report']
        
        clickbait_precision = report['clickbait']['precision']
        clickbait_f1 = report['clickbait']['f1-score']
        non_clickbait_precision = report['non-clickbait']['precision']
        non_clickbait_f1 = report['non-clickbait']['f1-score']
        macro_f1 = report['macro avg']['f1-score']
        weighted_f1 = report['weighted avg']['f1-score']
        
    else:
        # Fallback to basic metrics
        clickbait_precision = precision
        clickbait_f1 = f1
        non_clickbait_precision = precision
        non_clickbait_f1 = f1
        macro_f1 = f1
        weighted_f1 = f1
    
    return {
        'accuracy': accuracy,
        'clickbait_precision': clickbait_precision,
        'clickbait_f1': clickbait_f1,
        'non_clickbait_precision': non_clickbait_precision,
        'non_clickbait_f1': non_clickbait_f1,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'overall_f1': f1,
        'best_val_f1': results.get('history', {}).get('best_val_f1', 0)
    }

def display_comprehensive_table(models_data: List[Dict]):
    """Hiển thị bảng comprehensive metrics"""
    print("\n" + "="*100)
    print("📊 COMPREHENSIVE EVALUATION METRICS")
    print("="*100)
    
    # Table header
    print(f"{'Model':<20} {'Clickbait':<20} {'Non-Clickbait':<20} {'Macro F1':<10} {'Weighted F1':<12} {'Accuracy':<10}")
    print(f"{'':<20} {'Precision':<10} {'F1':<10} {'Precision':<10} {'F1':<10} {'':<10} {'':<12} {'':<10}")
    print("-" * 100)
    
    # Sort by weighted F1 score
    models_data.sort(key=lambda x: x['metrics']['weighted_f1'], reverse=True)
    
    # Data rows
    for model_data in models_data:
        name = model_data['name']
        metrics = model_data['metrics']
        
        data_row = (f"{name:<20} "
                   f"{metrics['clickbait_precision']:<10.4f} {metrics['clickbait_f1']:<10.4f} "
                   f"{metrics['non_clickbait_precision']:<10.4f} {metrics['non_clickbait_f1']:<10.4f} "
                   f"{metrics['macro_f1']:<10.4f} {metrics['weighted_f1']:<12.4f} {metrics['accuracy']:<10.4f}")
        print(data_row)
    
    print("="*100)
    
    # Best model summary
    if models_data:
        best_model = models_data[0]
        print(f"\n🏆 BEST MODEL: {best_model['name']}")
        print(f"   Accuracy: {best_model['metrics']['accuracy']:.4f}")
        print(f"   Weighted F1: {best_model['metrics']['weighted_f1']:.4f}")
        print(f"   Macro F1: {best_model['metrics']['macro_f1']:.4f}")
        print(f"   Validation F1: {best_model['metrics']['best_val_f1']:.4f}")

def scan_models_directory(base_dir: str = "models_lstm_enhanced") -> List[Dict]:
    """Scan thư mục models để tìm các model đã train"""
    models_data = []
    
    if not os.path.exists(base_dir):
        print(f"⚠️ Không tìm thấy thư mục {base_dir}")
        return models_data
    
    # Check if base_dir is a single model directory
    if os.path.exists(os.path.join(base_dir, 'test_results_enhanced.json')):
        print(f"📥 Loading model từ {base_dir}")
        results = load_model_results(base_dir)
        metrics = extract_comprehensive_metrics(results)
        
        if metrics:
            models_data.append({
                'name': os.path.basename(base_dir),
                'path': base_dir,
                'metrics': metrics
            })
    else:
        # Scan subdirectories
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            
            if os.path.isdir(item_path):
                print(f"📥 Loading model từ {item_path}")
                results = load_model_results(item_path)
                metrics = extract_comprehensive_metrics(results)
                
                if metrics:
                    models_data.append({
                        'name': item,
                        'path': item_path,
                        'metrics': metrics
                    })
    
    return models_data

def compare_with_other_models():
    """So sánh với các model khác trong workspace"""
    other_models = []
    
    # Check original LSTM
    original_lstm_path = "models/test_results.json"
    if os.path.exists(original_lstm_path):
        print(f"📥 Loading Original LSTM từ {original_lstm_path}")
        with open(original_lstm_path, 'r') as f:
            results = json.load(f)
        
        # Convert to our format
        metrics = {
            'accuracy': results.get('accuracy', 0),
            'clickbait_precision': results.get('precision', 0),
            'clickbait_f1': results.get('f1', 0),
            'non_clickbait_precision': results.get('precision', 0),
            'non_clickbait_f1': results.get('f1', 0),
            'macro_f1': results.get('f1', 0),
            'weighted_f1': results.get('f1', 0),
            'best_val_f1': 0
        }
        
        other_models.append({
            'name': 'Original LSTM',
            'path': 'models/',
            'metrics': metrics
        })
    
    # Check LSTM no gensim
    lstm_no_gensim_path = "models_lstm_no_gensim/test_results.json"
    if os.path.exists(lstm_no_gensim_path):
        print(f"📥 Loading LSTM No Gensim từ {lstm_no_gensim_path}")
        with open(lstm_no_gensim_path, 'r') as f:
            results = json.load(f)
        
        metrics = {
            'accuracy': results.get('accuracy', 0),
            'clickbait_precision': results.get('precision', 0),
            'clickbait_f1': results.get('f1', 0),
            'non_clickbait_precision': results.get('precision', 0),
            'non_clickbait_f1': results.get('f1', 0),
            'macro_f1': results.get('f1', 0),
            'weighted_f1': results.get('f1', 0),
            'best_val_f1': 0
        }
        
        other_models.append({
            'name': 'LSTM No Gensim',
            'path': 'models_lstm_no_gensim/',
            'metrics': metrics
        })
    
    return other_models

def main():
    parser = argparse.ArgumentParser(description="Show comprehensive model results")
    parser.add_argument('--model_dir', type=str, default='models_lstm_enhanced',
                       help='Model directory to analyze')
    parser.add_argument('--compare_all', action='store_true',
                       help='Compare với tất cả models trong workspace')
    
    args = parser.parse_args()
    
    print("=== COMPREHENSIVE MODEL RESULTS ===")
    print()
    
    # Load Enhanced LSTM models
    enhanced_models = scan_models_directory(args.model_dir)
    
    if not enhanced_models:
        print(f"❌ Không tìm thấy model nào trong {args.model_dir}")
        return
    
    if args.compare_all:
        # Load other models for comparison
        other_models = compare_with_other_models()
        all_models = enhanced_models + other_models
        
        if all_models:
            display_comprehensive_table(all_models)
        else:
            print("❌ Không tìm thấy model nào để so sánh")
    else:
        # Show only Enhanced LSTM models
        display_comprehensive_table(enhanced_models)

if __name__ == "__main__":
    main() 