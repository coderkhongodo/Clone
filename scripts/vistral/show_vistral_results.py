#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script để hiển thị kết quả comprehensive từ các Vistral model đã test
"""

import os
import json
import argparse
import glob
from typing import List, Dict

def load_vistral_metrics(metrics_file: str) -> Dict:
    """Load metrics từ JSON file"""
    try:
        with open(metrics_file, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        return metrics
    except Exception as e:
        print(f"⚠️ Lỗi đọc {metrics_file}: {e}")
        return None

def scan_vistral_results(base_dir: str = ".") -> List[Dict]:
    """Scan thư mục để tìm các kết quả Vistral"""
    models_data = []
    
    # Tìm tất cả file metrics JSON của Vistral
    pattern = os.path.join(base_dir, "vistral_v03_*_metrics.json")
    metrics_files = glob.glob(pattern)
    
    print(f"📥 Tìm thấy {len(metrics_files)} file metrics Vistral")
    
    for metrics_file in metrics_files:
        print(f"📥 Loading metrics từ {metrics_file}")
        metrics = load_vistral_metrics(metrics_file)
        
        if metrics:
            models_data.append({
                'name': metrics['method'],
                'path': metrics_file,
                'metrics': metrics
            })
    
    return models_data

def display_comprehensive_table(models_data: List[Dict]):
    """Hiển thị bảng comprehensive metrics cho Vistral"""
    print("\n" + "="*100)
    print("📊 COMPREHENSIVE VISTRAL V03 EVALUATION METRICS")
    print("="*100)
    
    # Table header
    print(f"{'Model':<25} {'Clickbait':<20} {'Non-Clickbait':<20} {'Macro F1':<10} {'Weighted F1':<12} {'Accuracy':<10}")
    print(f"{'':<25} {'Precision':<10} {'F1':<10} {'Precision':<10} {'F1':<10} {'':<10} {'':<12} {'':<10}")
    print("-" * 100)
    
    # Sort by weighted F1 score
    models_data.sort(key=lambda x: x['metrics']['weighted_f1'], reverse=True)
    
    # Data rows
    for model_data in models_data:
        name = model_data['name']
        metrics = model_data['metrics']
        
        data_row = (f"{name:<25} "
                   f"{metrics['clickbait_precision']:<10.4f} {metrics['clickbait_f1']:<10.4f} "
                   f"{metrics['non_clickbait_precision']:<10.4f} {metrics['non_clickbait_f1']:<10.4f} "
                   f"{metrics['macro_f1']:<10.4f} {metrics['weighted_f1']:<12.4f} {metrics['accuracy']:<10.4f}")
        print(data_row)
    
    print("="*100)
    
    # Best model summary
    if models_data:
        best_model = models_data[0]
        print(f"\n🏆 BEST VISTRAL MODEL: {best_model['name']}")
        print(f"   Accuracy: {best_model['metrics']['accuracy']:.4f}")
        print(f"   Weighted F1: {best_model['metrics']['weighted_f1']:.4f}")
        print(f"   Macro F1: {best_model['metrics']['macro_f1']:.4f}")
        print(f"   Clickbait F1: {best_model['metrics']['clickbait_f1']:.4f}")
        print(f"   Non-clickbait F1: {best_model['metrics']['non_clickbait_f1']:.4f}")

def compare_with_lstm_models():
    """So sánh với Enhanced LSTM models"""
    lstm_models = []
    
    # Check Enhanced LSTM
    enhanced_lstm_path = "../../models_lstm_enhanced/test_results_enhanced.json"
    if os.path.exists(enhanced_lstm_path):
        print(f"📥 Loading Enhanced LSTM từ {enhanced_lstm_path}")
        with open(enhanced_lstm_path, 'r') as f:
            results = json.load(f)
        
        # Extract metrics from detailed report if available
        if 'detailed_report' in results:
            report = results['detailed_report']
            metrics = {
                'method': 'Enhanced LSTM',
                'accuracy': results['accuracy'],
                'macro_f1': report['macro avg']['f1-score'],
                'weighted_f1': report['weighted avg']['f1-score'],
                'clickbait_precision': report['clickbait']['precision'],
                'clickbait_f1': report['clickbait']['f1-score'],
                'non_clickbait_precision': report['non-clickbait']['precision'],
                'non_clickbait_f1': report['non-clickbait']['f1-score']
            }
        else:
            # Fallback to basic metrics
            metrics = {
                'method': 'Enhanced LSTM',
                'accuracy': results.get('accuracy', 0),
                'macro_f1': results.get('f1', 0),
                'weighted_f1': results.get('f1', 0),
                'clickbait_precision': results.get('precision', 0),
                'clickbait_f1': results.get('f1', 0),
                'non_clickbait_precision': results.get('precision', 0),
                'non_clickbait_f1': results.get('f1', 0)
            }
        
        lstm_models.append({
            'name': 'Enhanced LSTM',
            'path': enhanced_lstm_path,
            'metrics': metrics
        })
    
    return lstm_models

def display_comparison_table(vistral_models: List[Dict], lstm_models: List[Dict]):
    """Hiển thị bảng so sánh giữa Vistral và LSTM models"""
    all_models = vistral_models + lstm_models
    
    if not all_models:
        print("❌ Không có model nào để so sánh")
        return
    
    print("\n" + "="*100)
    print("🔥 COMPREHENSIVE COMPARISON: VISTRAL V03 vs ENHANCED LSTM")
    print("="*100)
    
    # Table header
    print(f"{'Model':<25} {'Clickbait':<20} {'Non-Clickbait':<20} {'Macro F1':<10} {'Weighted F1':<12} {'Accuracy':<10}")
    print(f"{'':<25} {'Precision':<10} {'F1':<10} {'Precision':<10} {'F1':<10} {'':<10} {'':<12} {'':<10}")
    print("-" * 100)
    
    # Sort by weighted F1 score
    all_models.sort(key=lambda x: x['metrics']['weighted_f1'], reverse=True)
    
    # Data rows with model type indicator
    for model_data in all_models:
        name = model_data['name']
        metrics = model_data['metrics']
        
        # Add model type indicator
        if 'Vistral' in name:
            name_display = f"🇻🇳 {name}"
        elif 'LSTM' in name:
            name_display = f"🧠 {name}"
        else:
            name_display = name
        
        data_row = (f"{name_display:<25} "
                   f"{metrics['clickbait_precision']:<10.4f} {metrics['clickbait_f1']:<10.4f} "
                   f"{metrics['non_clickbait_precision']:<10.4f} {metrics['non_clickbait_f1']:<10.4f} "
                   f"{metrics['macro_f1']:<10.4f} {metrics['weighted_f1']:<12.4f} {metrics['accuracy']:<10.4f}")
        print(data_row)
    
    print("="*100)
    
    # Champion model
    if all_models:
        champion = all_models[0]
        print(f"\n🏆 CHAMPION MODEL: {champion['name']}")
        print(f"   Accuracy: {champion['metrics']['accuracy']:.4f}")
        print(f"   Weighted F1: {champion['metrics']['weighted_f1']:.4f}")
        print(f"   Macro F1: {champion['metrics']['macro_f1']:.4f}")
        
        # Performance comparison
        if len(all_models) > 1:
            runner_up = all_models[1]
            improvement = champion['metrics']['weighted_f1'] - runner_up['metrics']['weighted_f1']
            print(f"   Performance advantage: +{improvement:.4f} Weighted F1 over {runner_up['name']}")

def main():
    parser = argparse.ArgumentParser(description="Show comprehensive Vistral results")
    parser.add_argument('--compare_lstm', action='store_true',
                       help='So sánh với Enhanced LSTM models')
    parser.add_argument('--base_dir', type=str, default='.',
                       help='Base directory để tìm kết quả (default: current directory)')
    
    args = parser.parse_args()
    
    print("=== COMPREHENSIVE VISTRAL V03 RESULTS ===")
    print()
    
    # Load Vistral models
    vistral_models = scan_vistral_results(args.base_dir)
    
    if not vistral_models:
        print(f"❌ Không tìm thấy kết quả Vistral nào trong {args.base_dir}")
        print("💡 Đảm bảo đã chạy test: python fine_tune_clickbait_vistral_v03.py --mode test")
        return
    
    if args.compare_lstm:
        # Load LSTM models for comparison
        lstm_models = compare_with_lstm_models()
        display_comparison_table(vistral_models, lstm_models)
    else:
        # Show only Vistral models
        display_comprehensive_table(vistral_models)

if __name__ == "__main__":
    main() 