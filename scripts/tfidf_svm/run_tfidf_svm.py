#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple launcher script cho TF-IDF + SVM training
"""

import subprocess
import sys
import os

def main():
    print("=== TF-IDF + SVM LAUNCHER ===")
    print()
    
    # Kiểm tra dữ liệu
    if not os.path.exists('data'):
        print("ERROR: Không tìm thấy thư mục data!")
        print("TIP: Đảm bảo có thư mục data với train/val/test")
        return
    
    print("Chọn chế độ:")
    print("1. Training với default settings")
    print("2. Training với grid search (tốt hơn nhưng chậm hơn)")
    print("3. Test model đã training")
    
    choice = input("\nNhập lựa chọn (1/2/3): ").strip()
    
    if choice == '1':
        # Training default
        cmd = [
            'python', 'train_tfidf_svm.py',
            '--data_dir', 'data',
            '--output_dir', 'models_tfidf_svm',
            '--max_features', '10000',
            '--ngram_range', '1,2'
        ]
        
        print("START: Bắt đầu training với default settings...")
        
    elif choice == '2':
        # Training với grid search
        cmd = [
            'python', 'train_tfidf_svm.py',
            '--data_dir', 'data',
            '--output_dir', 'models_tfidf_svm',
            '--max_features', '15000',
            '--ngram_range', '1,3',
            '--grid_search'
        ]
        
        print("START: Bắt đầu training với grid search...")
        print("⏰ Quá trình này có thể mất 30-60 phút...")
        
    elif choice == '3':
        # Test model
        model_path = input("Nhập đường dẫn model (default: models_tfidf_svm/tfidf_svm_model.pkl): ").strip()
        if not model_path:
            model_path = 'models_tfidf_svm/tfidf_svm_model.pkl'
        
        if not os.path.exists(model_path):
            print(f"ERROR: Không tìm thấy model: {model_path}")
            return
        
        cmd = [
            'python', 'train_tfidf_svm.py',
            '--data_dir', 'data',
            '--output_dir', 'models_tfidf_svm',
            '--test_only', model_path
        ]
        
        print(f"TEST: Testing model: {model_path}")
        
    else:
        print("ERROR: Lựa chọn không hợp lệ")
        return
    
    # Chạy command
    try:
        subprocess.run(cmd, check=True)
        print("SUCCESS: Completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Error: {e}")
    except KeyboardInterrupt:
        print("⏹️ Interrupted by user")

if __name__ == "__main__":
    main() 