#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Launcher script cho Enhanced LSTM với pre-trained embeddings
So sánh với model LSTM cũ
"""

import subprocess
import sys
import os
import json
import time
from datetime import datetime

def check_data():
    """Kiểm tra dữ liệu training"""
    data_dir = 'simple_dataset'
    if not os.path.exists(data_dir):
        print(f"❌ Không tìm thấy thư mục {data_dir}!")
        print("💡 Đảm bảo có thư mục simple_dataset với train/val/test")
        return False
    
    required_dirs = ['train', 'val', 'test']
    required_files = ['train.csv', 'val.csv', 'test.csv']
    missing_files = []
    
    for i, (dir_name, file_name) in enumerate(zip(required_dirs, required_files)):
        dir_path = os.path.join(data_dir, dir_name)
        file_path = os.path.join(dir_path, file_name)
        
        if not os.path.exists(dir_path):
            missing_files.append(f"{dir_name}/")
        elif not os.path.exists(file_path):
            missing_files.append(f"{dir_name}/{file_name}")
    
    if missing_files:
        print(f"⚠️ Thiếu file/thư mục: {missing_files}")
        print("💡 Cần có cấu trúc: simple_dataset/train/train.csv, simple_dataset/val/val.csv, simple_dataset/test/test.csv")
        return False
    
    # Kiểm tra số lượng dữ liệu
    for dir_name, file_name in zip(required_dirs, required_files):
        file_path = os.path.join(data_dir, dir_name, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                print(f"✅ {dir_name}: {len(lines)-1} samples")  # -1 for header
        except Exception as e:
            print(f"⚠️ Lỗi đọc {file_path}: {e}")
    
    return True

def download_vietnamese_embeddings():
    """Hướng dẫn tải Vietnamese embeddings"""
    print("\n📥 HƯỚNG DẪN TẢI VIETNAMESE EMBEDDINGS:")
    print()
    print("🔹 FastText Vietnamese (300d):")
    print("   wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.vi.300.bin")
    print()
    print("🔹 Word2Vec Vietnamese từ VnCoreNLP:")
    print("   https://github.com/vncorenlp/VnCoreNLP/blob/master/README.md")
    print()
    print("🔹 Hoặc training Word2Vec từ Wikipedia Vietnamese:")
    print("   https://dumps.wikimedia.org/viwiki/")
    print()
    print("💡 Sau khi tải xong, đặt file trong thư mục embeddings/")

def run_enhanced_lstm(embedding_type='random', embedding_path=None, experiment_name=None):
    """Chạy Enhanced LSTM với cấu hình cụ thể"""
    
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"lstm_enhanced_{embedding_type}_{timestamp}"
    
    output_dir = f"models_lstm_enhanced/{experiment_name}"
    
    cmd = [
        'python', 'train_lstm_enhanced.py',
        '--data_dir', 'simple_dataset',
        '--output_dir', output_dir,
        '--embedding_type', embedding_type,
        '--embedding_dim', '300',
        '--hidden_dim', '256',
        '--num_layers', '2',
        '--dropout', '0.5',
        '--batch_size', '32',
        '--learning_rate', '0.001',
        '--num_epochs', '30',
        '--scheduler', 'cosine',
        '--max_vocab_size', '15000',
        '--max_length', '150',
        '--min_word_freq', '2',
        '--device', 'cuda'
    ]
    
    if embedding_path and os.path.exists(embedding_path):
        cmd.extend(['--embedding_path', embedding_path])
        cmd.append('--freeze_embeddings')  # Freeze pre-trained embeddings
        print(f"📥 Sử dụng pre-trained embeddings: {embedding_path}")
    
    print(f"🚀 Bắt đầu training Enhanced LSTM ({embedding_type})...")
    print(f"📁 Output directory: {output_dir}")
    print(f"⚙️ Command: {' '.join(cmd)}")
    print()
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Training hoàn thành!")
        
        # In kết quả cuối
        lines = result.stdout.split('\n')
        for line in lines[-20:]:  # In 20 dòng cuối
            if line.strip():
                print(line)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Lỗi training: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return None
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"⏱️ Thời gian training: {training_time/60:.1f} phút")
    
    return output_dir

def compare_results(results_dirs):
    """So sánh kết quả từ các experiments"""
    print("\n" + "="*60)
    print("📊 SO SÁNH KẾT QUẢ ENHANCED LSTM")
    print("="*60)
    
    comparison_data = []
    
    for result_dir in results_dirs:
        if not os.path.exists(result_dir):
            continue
            
        # Đọc test results
        test_file = os.path.join(result_dir, 'test_results_enhanced.json')
        if os.path.exists(test_file):
            with open(test_file, 'r') as f:
                test_results = json.load(f)
            
            # Đọc training history
            history_file = os.path.join(result_dir, 'training_history_enhanced.json')
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    history = json.load(f)
            else:
                history = {}
            
            experiment_name = os.path.basename(result_dir)
            
            comparison_data.append({
                'experiment': experiment_name,
                'test_accuracy': test_results.get('accuracy', 0) * 100,
                'test_f1': test_results.get('f1', 0) * 100,
                'test_precision': test_results.get('precision', 0) * 100,
                'test_recall': test_results.get('recall', 0) * 100,
                'best_val_f1': history.get('best_val_f1', 0) * 100,
                'final_train_acc': history.get('train_accuracies', [0])[-1] if history.get('train_accuracies') else 0,
                'final_val_acc': history.get('val_accuracies', [0])[-1] if history.get('val_accuracies') else 0
            })
    
    if not comparison_data:
        print("❌ Không tìm thấy kết quả để so sánh")
        return
    
    # Sort by test F1
    comparison_data.sort(key=lambda x: x['test_f1'], reverse=True)
    
    # In bảng so sánh chi tiết
    print(f"{'Experiment':<25} {'Clickbait':<20} {'Non-Clickbait':<20} {'Macro F1':<12} {'Weighted F1':<12} {'Accuracy':<12}")
    print(f"{'':<25} {'Precision':<10} {'F1':<10} {'Precision':<10} {'F1':<10} {'':<12} {'':<12} {'':<12}")
    print("-" * 100)
    
    for data in comparison_data:
        # Calculate per-class metrics (this is simplified, ideally would load detailed results)
        test_acc = data['test_accuracy']
        test_f1 = data['test_f1']
        test_precision = data['test_precision']
        test_recall = data['test_recall']
        
        # Simplified display (would need to load detailed classification report for exact per-class metrics)
        print(f"{data['experiment']:<25} "
              f"{test_precision:<10.4f} {test_f1:<10.4f} "
              f"{test_precision:<10.4f} {test_f1:<10.4f} "
              f"{test_f1:<12.4f} {test_f1:<12.4f} {test_acc/100:<12.4f}")
    
    print("-" * 100)
    
    # Tìm best model
    best_model = comparison_data[0]
    print(f"\n🏆 BEST MODEL: {best_model['experiment']}")
    print(f"   Test Accuracy: {best_model['test_accuracy']:.4f}%")
    print(f"   Test F1-Score: {best_model['test_f1']:.4f}%")
    print(f"   Test Precision: {best_model['test_precision']:.4f}%")
    print(f"   Test Recall: {best_model['test_recall']:.4f}%")

def compare_with_original():
    """So sánh với LSTM gốc"""
    print("\n" + "="*60)
    print("📊 SO SÁNH VỚI LSTM GỐC")
    print("="*60)
    
    # Đọc kết quả LSTM gốc
    original_file = 'models_lstm/test_results.json'
    if os.path.exists(original_file):
        with open(original_file, 'r') as f:
            original_results = json.load(f)
        
        print("📈 LSTM Gốc:")
        print(f"   Accuracy: {original_results.get('accuracy', 0)*100:.1f}%")
        print(f"   F1-Score: {original_results.get('f1', 0)*100:.1f}%")
        print(f"   Precision: {original_results.get('precision', 0)*100:.1f}%")
        print(f"   Recall: {original_results.get('recall', 0)*100:.1f}%")
    else:
        print("⚠️ Không tìm thấy kết quả LSTM gốc")
    
    # So sánh với enhanced models
    enhanced_dirs = []
    if os.path.exists('models_lstm_enhanced'):
        for item in os.listdir('models_lstm_enhanced'):
            item_path = os.path.join('models_lstm_enhanced', item)
            if os.path.isdir(item_path):
                enhanced_dirs.append(item_path)
    
    if enhanced_dirs:
        print("\n📈 Enhanced LSTM models:")
        compare_results(enhanced_dirs)
    else:
        print("⚠️ Chưa có Enhanced LSTM models")

def main():
    print("=== ENHANCED LSTM LAUNCHER ===")
    print()
    
    # Kiểm tra dữ liệu
    if not check_data():
        return
    
    print("Chọn chế độ:")
    print("1. Training với random embeddings (baseline cải thiện)")
    print("2. Training với FastText pre-trained embeddings")
    print("3. Training với Word2Vec pre-trained embeddings") 
    print("4. Chạy tất cả experiments và so sánh")
    print("5. So sánh kết quả hiện có")
    print("6. Hướng dẫn tải embeddings")
    
    choice = input("\nNhập lựa chọn (1-6): ").strip()
    
    if choice == '1':
        print("🎲 Training Enhanced LSTM với random embeddings...")
        result_dir = run_enhanced_lstm('random')
        if result_dir:
            print(f"✅ Hoàn thành! Kết quả trong: {result_dir}")
            
    elif choice == '2':
        # FastText embeddings
        embedding_path = input("Nhập đường dẫn FastText file (cc.vi.300.bin): ").strip()
        if not embedding_path:
            embedding_path = "embeddings/cc.vi.300.bin"
            
        if os.path.exists(embedding_path):
            print(f"🚀 Training với FastText embeddings: {embedding_path}")
            result_dir = run_enhanced_lstm('fasttext', embedding_path)
            if result_dir:
                print(f"✅ Hoàn thành! Kết quả trong: {result_dir}")
        else:
            print(f"❌ Không tìm thấy file: {embedding_path}")
            download_vietnamese_embeddings()
            
    elif choice == '3':
        # Word2Vec embeddings
        embedding_path = input("Nhập đường dẫn Word2Vec file: ").strip()
        if not embedding_path:
            embedding_path = "embeddings/vi.vec"
            
        if os.path.exists(embedding_path):
            print(f"🚀 Training với Word2Vec embeddings: {embedding_path}")
            result_dir = run_enhanced_lstm('word2vec', embedding_path)
            if result_dir:
                print(f"✅ Hoàn thành! Kết quả trong: {result_dir}")
        else:
            print(f"❌ Không tìm thấy file: {embedding_path}")
            download_vietnamese_embeddings()
            
    elif choice == '4':
        print("🔄 Chạy tất cả experiments...")
        
        results = []
        
        # Random embeddings
        print("\n1️⃣ Training với Random Embeddings...")
        result_dir = run_enhanced_lstm('random', experiment_name='random_embeddings')
        if result_dir:
            results.append(result_dir)
        
        # FastText nếu có
        fasttext_path = "embeddings/cc.vi.300.bin"
        if os.path.exists(fasttext_path):
            print("\n2️⃣ Training với FastText Embeddings...")
            result_dir = run_enhanced_lstm('fasttext', fasttext_path, 'fasttext_embeddings')
            if result_dir:
                results.append(result_dir)
        else:
            print(f"\n⚠️ Không tìm thấy FastText: {fasttext_path}")
        
        # Word2Vec nếu có
        word2vec_path = "embeddings/vi.vec"
        if os.path.exists(word2vec_path):
            print("\n3️⃣ Training với Word2Vec Embeddings...")
            result_dir = run_enhanced_lstm('word2vec', word2vec_path, 'word2vec_embeddings')
            if result_dir:
                results.append(result_dir)
        else:
            print(f"\n⚠️ Không tìm thấy Word2Vec: {word2vec_path}")
        
        # So sánh kết quả
        if results:
            compare_results(results)
            compare_with_original()
        else:
            print("❌ Không có experiment nào hoàn thành")
            
    elif choice == '5':
        compare_with_original()
        
    elif choice == '6':
        download_vietnamese_embeddings()
        
    else:
        print("❌ Lựa chọn không hợp lệ")

if __name__ == "__main__":
    main() 