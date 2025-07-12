#!/usr/bin/env python3

import os
import subprocess
import sys

def check_requirements():
    """Kiểm tra các thư viện cần thiết"""
    required_packages = [
        'torch', 'pandas', 'numpy', 'scikit-learn', 
        'tqdm', 'matplotlib'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {missing_packages}")
        print("Install them with: pip install " + " ".join(missing_packages))
        return False
    return True

def check_data_path():
    """Kiểm tra thư mục dữ liệu"""
    data_path = '/home/huflit/NCKH/simple_dataset'
    if not os.path.exists(data_path):
        print(f"Data directory not found: {data_path}")
        return False
    
    required_files = [
        'train/train.csv',
        'val/val.csv', 
        'test/test.csv'
    ]
    
    for file_path in required_files:
        full_path = os.path.join(data_path, file_path)
        if not os.path.exists(full_path):
            print(f"Required file not found: {full_path}")
            return False
    
    return True

def run_demo():
    """Chạy demo với dữ liệu từ simple_dataset"""
    print("="*60)
    print("DEMO: LSTM with Different Embeddings (No Gensim)")
    print("="*60)
    
    if not check_requirements():
        print("Please install missing packages first.")
        return
    
    if not check_data_path():
        print("Please check data directory and files.")
        return
    
    print("\n1. Testing Word2Vec embedding...")
    cmd1 = [
        sys.executable, 'train_lstm_no_gensim.py',
        '--embedding_type', 'word2vec',
        '--epochs', '3',
        '--embedding_epochs', '2',
        '--batch_size', '16'
    ]
    
    try:
        result = subprocess.run(cmd1, check=True)
        print("✓ Word2Vec test completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"✗ Word2Vec test failed: {e}")
        return
    
    print("\n2. Testing FastText embedding...")
    cmd2 = [
        sys.executable, 'train_lstm_no_gensim.py',
        '--embedding_type', 'fasttext',
        '--epochs', '3',
        '--embedding_epochs', '2',
        '--batch_size', '16'
    ]
    
    try:
        result = subprocess.run(cmd2, check=True)
        print("✓ FastText test completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"✗ FastText test failed: {e}")
        return
    
    print("\n3. Testing GloVe embedding...")
    cmd3 = [
        sys.executable, 'train_lstm_no_gensim.py',
        '--embedding_type', 'glove',
        '--epochs', '3',
        '--embedding_epochs', '2',
        '--batch_size', '16'
    ]
    
    try:
        result = subprocess.run(cmd3, check=True)
        print("✓ GloVe test completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"✗ GloVe test failed: {e}")
        return
    
    print("\n4. Running comparison...")
    cmd4 = [
        sys.executable, 'compare_embeddings_no_gensim.py',
        '--epochs', '3',
        '--embedding_epochs', '2',
        '--embeddings', 'word2vec', 'fasttext', 'glove'
    ]
    
    try:
        result = subprocess.run(cmd4, check=True)
        print("✓ Comparison completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"✗ Comparison failed: {e}")
        return
    
    print("\n" + "="*60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nFiles created:")
    files_to_check = [
        'best_model_no_gensim.pth',
        'results_word2vec_no_gensim.pkl',
        'results_fasttext_no_gensim.pkl', 
        'results_glove_no_gensim.pkl',
        'embedding_comparison_no_gensim.png',
        'embedding_comparison_results_no_gensim.csv',
        'embedding_comparison_summary_no_gensim.pkl'
    ]
    
    for file in files_to_check:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} (not found)")

def print_usage():
    """In hướng dẫn sử dụng"""
    print("\nCách sử dụng scripts (NO GENSIM):")
    print("="*50)
    print("\n1. Huấn luyện với một loại embedding:")
    print("   python train_lstm_no_gensim.py --embedding_type word2vec --epochs 20")
    print("   python train_lstm_no_gensim.py --embedding_type fasttext --epochs 20")
    print("   python train_lstm_no_gensim.py --embedding_type glove --epochs 20")
    
    print("\n2. So sánh tất cả embedding:")
    print("   python compare_embeddings_no_gensim.py --epochs 10")
    
    print("\n3. Sử dụng với thư mục dữ liệu khác:")
    print("   python train_lstm_no_gensim.py --data_path /path/to/dataset --embedding_type word2vec")
    
    print("\n4. Sử dụng BiLSTM:")
    print("   python train_lstm_no_gensim.py --embedding_type word2vec --bidirectional")
    
    print("\n5. Tùy chỉnh tham số:")
    print("   python train_lstm_no_gensim.py --embedding_dim 200 --hidden_dim 256 --epochs 50 --embedding_epochs 10")
    
    print("\nTham số chính:")
    print("  --data_path: Đường dẫn thư mục simple_dataset (mặc định: /home/huflit/NCKH/simple_dataset)")
    print("  --embedding_type: word2vec, glove, fasttext")
    print("  --embedding_dim: Chiều embedding (mặc định: 100)")
    print("  --hidden_dim: Chiều LSTM hidden (mặc định: 128)")
    print("  --epochs: Số epoch cho LSTM (mặc định: 20)")
    print("  --embedding_epochs: Số epoch cho embedding (mặc định: 5)")
    print("  --batch_size: Batch size (mặc định: 32)")
    print("  --bidirectional: Sử dụng BiLSTM")
    
    print("\nĐặc điểm:")
    print("  ✓ Không cần gensim")
    print("  ✓ Implement embedding bằng PyTorch thuần túy")
    print("  ✓ Sử dụng dữ liệu từ simple_dataset") 
    print("  ✓ Hỗ trợ Word2Vec, FastText, GloVe")
    print("  ✓ So sánh hiệu suất các embedding")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--help':
        print_usage()
    else:
        run_demo()
        print_usage() 