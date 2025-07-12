#!/usr/bin/env python3

import os
import subprocess
import sys

def check_requirements():
    """Kiểm tra các thư viện cần thiết"""
    required_packages = [
        'torch', 'pandas', 'numpy', 'scikit-learn', 
        'gensim', 'tqdm', 'matplotlib'
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

def run_demo():
    """Chạy demo với dữ liệu mẫu"""
    print("="*60)
    print("DEMO: LSTM with Different Embeddings")
    print("="*60)
    
    if not check_requirements():
        print("Please install missing packages first.")
        return
    
    print("\n1. Testing Word2Vec embedding...")
    cmd1 = [
        sys.executable, 'train_lstm_embedding.py',
        '--embedding_type', 'word2vec',
        '--epochs', '3',
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
        sys.executable, 'train_lstm_embedding.py',
        '--embedding_type', 'fasttext',
        '--epochs', '3',
        '--batch_size', '16'
    ]
    
    try:
        result = subprocess.run(cmd2, check=True)
        print("✓ FastText test completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"✗ FastText test failed: {e}")
        return
    
    print("\n3. Testing GloVe-like embedding...")
    cmd3 = [
        sys.executable, 'train_lstm_embedding.py',
        '--embedding_type', 'glove',
        '--epochs', '3',
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
        sys.executable, 'compare_embeddings.py',
        '--epochs', '3',
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
        'best_model.pth',
        'results_word2vec.pkl',
        'results_fasttext.pkl', 
        'results_glove.pkl',
        'embedding_comparison.png',
        'embedding_comparison_results.csv',
        'embedding_comparison_summary.pkl'
    ]
    
    for file in files_to_check:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} (not found)")

def print_usage():
    """In hướng dẫn sử dụng"""
    print("\nCách sử dụng:")
    print("="*50)
    print("\n1. Huấn luyện với một loại embedding:")
    print("   python train_lstm_embedding.py --embedding_type word2vec --epochs 20")
    print("   python train_lstm_embedding.py --embedding_type fasttext --epochs 20")
    print("   python train_lstm_embedding.py --embedding_type glove --epochs 20")
    
    print("\n2. So sánh tất cả embedding:")
    print("   python compare_embeddings.py --epochs 10")
    
    print("\n3. Sử dụng với dữ liệu thật:")
    print("   python train_lstm_embedding.py --data_path your_data.csv --embedding_type word2vec")
    
    print("\n4. Sử dụng BiLSTM:")
    print("   python train_lstm_embedding.py --embedding_type word2vec --bidirectional")
    
    print("\n5. Tùy chỉnh tham số:")
    print("   python train_lstm_embedding.py --embedding_dim 200 --hidden_dim 256 --epochs 50")
    
    print("\nTham số:")
    print("  --data_path: Đường dẫn file CSV (mặc định: final_dataset.csv)")
    print("  --embedding_type: word2vec, glove, fasttext")
    print("  --embedding_dim: Chiều embedding (mặc định: 100)")
    print("  --hidden_dim: Chiều LSTM hidden (mặc định: 128)")
    print("  --epochs: Số epoch (mặc định: 20)")
    print("  --batch_size: Batch size (mặc định: 32)")
    print("  --bidirectional: Sử dụng BiLSTM")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--help':
        print_usage()
    else:
        run_demo()
        print_usage() 