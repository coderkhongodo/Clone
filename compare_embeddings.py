import subprocess
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import argparse
import os

def run_training(embedding_type, data_path, epochs=10, embedding_dim=100, hidden_dim=128):
    """Chạy training cho một loại embedding"""
    print(f"\n{'='*50}")
    print(f"Training LSTM with {embedding_type.upper()} embedding")
    print(f"{'='*50}")
    
    cmd = [
        'python', 'train_lstm_embedding.py',
        '--data_path', data_path,
        '--embedding_type', embedding_type,
        '--epochs', str(epochs),
        '--embedding_dim', str(embedding_dim),
        '--hidden_dim', str(hidden_dim)
    ]
    
    # Chạy training
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error training {embedding_type}: {result.stderr}")
        return None
    
    # Load kết quả
    try:
        with open(f'results_{embedding_type}.pkl', 'rb') as f:
            results = pickle.load(f)
        return results
    except:
        print(f"Could not load results for {embedding_type}")
        return None

def plot_comparison(results_list):
    """Vẽ biểu đồ so sánh kết quả"""
    if not results_list:
        print("No results to plot")
        return
    
    # Chuẩn bị dữ liệu
    embedding_types = []
    test_accuracies = []
    val_accuracies = []
    
    for result in results_list:
        if result is not None:
            embedding_types.append(result['embedding_type'].upper())
            test_accuracies.append(result['test_accuracy'])
            val_accuracies.append(result['best_val_accuracy'])
    
    # Tạo DataFrame
    df = pd.DataFrame({
        'Embedding': embedding_types,
        'Test Accuracy': test_accuracies,
        'Best Val Accuracy': val_accuracies
    })
    
    print("\nComparison Results:")
    print(df.to_string(index=False))
    
    # Vẽ biểu đồ
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Test Accuracy
    bars1 = ax1.bar(embedding_types, test_accuracies, color=['skyblue', 'lightgreen', 'salmon'])
    ax1.set_title('Test Accuracy Comparison')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    
    # Thêm giá trị trên cột
    for bar, acc in zip(bars1, test_accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom')
    
    # Validation Accuracy
    bars2 = ax2.bar(embedding_types, val_accuracies, color=['skyblue', 'lightgreen', 'salmon'])
    ax2.set_title('Best Validation Accuracy Comparison')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1)
    
    # Thêm giá trị trên cột
    for bar, acc in zip(bars2, val_accuracies):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('embedding_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Lưu bảng kết quả
    df.to_csv('embedding_comparison_results.csv', index=False)
    print("\nResults saved to embedding_comparison_results.csv")
    print("Plot saved to embedding_comparison.png")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Compare different embeddings for LSTM')
    parser.add_argument('--data_path', type=str, default='final_dataset.csv', 
                       help='Path to the dataset CSV file')
    parser.add_argument('--epochs', type=int, default=10, 
                       help='Number of epochs for training')
    parser.add_argument('--embedding_dim', type=int, default=100, 
                       help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=128, 
                       help='LSTM hidden dimension')
    parser.add_argument('--embeddings', nargs='+', 
                       choices=['word2vec', 'glove', 'fasttext'],
                       default=['word2vec', 'glove', 'fasttext'],
                       help='List of embeddings to compare')
    
    args = parser.parse_args()
    
    print("Starting embedding comparison experiment...")
    print(f"Embeddings to compare: {args.embeddings}")
    print(f"Configuration: epochs={args.epochs}, embedding_dim={args.embedding_dim}, hidden_dim={args.hidden_dim}")
    
    # Kiểm tra file dữ liệu
    if not os.path.exists(args.data_path):
        print(f"Warning: Data file {args.data_path} not found. Will use sample data.")
    
    # Chạy training cho từng embedding
    results_list = []
    for embedding_type in args.embeddings:
        result = run_training(
            embedding_type=embedding_type,
            data_path=args.data_path,
            epochs=args.epochs,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim
        )
        results_list.append(result)
    
    # So sánh kết quả
    comparison_df = plot_comparison(results_list)
    
    # Tìm embedding tốt nhất
    if comparison_df is not None and len(comparison_df) > 0:
        best_test = comparison_df.loc[comparison_df['Test Accuracy'].idxmax()]
        best_val = comparison_df.loc[comparison_df['Best Val Accuracy'].idxmax()]
        
        print(f"\nBest Test Accuracy: {best_test['Embedding']} ({best_test['Test Accuracy']:.4f})")
        print(f"Best Validation Accuracy: {best_val['Embedding']} ({best_val['Best Val Accuracy']:.4f})")
        
        # Lưu tổng kết
        summary = {
            'best_test_embedding': best_test['Embedding'],
            'best_test_accuracy': best_test['Test Accuracy'],
            'best_val_embedding': best_val['Embedding'],
            'best_val_accuracy': best_val['Best Val Accuracy'],
            'all_results': results_list
        }
        
        with open('embedding_comparison_summary.pkl', 'wb') as f:
            pickle.dump(summary, f)
        
        print("Summary saved to embedding_comparison_summary.pkl")

if __name__ == "__main__":
    main() 