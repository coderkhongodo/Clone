#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PhoBERT Pipeline V2 - Complete training pipeline với dữ liệu từ simple_dataset
Updated để sử dụng:
- Input: simple_dataset/ 
- Processed data: data-bert-v2/
- Model output: checkpoints/phobert-{base|large}-v2/
"""

import subprocess
import sys
import os

def check_prerequisites():
    """Kiểm tra các điều kiện tiên quyết"""
    print("SEARCH: Kiểm tra điều kiện tiên quyết...")
    
    # Kiểm tra dữ liệu nguồn
    if not os.path.exists('simple_dataset'):
        print("ERROR: Không tìm thấy thư mục simple_dataset!")
        print("TIP: Đảm bảo có thư mục simple_dataset với train/val/test")
        return False
    
    # Kiểm tra cấu trúc dữ liệu
    required_dirs = ['train', 'val', 'test']
    for split in required_dirs:
        split_dir = os.path.join('simple_dataset', split)
        if not os.path.exists(split_dir):
            print(f"ERROR: Không tìm thấy {split_dir}!")
            return False
        
        csv_file = os.path.join(split_dir, f'{split}.csv')
        if not os.path.exists(csv_file):
            print(f"ERROR: Không tìm thấy {csv_file}!")
            return False
    
    print("SUCCESS: Dữ liệu nguồn OK")
    return True

def run_preprocessing():
    """Chạy preprocessing cho PhoBERT V2"""
    print("\n" + "="*60)
    print("DATA: BƯỚC 1: PREPROCESSING DỮ LIỆU")
    print("="*60)
    
    cmd = ['python', 'preprocess_for_phobert.py']
    
    try:
        print("PROCESS: Đang chạy preprocessing...")
        print(f"STATS: Command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print("SUCCESS: Preprocessing hoàn thành!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Preprocessing thất bại: {e}")
        return False
    except KeyboardInterrupt:
        print("⏹️ Preprocessing bị dừng bởi user")
        return False

def run_training(model_type='base', use_wandb=False, epochs=5, batch_size=16):
    """Chạy training PhoBERT V2"""
    print(f"\n" + "="*60)
    print(f"START: BƯỚC 2: TRAINING PHOBERT-{model_type.upper()}-V2")
    print("="*60)
    
    # Xác định model name và output directory
    if model_type == 'base':
        model_name = 'vinai/phobert-base'
        output_dir = 'checkpoints/phobert-base-v2'
        learning_rate = '8e-6'
        dropout_rate = '0.35'
        freeze_layers = '6'
        grad_accumulation = '4'
        actual_batch_size = batch_size
    else:  # large
        model_name = 'vinai/phobert-large'
        output_dir = 'checkpoints/phobert-large-v2'
        learning_rate = '5e-6'
        dropout_rate = '0.45'
        freeze_layers = '10'
        grad_accumulation = '8'
        actual_batch_size = max(4, batch_size // 2)
    
    cmd = [
        'python', 'train_phobert.py',
        '--model_name', model_name,
        '--data_dir', 'data-bert-v2',
        '--output_dir', output_dir,
        '--epochs', str(epochs),
        '--batch_size', str(actual_batch_size),
        '--learning_rate', learning_rate,
        '--weight_decay', '0.01',
        '--warmup_ratio', '0.2',
        '--scheduler', 'cosine',
        '--loss_type', 'f1',
        '--dropout_rate', dropout_rate,
        '--num_freeze_layers', freeze_layers,
        '--gradient_accumulation_steps', grad_accumulation,
        '--pooling_strategy', 'cls_mean',
        '--use_amp',
        '--use_ema',
        '--ema_decay', '0.9999',
        '--patience', '8'
    ]
    
    if use_wandb:
        cmd.extend(['--use_wandb', '--wandb_project', 'phobert-clickbait-v2'])
    
    try:
        print(f"START: Bắt đầu training {model_type}...")
        print(f"STATS: Config: {epochs} epochs, batch={actual_batch_size}, lr={learning_rate}")
        print(f"📁 Output: {output_dir}")
        print(f"STATS: Command: {' '.join(cmd)}")
        
        subprocess.run(cmd, check=True)
        print(f"SUCCESS: Training {model_type} hoàn thành!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Training {model_type} thất bại: {e}")
        return False
    except KeyboardInterrupt:
        print(f"⏹️ Training {model_type} bị dừng bởi user")
        return False

def run_evaluation(model_type='base'):
    """Chạy evaluation cho model đã train"""
    print(f"\n" + "="*60)
    print(f"TEST: BƯỚC 3: EVALUATION PHOBERT-{model_type.upper()}-V2")
    print("="*60)
    
    # Đường dẫn model và dữ liệu
    if model_type == 'base':
        model_name = 'vinai/phobert-base'
        model_path = 'checkpoints/phobert-base-v2/best_model.pth'
        data_path = 'data-bert-v2/phobert-base-v2/test_processed.pkl'
        output_dir = 'evaluation_results/phobert-base-v2'
    else:
        model_name = 'vinai/phobert-large'
        model_path = 'checkpoints/phobert-large-v2/best_model.pth'
        data_path = 'data-bert-v2/phobert-large-v2/test_processed.pkl'
        output_dir = 'evaluation_results/phobert-large-v2'
    
    # Kiểm tra file model có tồn tại không
    if not os.path.exists(model_path):
        print(f"ERROR: Không tìm thấy model: {model_path}")
        print("TIP: Hãy chạy training trước khi evaluation")
        return False
    
    if not os.path.exists(data_path):
        print(f"ERROR: Không tìm thấy test data: {data_path}")
        print("TIP: Hãy chạy preprocessing trước khi evaluation")
        return False
    
    cmd = [
        'python', 'evaluate_phobert.py',
        '--model_path', model_path,
        '--model_name', model_name,
        '--data_path', data_path,
        '--output_dir', output_dir,
        '--batch_size', '32'
    ]
    
    try:
        print(f"TEST: Đang đánh giá {model_type}...")
        print(f"📁 Model: {model_path}")
        print(f"📁 Data: {data_path}")
        print(f"📁 Output: {output_dir}")
        print(f"STATS: Command: {' '.join(cmd)}")
        
        subprocess.run(cmd, check=True)
        print(f"SUCCESS: Evaluation {model_type} hoàn thành!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Evaluation {model_type} thất bại: {e}")
        return False
    except KeyboardInterrupt:
        print(f"⏹️ Evaluation {model_type} bị dừng bởi user")
        return False

def main():
    print("=" * 80)
    print("MODEL: PHOBERT PIPELINE V2 - COMPLETE TRAINING SYSTEM")
    print("=" * 80)
    print("📁 Input data: simple_dataset/")
    print("📁 Processed data: data-bert-v2/")
    print("📁 Model output: checkpoints/phobert-{base|large}-v2/")
    print("=" * 80)
    
    # Kiểm tra điều kiện tiên quyết
    if not check_prerequisites():
        return
    
    print("\nChọn chế độ chạy:")
    print("1. Full pipeline: Preprocessing + Training + Evaluation (Base)")
    print("2. Full pipeline: Preprocessing + Training + Evaluation (Large)")
    print("3. Full pipeline: Preprocessing + Training cả Base và Large")
    print("4. Chỉ preprocessing")
    print("5. Chỉ training (cần preprocessing trước)")
    print("6. Chỉ evaluation (cần model đã train)")
    print("7. Chạy lại từ đầu (xóa dữ liệu cũ)")
    
    choice = input("\nNhập lựa chọn (1-7): ").strip()
    
    # Cấu hình
    print("\n=== CẤU HÌNH ===")
    try:
        epochs = int(input("Số epochs (default: 5): ") or "5")
        batch_size = int(input("Batch size (default: 16): ") or "16")
    except ValueError:
        print("ERROR: Input không hợp lệ, sử dụng default values")
        epochs, batch_size = 5, 16
    
    use_wandb_input = input("Sử dụng wandb logging? (y/n, default: n): ").strip().lower()
    use_wandb = use_wandb_input in ['y', 'yes', '1']
    
    print(f"\nSTATS: Cấu hình:")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Wandb: {'Yes' if use_wandb else 'No'}")
    
    confirm = input("\nBắt đầu pipeline? (y/n): ").strip().lower()
    if confirm not in ['y', 'yes', '1']:
        print("ERROR: Pipeline cancelled")
        return
    
    success = True
    
    # Xử lý theo lựa chọn
    if choice == '7':
        # Xóa dữ liệu cũ
        print("\n🧹 Xóa dữ liệu cũ...")
        if os.path.exists('data-bert-v2'):
            import shutil
            shutil.rmtree('data-bert-v2')
            print("SUCCESS: Đã xóa data-bert-v2")
        choice = '1'  # Chuyển về full pipeline
    
    if choice in ['1', '2', '3', '4']:
        # Chạy preprocessing
        success = run_preprocessing()
        if not success:
            return
    
    if choice in ['1', '2', '3', '5']:
        # Chạy training
        if choice == '1' or choice == '5':
            # Base model
            success = run_training('base', use_wandb, epochs, batch_size)
        elif choice == '2':
            # Large model
            success = run_training('large', use_wandb, epochs, batch_size)
        elif choice == '3':
            # Cả hai models
            success = run_training('base', use_wandb, epochs, batch_size)
            if success:
                success = run_training('large', use_wandb, epochs, batch_size)
        
        if not success:
            return
    
    if choice in ['1', '2', '3', '6']:
        # Chạy evaluation
        if choice == '1' or choice == '6':
            # Base model
            success = run_evaluation('base')
        elif choice == '2':
            # Large model
            success = run_evaluation('large')
        elif choice == '3':
            # Cả hai models
            success = run_evaluation('base')
            if success:
                success = run_evaluation('large')
    
    if success:
        print("\n" + "="*80)
        print("COMPLETE: PIPELINE HOÀN THÀNH THÀNH CÔNG!")
        print("="*80)
        print("📁 Kết quả:")
        if os.path.exists('checkpoints/phobert-base-v2'):
            print("   SUCCESS: PhoBERT-Base-V2: checkpoints/phobert-base-v2/")
        if os.path.exists('checkpoints/phobert-large-v2'):
            print("   SUCCESS: PhoBERT-Large-V2: checkpoints/phobert-large-v2/")
        if os.path.exists('evaluation_results'):
            print("   SUCCESS: Evaluation results: evaluation_results/")
        print("\nTIP: Để test model:")
        print("   python evaluate_phobert.py --model_path checkpoints/phobert-base-v2/best_model.pth --model_name vinai/phobert-base --data_path data-bert-v2/phobert-base-v2/test_processed.pkl")
    else:
        print("\nERROR: Pipeline không hoàn thành do lỗi")

if __name__ == "__main__":
    main() 