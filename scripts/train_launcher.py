#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import sys
import os

def run_training(model_type='base', use_wandb=False, epochs=5, batch_size=16):
    """
    Chạy training PhoBERT với cấu hình được định sẵn
    
    Args:
        model_type: 'base' hoặc 'large'
        use_wandb: Có sử dụng wandb logging không
        epochs: Số epochs
        batch_size: Batch size
    """
    
    # Xác định model name
    if model_type == 'base':
        model_name = 'vinai/phobert-base'
        output_dir = 'checkpoints/phobert-base-v2'
    elif model_type == 'large':
        model_name = 'vinai/phobert-large'
        output_dir = 'checkpoints/phobert-large-v2'
    else:
        print("ERROR: model_type phải là 'base' hoặc 'large'")
        return
    
    # Model-specific optimized hyperparameters for maximum F1
    if model_type == 'base':
        learning_rate = '8e-6'
        dropout_rate = '0.35'
        freeze_layers = '6'
        grad_accumulation = '4'
        actual_batch_size = batch_size
    else:  # large
        learning_rate = '5e-6'  # Lower LR for large model
        dropout_rate = '0.45'   # Higher dropout for large model
        freeze_layers = '10'    # Freeze more layers
        grad_accumulation = '8' # Larger accumulation
        actual_batch_size = max(4, batch_size // 2)  # Smaller batch for memory
    
    # Tạo command với advanced hyperparameters for higher F1
    cmd = [
        'python', 'train_phobert.py',
        '--model_name', model_name,
        '--output_dir', output_dir,
        '--epochs', str(epochs),
        '--batch_size', str(actual_batch_size),
        '--learning_rate', learning_rate,
        '--weight_decay', '0.01',
        '--warmup_ratio', '0.2',     # More warmup
        '--scheduler', 'cosine',     # Cosine scheduler
        '--loss_type', 'f1',         # Direct F1 optimization
        '--dropout_rate', dropout_rate,
        '--num_freeze_layers', freeze_layers,
        '--gradient_accumulation_steps', grad_accumulation,
        '--pooling_strategy', 'cls_mean',  # Better pooling
        '--use_amp',  # Mixed precision
        '--use_ema',  # Exponential moving average
        '--ema_decay', '0.9999',     # EMA decay
        '--patience', '8'  # More patience for better convergence
    ]
    
    if use_wandb:
        cmd.extend(['--use_wandb', '--wandb_project', 'phobert-clickbait'])
    
    print(f"START: Starting training PhoBERT-{model_type} with F1-optimized settings")
    print(f"STATS: Config: {epochs} epochs, batch={actual_batch_size}, lr={learning_rate}")
    print(f"TARGET: Advanced: dropout={dropout_rate}, freeze={freeze_layers}, grad_accum={grad_accumulation}")
    print(f"CONFIG: Features: EMA, F1-loss, cls_mean pooling, cosine scheduler")
    print(f"📁 Output: {output_dir}")
    print(f"METRICS: Wandb: {'Yes' if use_wandb else 'No'}")
    print()
    
    # Chạy training
    try:
        subprocess.run(cmd, check=True)
        print("SUCCESS: Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Training failed with error: {e}")
        return False
    except KeyboardInterrupt:
        print("⏹️ Training interrupted by user")
        return False
    
    return True

def main():
    print("=== PHOBERT TRAINING LAUNCHER ===")
    print()
    
    # Kiểm tra dữ liệu đã preprocessing chưa
    if not os.path.exists('data-bert-v2'):
        print("ERROR: Không tìm thấy dữ liệu đã preprocessing!")
        print("TIP: Chạy: python preprocess_for_phobert.py")
        return
    
    print("Chọn loại model:")
    print("1. PhoBERT-base (nhẹ hơn, nhanh hơn)")
    print("2. PhoBERT-large (chính xác hơn, chậm hơn)")
    print("3. Cả hai (train base trước, sau đó large)")
    
    choice = input("\nNhập lựa chọn (1/2/3): ").strip()
    
    # Cấu hình training
    print("\n=== CẤU HÌNH TRAINING ===")
    
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
    
    confirm = input("\nBắt đầu training? (y/n): ").strip().lower()
    if confirm not in ['y', 'yes', '1']:
        print("ERROR: Training cancelled")
        return
    
    # Bắt đầu training
    if choice == '1':
        run_training('base', use_wandb, epochs, batch_size)
    elif choice == '2':
        run_training('large', use_wandb, epochs, batch_size)
    elif choice == '3':
        print("PROCESS: Training PhoBERT-base first...")
        success = run_training('base', use_wandb, epochs, batch_size)
        if success:
            print("\nPROCESS: Training PhoBERT-large...")
            run_training('large', use_wandb, epochs, batch_size)
    else:
        print("ERROR: Lựa chọn không hợp lệ")

if __name__ == "__main__":
    main() 