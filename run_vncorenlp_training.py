#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Launcher Script cho PhoBERT Training với VnCoreNLP Data
Script này giúp chạy training với dữ liệu đã được xử lý bằng VnCoreNLP
"""

import subprocess
import sys
import os
from datetime import datetime

def run_training(model_type='base', epochs=15, batch_size=32, use_wandb=False):
    """
    Chạy training PhoBERT với VnCoreNLP data
    
    Args:
        model_type: 'base' hoặc 'large'
        epochs: Số epochs
        batch_size: Batch size
        use_wandb: Có sử dụng wandb logging không
    """
    # Cấu hình model
    if model_type == 'base':
        model_name = 'vinai/phobert-base'
        output_dir = 'checkpoints-vncorenlp/phobert-base'
    else:
        model_name = 'vinai/phobert-large'
        output_dir = 'checkpoints-vncorenlp/phobert-large'
    
    print(f"🚀 TRAINING PHOBERT-{model_type.upper()} VỚI VNCORENLP DATA")
    print("=" * 60)
    print(f"📂 Model: {model_name}")
    print(f"💾 Output: {output_dir}")
    print(f"🔧 Data: VnCoreNLP preprocessed")
    print(f"📅 Epochs: {epochs}")
    print(f"📦 Batch size: {batch_size}")
    print("=" * 60)
    
    # Kiểm tra dữ liệu có tồn tại không
    data_subdir = f'phobert-{model_type}'
    data_path = os.path.join('data-vncorenlp-v2', data_subdir)
    train_path = os.path.join(data_path, 'train_processed.pkl')
    
    if not os.path.exists(train_path):
        print(f"❌ Không tìm thấy dữ liệu: {train_path}")
        print("💡 Hãy chạy apply_vncorenlp_to_phobert.py trước")
        return False
    
    # Tạo command
    cmd = [
        'python', 'train_phobert_vncorenlp.py',
        '--model_name', model_name,
        '--data_dir', 'data-vncorenlp-v2',
        '--output_dir', output_dir,
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--learning_rate', '1e-5',
        '--weight_decay', '0.01',
        '--warmup_ratio', '0.15',
        '--scheduler', 'cosine',
        '--dropout_rate', '0.3',
        '--num_freeze_layers', '6',
        '--loss_type', 'f1',
        '--pooling_strategy', 'cls_mean',
        '--use_amp',
        '--use_ema',
        '--ema_decay', '0.9999',
        '--patience', '5',
        '--balance_data',
        '--balance_strategy', 'oversample',
        '--gradient_clipping', '1.0',
        '--seed', '42'
    ]
    
    # Thêm wandb nếu được yêu cầu
    if use_wandb:
        cmd.extend(['--use_wandb', '--wandb_project', 'phobert-vncorenlp'])
    
    # Tạo run name với timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"phobert-{model_type}_vncorenlp_{timestamp}"
    cmd.extend(['--run_name', run_name])
    
    print(f"📝 Command: {' '.join(cmd)}")
    print()
    
    try:
        # Chạy training
        result = subprocess.run(cmd, check=True)
        print(f"\n✅ Training PhoBERT-{model_type} hoàn thành thành công!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Lỗi training PhoBERT-{model_type}: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⏹️ Training PhoBERT-{model_type} bị hủy bởi user")
        return False

def main():
    print("🇻🇳 PHOBERT TRAINING LAUNCHER - VNCORENLP DATA")
    print("=" * 60)
    
    # Kiểm tra dữ liệu đã có chưa
    if not os.path.exists('data-vncorenlp-v2'):
        print("❌ Không tìm thấy dữ liệu VnCoreNLP!")
        print("💡 Hãy chạy: python apply_vncorenlp_to_phobert.py")
        return
    
    print("Chọn model để train:")
    print("1. PhoBERT-base (nhẹ hơn, nhanh hơn)")
    print("2. PhoBERT-large (chính xác hơn, chậm hơn)")
    print("3. Cả hai (train base trước, sau đó large)")
    
    choice = input("\nNhập lựa chọn (1/2/3): ").strip()
    
    # Cấu hình training
    print("\n=== CẤU HÌNH TRAINING ===")
    
    try:
        epochs = int(input("Số epochs (default: 15): ") or "15")
        batch_size = int(input("Batch size (default: 32): ") or "32")
    except ValueError:
        print("❌ Input không hợp lệ, sử dụng default values")
        epochs, batch_size = 15, 32
    
    use_wandb_input = input("Sử dụng wandb logging? (y/n, default: n): ").strip().lower()
    use_wandb = use_wandb_input in ['y', 'yes', '1']
    
    print(f"\n📊 Cấu hình:")
    print(f"   🕒 Epochs: {epochs}")
    print(f"   📦 Batch size: {batch_size}")
    print(f"   📈 Wandb: {'Yes' if use_wandb else 'No'}")
    print(f"   🔧 Data source: VnCoreNLP preprocessed")
    
    confirm = input("\nBắt đầu training? (y/n): ").strip().lower()
    if confirm not in ['y', 'yes', '1']:
        print("❌ Training cancelled")
        return
    
    # Bắt đầu training
    if choice == '1':
        run_training('base', epochs, batch_size, use_wandb)
    elif choice == '2':
        run_training('large', epochs, batch_size, use_wandb)
    elif choice == '3':
        print("🔄 Training PhoBERT-base first...")
        success = run_training('base', epochs, batch_size, use_wandb)
        if success:
            print("\n🔄 Training PhoBERT-large...")
            run_training('large', epochs, batch_size, use_wandb)
    else:
        print("❌ Lựa chọn không hợp lệ")

if __name__ == "__main__":
    main() 