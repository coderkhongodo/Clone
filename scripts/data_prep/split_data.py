#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import os
import random
from collections import defaultdict

def create_directories():
    """Tạo các thư mục con trong folder data"""
    base_path = "data"
    subdirs = ["train", "val", "test"]
    
    for subdir in subdirs:
        path = os.path.join(base_path, subdir)
        os.makedirs(path, exist_ok=True)
        print(f"Đã tạo thư mục: {path}")

def load_data_csv(filename):
    """Đọc dữ liệu từ file CSV và chỉ lấy cột title và label"""
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'title': row['title'],
                'label': row['label']
            })
    return data

def load_data_jsonl(filename):
    """Đọc dữ liệu từ file JSONL và chỉ lấy cột title và label"""
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line.strip())
            data.append({
                'title': row['title'],
                'label': row['label']
            })
    return data

def split_by_class(data):
    """Chia dữ liệu theo từng lớp"""
    classes = defaultdict(list)
    for item in data:
        classes[item['label']].append(item)
    return classes

def stratified_split(classes, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Chia dữ liệu theo tỉ lệ và cân bằng giữa các lớp"""
    train_data = []
    val_data = []
    test_data = []
    
    for label, items in classes.items():
        # Xáo trộn dữ liệu
        random.shuffle(items)
        
        total = len(items)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        
        # Chia dữ liệu
        train_items = items[:train_size]
        val_items = items[train_size:train_size + val_size]
        test_items = items[train_size + val_size:]
        
        train_data.extend(train_items)
        val_data.extend(val_items)
        test_data.extend(test_items)
        
        print(f"Lớp {label}:")
        print(f"  Train: {len(train_items)} mẫu")
        print(f"  Val: {len(val_items)} mẫu")
        print(f"  Test: {len(test_items)} mẫu")
    
    return train_data, val_data, test_data

def save_data_csv(data, filename):
    """Lưu dữ liệu vào file CSV"""
    with open(filename, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['title', 'label']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def save_data_jsonl(data, filename):
    """Lưu dữ liệu vào file JSONL"""
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    # Thiết lập seed để có thể tái tạo kết quả
    random.seed(42)
    
    print("=== CHIA DỮ LIỆU CLICKBAIT VIETNAMESE ===")
    print("Tỉ lệ: Train 70% - Val 15% - Test 15%")
    print("Cân bằng giữa các lớp: Có")
    print("Format: CSV và JSONL")
    print()
    
    # Tạo thư mục
    create_directories()
    print()
    
    # Đọc dữ liệu từ CSV
    print("Đang đọc dữ liệu từ CSV...")
    data_csv = load_data_csv('clickbait_dataset_vietnamese.csv')
    print(f"CSV - Tổng số mẫu: {len(data_csv)}")
    
    # Đọc dữ liệu từ JSONL
    print("Đang đọc dữ liệu từ JSONL...")
    data_jsonl = load_data_jsonl('clickbait_dataset_vietnamese.jsonl')
    print(f"JSONL - Tổng số mẫu: {len(data_jsonl)}")
    print()
    
    # Sử dụng dữ liệu từ CSV (hoặc JSONL, cả hai giống nhau)
    data = data_csv
    
    # Chia theo lớp
    classes = split_by_class(data)
    print("Phân bố ban đầu:")
    for label, items in classes.items():
        print(f"  {label}: {len(items)} mẫu")
    print()
    
    # Chia dữ liệu
    print("Đang chia dữ liệu...")
    train_data, val_data, test_data = stratified_split(classes)
    print()
    
    # Xáo trộn lại toàn bộ để trộn các lớp
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    # Lưu dữ liệu CSV
    print("Đang lưu dữ liệu CSV...")
    save_data_csv(train_data, 'data/train/train.csv')
    save_data_csv(val_data, 'data/val/val.csv')
    save_data_csv(test_data, 'data/test/test.csv')
    
    # Lưu dữ liệu JSONL
    print("Đang lưu dữ liệu JSONL...")
    save_data_jsonl(train_data, 'data/train/train.jsonl')
    save_data_jsonl(val_data, 'data/val/val.jsonl')
    save_data_jsonl(test_data, 'data/test/test.jsonl')
    
    print(f"Đã lưu train: {len(train_data)} mẫu (CSV + JSONL)")
    print(f"Đã lưu val: {len(val_data)} mẫu (CSV + JSONL)") 
    print(f"Đã lưu test: {len(test_data)} mẫu (CSV + JSONL)")
    print()
    
    # Kiểm tra phân bố cuối cùng
    print("=== KIỂM TRA PHÂN BỐ CUỐI CÙNG ===")
    
    def check_distribution(data, name):
        classes = defaultdict(int)
        for item in data:
            classes[item['label']] += 1
        print(f"{name}:")
        for label, count in classes.items():
            percentage = (count / len(data)) * 100
            print(f"  {label}: {count} mẫu ({percentage:.1f}%)")
    
    check_distribution(train_data, "Train")
    check_distribution(val_data, "Val")
    check_distribution(test_data, "Test")
    
    print("\nSUCCESS: Hoàn thành chia dữ liệu! (Cả CSV và JSONL)")

if __name__ == "__main__":
    main() 