#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import pickle
import csv
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm

# Import PhoBERT và VnCoreNLP
try:
    from transformers import AutoTokenizer
    from vncorenlp import VnCoreNLP
    from sklearn.preprocessing import LabelEncoder
    import torch
except ImportError as e:
    print(f"ERROR: Lỗi import: {e}")
    print("Vui lòng cài đặt các thư viện cần thiết:")
    print("pip install -r requirements.txt")
    exit(1)

class PhoBERTPreprocessor:
    def __init__(self, model_name="vinai/phobert-base", max_length=256, vncorenlp_path=None):
        """
        Khởi tạo preprocessor cho PhoBERT
        
        Args:
            model_name: "vinai/phobert-base" hoặc "vinai/phobert-large"
            max_length: Độ dài tối đa của sequence
            vncorenlp_path: Đường dẫn đến VnCoreNLP jar file
        """
        self.model_name = model_name
        self.max_length = max_length
        
        print(f"PROCESS: Đang khởi tạo {model_name}...")
        
        # Khởi tạo tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        
        # Khởi tạo VnCoreNLP cho word segmentation
        self.setup_vncorenlp(vncorenlp_path)
        
        # Label encoder
        self.label_encoder = LabelEncoder()
        
        print("SUCCESS: Khởi tạo hoàn thành!")
    
    def setup_vncorenlp(self, vncorenlp_path):
        """Setup VnCoreNLP cho word segmentation"""
        try:
            if vncorenlp_path and os.path.exists(vncorenlp_path):
                self.segmenter = VnCoreNLP(vncorenlp_path, annotators="wseg", max_heap_size='-Xmx500m')
            else:
                # Tự động download VnCoreNLP
                print("PROCESS: Đang tải VnCoreNLP...")
                self.segmenter = VnCoreNLP(annotators="wseg", max_heap_size='-Xmx500m')
            print("SUCCESS: VnCoreNLP đã sẵn sàng!")
        except Exception as e:
            print(f"ERROR: Lỗi khởi tạo VnCoreNLP: {e}")
            print("TIP: Có thể cần download VnCoreNLP model:")
            print("vncorenlp download")
            self.segmenter = None
    
    def word_segment(self, text: str) -> str:
        """
        Word segmentation sử dụng VnCoreNLP
        """
        if not self.segmenter:
            return text
        
        try:
            # VnCoreNLP word segmentation
            segmented = self.segmenter.tokenize(text)
            # Nối các từ đã phân đoạn
            result = []
            for sentence in segmented:
                result.extend(sentence)
            return " ".join(result)
        except Exception as e:
            print(f"WARNING: Lỗi word segmentation: {e}")
            return text
    
    def preprocess_text(self, text: str) -> str:
        """
        Tiền xử lý text cho PhoBERT
        """
        # Làm sạch text cơ bản
        text = text.strip()
        if not text:
            return ""
        
        # Word segmentation
        segmented_text = self.word_segment(text)
        
        return segmented_text
    
    def encode_texts(self, texts: List[str]) -> Dict:
        """
        Encode danh sách texts thành format cho PhoBERT
        """
        print(f"PROCESS: Đang encode {len(texts)} texts...")
        
        # Tiền xử lý texts
        processed_texts = []
        for text in tqdm(texts, desc="Word segmentation"):
            processed_text = self.preprocess_text(text)
            processed_texts.append(processed_text)
        
        # Tokenize với PhoBERT tokenizer
        print("PROCESS: Đang tokenize với PhoBERT...")
        encoded = self.tokenizer(
            processed_texts,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'processed_texts': processed_texts
        }
    
    def encode_labels(self, labels: List[str]) -> Dict:
        """
        Encode labels
        """
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        return {
            'labels': torch.tensor(encoded_labels, dtype=torch.long),
            'label_mapping': {label: idx for idx, label in enumerate(self.label_encoder.classes_)},
            'num_classes': len(self.label_encoder.classes_)
        }
    
    def load_data(self, file_path: str) -> Tuple[List[str], List[str]]:
        """
        Load dữ liệu từ CSV hoặc JSONL
        """
        texts = []
        labels = []
        
        if file_path.endswith('.csv'):
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    texts.append(row['title'])
                    labels.append(row['label'])
        elif file_path.endswith('.jsonl'):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    texts.append(data['title'])
                    labels.append(data['label'])
        
        return texts, labels
    
    def save_processed_data(self, data: Dict, output_path: str):
        """
        Lưu dữ liệu đã xử lý
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"SUCCESS: Đã lưu: {output_path}")
    
    def process_dataset(self, input_dir: str, output_dir: str):
        """
        Xử lý toàn bộ dataset
        """
        print(f"=== TIỀN XỬ LÝ DỮ LIỆU CHO {self.model_name.upper()} ===")
        print(f"Max length: {self.max_length}")
        print()
        
        # Tạo thư mục output
        os.makedirs(output_dir, exist_ok=True)
        
        splits = ['train', 'val', 'test']
        all_labels = []
        
        # Thu thập tất cả labels để fit label encoder
        print("PROCESS: Đang thu thập labels...")
        for split in splits:
            csv_path = os.path.join(input_dir, split, f'{split}.csv')
            if os.path.exists(csv_path):
                _, labels = self.load_data(csv_path)
                all_labels.extend(labels)
        
        # Fit label encoder
        self.label_encoder.fit(all_labels)
        label_info = {
            'label_mapping': {label: idx for idx, label in enumerate(self.label_encoder.classes_)},
            'num_classes': len(self.label_encoder.classes_)
        }
        
        print(f"STATS: Label mapping: {label_info['label_mapping']}")
        print()
        
        # Xử lý từng split
        for split in splits:
            print(f"PROCESS: Đang xử lý {split} set...")
            
            csv_path = os.path.join(input_dir, split, f'{split}.csv')
            if not os.path.exists(csv_path):
                print(f"WARNING: Không tìm thấy {csv_path}")
                continue
            
            # Load dữ liệu
            texts, labels = self.load_data(csv_path)
            print(f"STATS: {split}: {len(texts)} mẫu")
            
            # Encode texts
            encoded_texts = self.encode_texts(texts)
            
            # Encode labels
            encoded_labels = self.label_encoder.transform(labels)
            
            # Tạo dataset
            dataset = {
                'input_ids': encoded_texts['input_ids'],
                'attention_mask': encoded_texts['attention_mask'],
                'labels': torch.tensor(encoded_labels, dtype=torch.long),
                'original_texts': texts,
                'processed_texts': encoded_texts['processed_texts'],
                'original_labels': labels,
                'label_mapping': label_info['label_mapping'],
                'num_classes': label_info['num_classes']
            }
            
            # Lưu dữ liệu
            output_path = os.path.join(output_dir, f'{split}_processed.pkl')
            self.save_processed_data(dataset, output_path)
            
            # Lưu thông tin thống kê
            stats = {
                'split': split,
                'num_samples': len(texts),
                'max_length': self.max_length,
                'model_name': self.model_name,
                'avg_length': np.mean([len(text.split()) for text in encoded_texts['processed_texts']]),
                'label_distribution': {label: labels.count(label) for label in set(labels)}
            }
            
            stats_path = os.path.join(output_dir, f'{split}_stats.json')
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            
            print(f"SUCCESS: Hoàn thành {split} set")
            print()
        
        # Lưu thông tin chung
        info = {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'tokenizer_vocab_size': self.tokenizer.vocab_size,
            'label_mapping': label_info['label_mapping'],
            'num_classes': label_info['num_classes'],
            'preprocessing_info': {
                'word_segmentation': 'VnCoreNLP',
                'special_tokens': self.tokenizer.special_tokens_map
            }
        }
        
        info_path = os.path.join(output_dir, 'preprocessing_info.json')
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        
        print("SUCCESS: Hoàn thành tiền xử lý dữ liệu!")
        print(f"📁 Dữ liệu đã lưu tại: {output_dir}")

def main():
    # Cấu hình - SỬA ĐỔI ĐƯỜNG DẪN
    INPUT_DIR = "simple_dataset"  # Thư mục chứa dữ liệu gốc MỚI
    
    # Xử lý cho PhoBERT-base-v2
    print("=== PHOBERT-BASE-V2 ===")
    processor_base = PhoBERTPreprocessor(
        model_name="vinai/phobert-base",
        max_length=256
    )
    processor_base.process_dataset(INPUT_DIR, "data-bert-v2/phobert-base-v2")
    
    print("\n" + "="*50 + "\n")
    
    # Xử lý cho PhoBERT-large-v2
    print("=== PHOBERT-LARGE-V2 ===")
    processor_large = PhoBERTPreprocessor(
        model_name="vinai/phobert-large", 
        max_length=256
    )
    processor_large.process_dataset(INPUT_DIR, "data-bert-v2/phobert-large-v2")

if __name__ == "__main__":
    main() 