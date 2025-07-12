#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Áp dụng VnCoreNLP cho PhoBERT Data Preprocessing
Script này sử dụng VnCoreNLP package để tiền xử lý dữ liệu cho PhoBERT-base và PhoBERT-large
"""

import os
import json
import pickle
import csv
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm

try:
    from transformers import AutoTokenizer
    from vncorenlp import VnCoreNLP
    from sklearn.preprocessing import LabelEncoder
    import torch
    print("✅ Import packages thành công!")
except ImportError as e:
    print(f"❌ Lỗi import: {e}")
    print("💡 Cài đặt packages cần thiết:")
    print("pip install torch transformers scikit-learn vncorenlp")
    exit(1)

class VnCoreNLPPhoBERTPreprocessor:
    """
    Preprocessor cho PhoBERT sử dụng VnCoreNLP package
    """
    
    def __init__(self, model_name="vinai/phobert-base", max_length=256, vncorenlp_jar_path=None):
        """
        Khởi tạo preprocessor
        
        Args:
            model_name: "vinai/phobert-base" hoặc "vinai/phobert-large"
            max_length: Độ dài tối đa của sequence
            vncorenlp_jar_path: Đường dẫn đến VnCoreNLP jar file
        """
        self.model_name = model_name
        self.max_length = max_length
        
        print(f"🚀 Khởi tạo {model_name} với VnCoreNLP...")
        
        # Khởi tạo PhoBERT tokenizer
        print("📝 Đang tải PhoBERT tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        
        # Khởi tạo VnCoreNLP
        self.setup_vncorenlp(vncorenlp_jar_path)
        
        # Label encoder
        self.label_encoder = LabelEncoder()
        
        print("✅ Khởi tạo hoàn thành!")
        
    def setup_vncorenlp(self, vncorenlp_jar_path=None):
        """Setup VnCoreNLP từ package đã cài đặt"""
        try:
            if not vncorenlp_jar_path:
                # Sử dụng VnCoreNLP từ project
                vncorenlp_jar_path = os.path.join(os.getcwd(), 'VnCoreNLP', 'VnCoreNLP-1.2.jar')
            
            print(f"🔧 Đang khởi tạo VnCoreNLP từ: {vncorenlp_jar_path}")
            
            if not os.path.exists(vncorenlp_jar_path):
                raise FileNotFoundError(f"VnCoreNLP jar file không tìm thấy: {vncorenlp_jar_path}")
            
            # Khởi tạo VnCoreNLP với word segmentation
            self.vncorenlp = VnCoreNLP(
                vncorenlp_jar_path, 
                annotators="wseg", 
                max_heap_size='-Xmx2g'
            )
            
            print("✅ VnCoreNLP đã sẵn sàng!")
            
            # Test VnCoreNLP
            test_text = "Đây là test VnCoreNLP."
            test_result = self.vncorenlp.annotate(test_text)
            print(f"🧪 Test VnCoreNLP: {test_result}")
            
        except Exception as e:
            print(f"❌ Lỗi khởi tạo VnCoreNLP: {e}")
            print("💡 Kiểm tra:")
            print("1. Java đã được cài đặt: java -version")
            print("2. VnCoreNLP jar file tồn tại")
            print("3. Đủ RAM (>= 2GB)")
            raise
    
    def word_segment(self, text: str) -> str:
        """
        Word segmentation sử dụng VnCoreNLP package
        """
        try:
            if not text or not text.strip():
                return ""
            
            # VnCoreNLP word segmentation
            result = self.vncorenlp.annotate(text.strip())
            
            # Extract segmented words
            segmented_words = []
            for sentence in result['sentences']:
                for word in sentence:
                    segmented_words.append(word['form'])
            
            return " ".join(segmented_words)
            
        except Exception as e:
            print(f"⚠️ Lỗi word segmentation cho text: '{text[:50]}...': {e}")
            return text  # Fallback về text gốc
    
    def preprocess_text(self, text: str) -> str:
        """
        Tiền xử lý text cho PhoBERT
        """
        # Làm sạch text cơ bản
        text = text.strip()
        if not text:
            return ""
        
        # Word segmentation với VnCoreNLP
        segmented_text = self.word_segment(text)
        
        return segmented_text
    
    def encode_texts(self, texts: List[str]) -> Dict:
        """
        Encode danh sách texts thành format cho PhoBERT
        """
        print(f"🔄 Đang encode {len(texts)} texts...")
        
        # Tiền xử lý texts với VnCoreNLP
        processed_texts = []
        print("🔤 Đang thực hiện word segmentation...")
        
        for text in tqdm(texts, desc="VnCoreNLP word segmentation"):
            processed_text = self.preprocess_text(text)
            processed_texts.append(processed_text)
        
        # Tokenize với PhoBERT tokenizer
        print("🤖 Đang tokenize với PhoBERT...")
        encoded = self.tokenizer(
            processed_texts,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Thống kê
        print(f"📊 Thống kê encoding:")
        print(f"   - Số texts: {len(texts)}")
        print(f"   - Max length: {self.max_length}")
        print(f"   - Input IDs shape: {encoded['input_ids'].shape}")
        print(f"   - Attention mask shape: {encoded['attention_mask'].shape}")
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'processed_texts': processed_texts,
            'original_texts': texts
        }
    
    def encode_labels(self, labels: List[str]) -> Dict:
        """
        Encode labels
        """
        print(f"🏷️ Đang encode {len(labels)} labels...")
        
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Thống kê labels
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"📊 Phân bố labels:")
        for label, count in zip(unique_labels, counts):
            percentage = (count / len(labels)) * 100
            print(f"   - {label}: {count} ({percentage:.1f}%)")
        
        return {
            'labels': torch.tensor(encoded_labels, dtype=torch.long),
            'label_mapping': {label: idx for idx, label in enumerate(self.label_encoder.classes_)},
            'num_classes': len(self.label_encoder.classes_),
            'original_labels': labels
        }
    
    def load_data_from_csv(self, file_path: str) -> Tuple[List[str], List[str]]:
        """
        Load dữ liệu từ CSV file
        """
        print(f"📂 Đang load dữ liệu từ: {file_path}")
        
        texts = []
        labels = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                texts.append(row['title'])
                labels.append(row['label'])
        
        print(f"✅ Đã load {len(texts)} samples")
        return texts, labels
    
    def save_processed_data(self, data: Dict, output_path: str, info: Dict = None):
        """
        Lưu dữ liệu đã xử lý
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Thêm thông tin preprocessing
        data['preprocessing_info'] = {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'tokenizer_vocab_size': self.tokenizer.vocab_size,
            'word_segmentation': 'VnCoreNLP Package',
            'vncorenlp_version': '1.2',
            'special_tokens': {
                'bos_token': self.tokenizer.bos_token,
                'eos_token': self.tokenizer.eos_token,
                'unk_token': self.tokenizer.unk_token,
                'sep_token': self.tokenizer.sep_token,
                'pad_token': self.tokenizer.pad_token,
                'cls_token': self.tokenizer.cls_token,
                'mask_token': self.tokenizer.mask_token,
            }
        }
        
        if info:
            data['preprocessing_info'].update(info)
        
        # Lưu pickle file
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        # Lưu thông tin preprocessing dưới dạng JSON
        info_path = output_path.replace('.pkl', '_info.json')
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(data['preprocessing_info'], f, indent=2, ensure_ascii=False)
        
        print(f"✅ Đã lưu: {output_path}")
        print(f"📋 Thông tin: {info_path}")
    
    def process_split(self, input_dir: str, split: str, output_dir: str, label_encoder_fitted=False):
        """
        Xử lý một split (train/val/test)
        """
        print(f"\n📁 Xử lý {split.upper()}...")
        
        # Load dữ liệu
        csv_path = os.path.join(input_dir, split, f'{split}.csv')
        
        if not os.path.exists(csv_path):
            print(f"⚠️ Không tìm thấy {csv_path}")
            return None
        
        texts, labels = self.load_data_from_csv(csv_path)
        
        # Encode texts
        encoded_texts = self.encode_texts(texts)
        
        # Encode labels
        if not label_encoder_fitted:
            encoded_labels = self.encode_labels(labels)
        else:
            # Sử dụng label encoder đã fit từ trước
            encoded_label_values = self.label_encoder.transform(labels)
            encoded_labels = {
                'labels': torch.tensor(encoded_label_values, dtype=torch.long),
                'label_mapping': {label: idx for idx, label in enumerate(self.label_encoder.classes_)},
                'num_classes': len(self.label_encoder.classes_),
                'original_labels': labels
            }
        
        # Combine data
        processed_data = {
            **encoded_texts,
            **encoded_labels
        }
        
        # Lưu dữ liệu
        output_path = os.path.join(output_dir, f'{split}_processed.pkl')
        self.save_processed_data(processed_data, output_path)
        
        return processed_data
    
    def process_dataset(self, input_dir: str, output_dir: str):
        """
        Xử lý toàn bộ dataset
        """
        print(f"=" * 70)
        print(f"🎯 TIỀN XỬ LÝ DỮ LIỆU CHO {self.model_name.upper()}")
        print(f"📏 Max length: {self.max_length}")
        print(f"🔧 VnCoreNLP: Package version với JAR file")
        print(f"📂 Input: {input_dir}")
        print(f"📂 Output: {output_dir}")
        print(f"=" * 70)
        
        # Tạo thư mục output
        os.makedirs(output_dir, exist_ok=True)
        
        # Các splits cần xử lý
        splits = ['train', 'val', 'test']
        
        # Thu thập tất cả labels để fit label encoder
        print("🔄 Thu thập labels từ tất cả splits...")
        all_labels = []
        for split in splits:
            csv_path = os.path.join(input_dir, split, f'{split}.csv')
            if os.path.exists(csv_path):
                _, labels = self.load_data_from_csv(csv_path)
                all_labels.extend(labels)
        
        # Fit label encoder với tất cả labels
        print("🏷️ Đang fit label encoder...")
        self.label_encoder.fit(all_labels)
        label_mapping = {label: idx for idx, label in enumerate(self.label_encoder.classes_)}
        print(f"📋 Label mapping: {label_mapping}")
        
        # Xử lý từng split
        results = {}
        for i, split in enumerate(splits):
            label_encoder_fitted = (i > 0)  # Chỉ fit ở split đầu tiên
            result = self.process_split(input_dir, split, output_dir, label_encoder_fitted)
            if result:
                results[split] = result
        
        # Lưu thống kê tổng hợp
        summary = {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'label_mapping': label_mapping,
            'num_classes': len(self.label_encoder.classes_),
            'splits_processed': list(results.keys()),
            'total_samples': sum(len(results[split]['labels']) for split in results),
            'split_sizes': {split: len(results[split]['labels']) for split in results}
        }
        
        summary_path = os.path.join(output_dir, 'processing_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n" + "=" * 70)
        print("✅ HOÀN THÀNH TIỀN XỬ LÝ DỮ LIỆU!")
        print(f"📊 Tổng kết:")
        print(f"   - Model: {self.model_name}")
        print(f"   - Splits processed: {results.keys()}")
        print(f"   - Total samples: {summary['total_samples']}")
        print(f"   - Output directory: {output_dir}")
        print(f"   - Summary: {summary_path}")
        print("=" * 70)
        
        return results

def main():
    """
    Main function để chạy preprocessing
    """
    print("🇻🇳 VnCoreNLP + PhoBERT Data Preprocessing")
    print("=" * 50)
    
    # Cấu hình
    INPUT_DIR = "simple_dataset"
    BASE_OUTPUT_DIR = "data-vncorenlp-v2"
    
    # Kiểm tra dữ liệu nguồn
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Không tìm thấy thư mục dữ liệu: {INPUT_DIR}")
        print("💡 Đảm bảo có thư mục simple_dataset với train/val/test")
        return
    
    print(f"📂 Input directory: {INPUT_DIR}")
    print(f"📂 Output base directory: {BASE_OUTPUT_DIR}")
    print()
    
    # Xử lý PhoBERT-base
    print("🔥 PHẦN 1: PHOBERT-BASE")
    try:
        processor_base = VnCoreNLPPhoBERTPreprocessor(
            model_name="vinai/phobert-base",
            max_length=256
        )
        
        output_dir_base = os.path.join(BASE_OUTPUT_DIR, "phobert-base")
        processor_base.process_dataset(INPUT_DIR, output_dir_base)
        
    except Exception as e:
        print(f"❌ Lỗi xử lý PhoBERT-base: {e}")
        print("💡 Kiểm tra lại cấu hình và dependencies")
    
    print("\n" + "="*70 + "\n")
    
    # Xử lý PhoBERT-large
    print("🔥 PHẦN 2: PHOBERT-LARGE")
    try:
        processor_large = VnCoreNLPPhoBERTPreprocessor(
            model_name="vinai/phobert-large",
            max_length=256
        )
        
        output_dir_large = os.path.join(BASE_OUTPUT_DIR, "phobert-large")
        processor_large.process_dataset(INPUT_DIR, output_dir_large)
        
    except Exception as e:
        print(f"❌ Lỗi xử lý PhoBERT-large: {e}")
        print("💡 Kiểm tra lại cấu hình và dependencies")
    
    print("\n" + "="*70)
    print("🎉 HOÀN THÀNH TẤT CẢ!")
    print("📁 Dữ liệu đã được xử lý và lưu trong:")
    print(f"   - PhoBERT-base: {BASE_OUTPUT_DIR}/phobert-base/")
    print(f"   - PhoBERT-large: {BASE_OUTPUT_DIR}/phobert-large/")
    print("\n💡 Bước tiếp theo:")
    print("   python scripts/phobert/train_phobert.py --data_dir data-vncorenlp-v2")
    print("="*70)

if __name__ == "__main__":
    main() 