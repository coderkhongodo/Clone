#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import pickle
import csv
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm

# Import PhoBERT (không cần VnCoreNLP thư viện nữa)
try:
    from transformers import AutoTokenizer
    from sklearn.preprocessing import LabelEncoder
    import torch
except ImportError as e:
    print(f"ERROR: Lỗi import: {e}")
    print("Vui lòng cài đặt các thư viện cần thiết:")
    print("pip install -r requirements.txt")
    exit(1)

class LocalVnCoreNLP:
    """
    Local VnCoreNLP wrapper sử dụng jar file thay vì thư viện Python
    """
    
    def __init__(self, jar_path: str, max_heap_size: str = '-Xmx1g'):
        """
        Khởi tạo VnCoreNLP từ jar file local
        
        Args:
            jar_path: Đường dẫn đến VnCoreNLP-1.1.1.jar
            max_heap_size: Heap size cho JVM (default: -Xmx1g)
        """
        self.jar_path = jar_path
        self.max_heap_size = max_heap_size
        
        if not os.path.exists(jar_path):
            raise FileNotFoundError(f"VnCoreNLP jar file không tìm thấy: {jar_path}")
        
        # Test VnCoreNLP có hoạt động không
        self._test_vncorenlp()
        
        print(f"✅ VnCoreNLP initialized successfully từ {jar_path}")
    
    def _test_vncorenlp(self):
        """Test VnCoreNLP hoạt động bình thường"""
        test_text = "Xin chào các bạn"
        try:
            result = self._run_vncorenlp(test_text)
            if not result:
                raise Exception("VnCoreNLP không trả về kết quả")
        except Exception as e:
            raise Exception(f"VnCoreNLP test failed: {e}")
    
    def _run_vncorenlp(self, text: str) -> List[List[str]]:
        """
        Chạy VnCoreNLP qua command line và parse kết quả
        
        Args:
            text: Text cần phân tách từ
            
        Returns:
            List of sentences, mỗi sentence là list of words
        """
        # Tạo file tạm cho input
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.txt', delete=False) as input_file:
            input_file.write(text)
            input_file_path = input_file.name
        
        # Tạo file tạm cho output  
        output_file_path = input_file_path + '.output'
        
        try:
            # Chạy VnCoreNLP
            cmd = [
                'java', self.max_heap_size, '-jar', self.jar_path,
                '-fin', input_file_path,
                '-fout', output_file_path,
                '-annotators', 'wseg'
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=30,
                encoding='utf-8'
            )
            
            if result.returncode != 0:
                raise Exception(f"VnCoreNLP failed: {result.stderr}")
            
            # Đọc kết quả
            if not os.path.exists(output_file_path):
                raise Exception("VnCoreNLP output file không được tạo")
            
            return self._parse_vncorenlp_output(output_file_path)
        
        finally:
            # Cleanup files
            for file_path in [input_file_path, output_file_path]:
                if os.path.exists(file_path):
                    os.unlink(file_path)
    
    def _parse_vncorenlp_output(self, output_file_path: str) -> List[List[str]]:
        """Parse VnCoreNLP output file thành list of sentences"""
        sentences = []
        current_sentence = []
        
        with open(output_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    if current_sentence:
                        sentences.append(current_sentence)
                        current_sentence = []
                    continue
                
                # Parse line format: word_index word pos_tag ner_tag head dep_rel
                parts = line.split('\t')
                if len(parts) >= 2:
                    word = parts[1]  # Word ở column thứ 2
                    current_sentence.append(word)
        
        # Add last sentence if exists
        if current_sentence:
            sentences.append(current_sentence)
        
        return sentences
    
    def tokenize(self, text: str) -> List[List[str]]:
        """
        Tokenize text thành sentences của words
        
        Args:
            text: Input text
            
        Returns:
            List of sentences, mỗi sentence là list of words
        """
        if not text or not text.strip():
            return [[]]
        
        return self._run_vncorenlp(text.strip())

class PhoBERTPreprocessor:
    def __init__(self, model_name="vinai/phobert-base", max_length=256, vncorenlp_jar_path=None):
        """
        Khởi tạo preprocessor cho PhoBERT với VnCoreNLP local
        
        Args:
            model_name: "vinai/phobert-base" hoặc "vinai/phobert-large"
            max_length: Độ dài tối đa của sequence
            vncorenlp_jar_path: Đường dẫn đến VnCoreNLP jar file
        """
        self.model_name = model_name
        self.max_length = max_length
        
        print(f"🚀 Đang khởi tạo {model_name}...")
        
        # Khởi tạo tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        
        # Khởi tạo VnCoreNLP local
        self.setup_local_vncorenlp(vncorenlp_jar_path)
        
        # Label encoder
        self.label_encoder = LabelEncoder()
        
        print("✅ Khởi tạo hoàn thành!")
    
    def setup_local_vncorenlp(self, vncorenlp_jar_path):
        """Setup VnCoreNLP từ jar file local"""
        try:
            if not vncorenlp_jar_path:
                # Default path trong project
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                vncorenlp_jar_path = os.path.join(project_root, 'VnCoreNLP', 'VnCoreNLP-1.1.1.jar')
            
            print(f"🔄 Đang setup VnCoreNLP từ: {vncorenlp_jar_path}")
            
            self.segmenter = LocalVnCoreNLP(
                jar_path=vncorenlp_jar_path,
                max_heap_size='-Xmx1g'
            )
            
            print("✅ VnCoreNLP local đã sẵn sàng!")
            
        except Exception as e:
            print(f"❌ Lỗi khởi tạo VnCoreNLP: {e}")
            print("💡 Hướng dẫn:")
            print("   1. Đảm bảo Java đã được cài đặt: java -version")
            print("   2. Đảm bảo file VnCoreNLP-1.1.1.jar tồn tại")
            print(f"   3. Path hiện tại: {vncorenlp_jar_path}")
            self.segmenter = None
    
    def word_segment(self, text: str) -> str:
        """
        Word segmentation sử dụng VnCoreNLP local
        """
        if not self.segmenter:
            return text
        
        try:
            # VnCoreNLP word segmentation
            segmented = self.segmenter.tokenize(text)
            
            # Flatten sentences thành single string
            result = []
            for sentence in segmented:
                result.extend(sentence)
            
            return " ".join(result)
            
        except Exception as e:
            print(f"⚠️ Lỗi word segmentation: {e}")
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
        print(f"🔄 Đang encode {len(texts)} texts...")
        
        # Tiền xử lý texts với progress bar
        processed_texts = []
        for text in tqdm(texts, desc="VnCoreNLP word segmentation"):
            processed_text = self.preprocess_text(text)
            processed_texts.append(processed_text)
        
        # Tokenize với PhoBERT tokenizer
        print("🔄 Đang tokenize với PhoBERT...")
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
        
        print(f"✅ Đã lưu: {output_path}")
    
    def process_dataset(self, input_dir: str, output_dir: str):
        """
        Xử lý toàn bộ dataset
        """
        print(f"=" * 60)
        print(f"📊 TIỀN XỬ LÝ DỮ LIỆU CHO {self.model_name.upper()}")
        print(f"📏 Max length: {self.max_length}")
        print(f"🔧 VnCoreNLP: Local JAR file")
        print(f"=" * 60)
        
        # Tạo thư mục output
        os.makedirs(output_dir, exist_ok=True)
        
        splits = ['train', 'val', 'test']
        all_labels = []
        
        # Thu thập tất cả labels để fit label encoder
        print("🔄 Đang thu thập labels...")
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
        
        print(f"📋 Label mapping: {label_info['label_mapping']}")
        print()
        
        # Xử lý từng split
        for split in splits:
            print(f"🔄 Đang xử lý {split} set...")
            
            csv_path = os.path.join(input_dir, split, f'{split}.csv')
            if not os.path.exists(csv_path):
                print(f"⚠️ Không tìm thấy {csv_path}")
                continue
            
            # Load dữ liệu
            texts, labels = self.load_data(csv_path)
            print(f"📊 {split}: {len(texts)} samples")
            
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
            
            # Thống kê
            stats = {
                'split': split,
                'num_samples': len(texts),
                'max_length': self.max_length,
                'model_name': self.model_name,
                'avg_length': np.mean([len(text.split()) for text in encoded_texts['processed_texts']]),
                'label_distribution': {label: labels.count(label) for label in set(labels)},
                'vncorenlp_method': 'Local JAR file'
            }
            
            stats_path = os.path.join(output_dir, f'{split}_stats.json')
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Hoàn thành {split} set")
            print()
        
        # Lưu thông tin preprocessing
        info = {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'tokenizer_vocab_size': self.tokenizer.vocab_size,
            'label_mapping': label_info['label_mapping'],
            'num_classes': label_info['num_classes'],
            'preprocessing_info': {
                'word_segmentation': 'VnCoreNLP Local JAR',
                'vncorenlp_path': self.segmenter.jar_path if self.segmenter else None,
                'special_tokens': self.tokenizer.special_tokens_map
            }
        }
        
        info_path = os.path.join(output_dir, 'preprocessing_info.json')
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        
        print("🎉 Hoàn thành tiền xử lý dữ liệu!")
        print(f"📁 Dữ liệu đã lưu tại: {output_dir}")

def main():
    """
    Main function - Cấu hình đường dẫn ở đây
    """
    print("🚀 PhoBERT Preprocessing với VnCoreNLP Local")
    print()
    
    # Cấu hình đường dẫn
    INPUT_DIR = "simple_dataset"  # Thư mục chứa dữ liệu gốc
    
    # Đường dẫn VnCoreNLP jar file (tự động detect từ project structure)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    VNCORENLP_JAR = os.path.join(project_root, 'VnCoreNLP', 'VnCoreNLP-1.1.1.jar')
    
    print(f"📂 Input directory: {INPUT_DIR}")
    print(f"🔧 VnCoreNLP JAR: {VNCORENLP_JAR}")
    
    if not os.path.exists(VNCORENLP_JAR):
        print(f"❌ VnCoreNLP jar file không tìm thấy: {VNCORENLP_JAR}")
        print("💡 Hãy đảm bảo file VnCoreNLP-1.1.1.jar tồn tại!")
        return
    
    print()
    
    # Xử lý cho PhoBERT-base
    print("=" * 60)
    print("📊 PHOBERT-BASE với VnCoreNLP Local")
    print("=" * 60)
    
    try:
        processor_base = PhoBERTPreprocessor(
            model_name="vinai/phobert-base",
            max_length=256,
            vncorenlp_jar_path=VNCORENLP_JAR
        )
        processor_base.process_dataset(INPUT_DIR, "data-bert-local/phobert-base")
        
    except Exception as e:
        print(f"❌ Lỗi xử lý PhoBERT-base: {e}")
        return
    
    print("\n" + "="*60 + "\n")
    
    # Xử lý cho PhoBERT-large  
    print("=" * 60)
    print("📊 PHOBERT-LARGE với VnCoreNLP Local")
    print("=" * 60)
    
    try:
        processor_large = PhoBERTPreprocessor(
            model_name="vinai/phobert-large", 
            max_length=256,
            vncorenlp_jar_path=VNCORENLP_JAR
        )
        processor_large.process_dataset(INPUT_DIR, "data-bert-local/phobert-large")
        
    except Exception as e:
        print(f"❌ Lỗi xử lý PhoBERT-large: {e}")
        return
    
    print("\n" + "="*60)
    print("🎉 HOÀN THÀNH TẤT CẢ!")
    print("📁 Dữ liệu đã được lưu trong data-bert-local/")
    print("🚀 Bây giờ có thể chạy training script!")

if __name__ == "__main__":
    main() 