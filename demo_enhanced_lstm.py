#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Demo script cho Enhanced LSTM với pre-trained embeddings
Load model đã train và test với text mới
"""

import os
import pickle
import torch
import json
import argparse
from typing import List, Dict
import numpy as np

# Import từ train script
from train_lstm_enhanced import EnhancedLSTM, EnhancedVietnameseTextProcessor

class EnhancedLSTMPredictor:
    """Predictor class cho Enhanced LSTM model"""
    
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.model = None
        self.processor = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_names = ['non-clickbait', 'clickbait']
        
        self.load_model()
    
    def load_model(self):
        """Load trained model và text processor"""
        print(f"📥 Loading Enhanced LSTM model từ {self.model_dir}")
        
        # Load text processor
        processor_path = os.path.join(self.model_dir, 'text_processor_enhanced.pkl')
        if not os.path.exists(processor_path):
            raise FileNotFoundError(f"Không tìm thấy text processor: {processor_path}")
        
        with open(processor_path, 'rb') as f:
            self.processor = pickle.load(f)
        
        print(f"✅ Loaded text processor:")
        print(f"   Vocab size: {self.processor.vocab_size}")
        print(f"   Embedding dim: {self.processor.embedding_dim}")
        print(f"   Max length: {self.processor.max_length}")
        print(f"   Embedding type: {self.processor.embedding_type}")
        
        # Load model architecture từ training history
        history_path = os.path.join(self.model_dir, 'training_history_enhanced.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
            print(f"✅ Best validation F1: {history.get('best_val_f1', 0):.4f}")
        
        # Create model
        self.model = EnhancedLSTM(
            vocab_size=self.processor.vocab_size,
            embedding_dim=self.processor.embedding_dim,
            hidden_dim=256,  # Default values
            num_layers=2,
            dropout=0.5,
            num_classes=2,
            embedding_matrix=self.processor.embedding_matrix
        )
        
        # Load model weights
        model_path = os.path.join(self.model_dir, 'best_model_enhanced.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy model weights: {model_path}")
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded successfully!")
        print(f"   Device: {self.device}")
    
    def predict_single(self, text: str) -> Dict:
        """Predict cho một text"""
        # Preprocess text
        sequence = self.processor.text_to_sequence(text)
        
        # Convert to tensor
        input_tensor = torch.tensor([sequence], dtype=torch.long).to(self.device)
        attention_mask = (input_tensor != 0).long()
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_tensor, attention_mask)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        result = {
            'text': text,
            'predicted_label': self.label_names[predicted_class],
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': {
                'non-clickbait': probabilities[0][0].item(),
                'clickbait': probabilities[0][1].item()
            },
            'sequence_length': len(sequence)
        }
        
        return result
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Predict cho nhiều texts"""
        results = []
        
        print(f"🔄 Predicting {len(texts)} texts...")
        
        for i, text in enumerate(texts):
            result = self.predict_single(text)
            results.append(result)
            
            if (i + 1) % 10 == 0:
                print(f"   Processed {i + 1}/{len(texts)}")
        
        return results
    
    def analyze_predictions(self, results: List[Dict]):
        """Phân tích kết quả predictions"""
        print("\n" + "="*60)
        print("📊 PHÂN TÍCH KẾT QUẢ PREDICTIONS")
        print("="*60)
        
        clickbait_count = sum(1 for r in results if r['predicted_class'] == 1)
        non_clickbait_count = len(results) - clickbait_count
        
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        print(f"📈 Tổng quan:")
        print(f"   Total texts: {len(results)}")
        print(f"   Clickbait: {clickbait_count} ({clickbait_count/len(results)*100:.1f}%)")
        print(f"   Non-clickbait: {non_clickbait_count} ({non_clickbait_count/len(results)*100:.1f}%)")
        print(f"   Average confidence: {avg_confidence:.3f}")
        
        # High confidence predictions
        high_conf_threshold = 0.8
        high_conf_results = [r for r in results if r['confidence'] > high_conf_threshold]
        
        print(f"\n🎯 High confidence predictions (>{high_conf_threshold}):")
        print(f"   Count: {len(high_conf_results)}/{len(results)} ({len(high_conf_results)/len(results)*100:.1f}%)")
        
        if high_conf_results:
            print("\n📋 Top confident predictions:")
            sorted_results = sorted(results, key=lambda x: x['confidence'], reverse=True)
            
            for i, result in enumerate(sorted_results[:5]):
                print(f"\n{i+1}. [{result['predicted_label'].upper()}] (confidence: {result['confidence']:.3f})")
                print(f"   Text: {result['text'][:100]}{'...' if len(result['text']) > 100 else ''}")

def demo_interactive(predictor: EnhancedLSTMPredictor):
    """Demo tương tác"""
    print("\n" + "="*60)
    print("🎮 INTERACTIVE DEMO - Enhanced LSTM")
    print("="*60)
    print("Nhập text để phân loại clickbait (gõ 'quit' để thoát)")
    print()
    
    while True:
        text = input("📝 Nhập text: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("👋 Tạm biệt!")
            break
        
        if not text:
            continue
        
        result = predictor.predict_single(text)
        
        print(f"\n📊 Kết quả:")
        print(f"   Label: {result['predicted_label'].upper()}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Probabilities:")
        print(f"     Non-clickbait: {result['probabilities']['non-clickbait']:.3f}")
        print(f"     Clickbait: {result['probabilities']['clickbait']:.3f}")
        print(f"   Sequence length: {result['sequence_length']}")
        
        # Interpretation
        if result['confidence'] > 0.8:
            conf_level = "RẤT TIN CẬY"
        elif result['confidence'] > 0.6:
            conf_level = "TIN CẬY"
        else:
            conf_level = "KHÔNG CHẮC CHẮN"
        
        print(f"   Đánh giá: {conf_level}")
        print()

def demo_predefined_examples(predictor: EnhancedLSTMPredictor):
    """Demo với examples có sẵn"""
    print("\n" + "="*60)
    print("📋 DEMO VỚI EXAMPLES CÓ SẴN")
    print("="*60)
    
    examples = [
        # Clickbait examples
        "Bạn sẽ không thể tin được điều gì xảy ra tiếp theo!",
        "10 bí mật mà các bác sĩ không muốn bạn biết",
        "Cách kiếm tiền online 100 triệu/tháng chỉ với 1 click",
        "Sao Việt bất ngờ tiết lộ bí mật gây sốc",
        "Thực phẩm này có thể giết bạn - bạn ăn mỗi ngày!",
        
        # Non-clickbait examples  
        "Chính phủ công bố chính sách mới về thuế thu nhập cá nhân",
        "Nghiên cứu mới về tác động của biến đổi khí hậu đến nông nghiệp",
        "Đại học Quốc gia Hà Nội tuyển sinh năm 2024",
        "Tỷ giá USD/VND hôm nay ổn định ở mức 24.000",
        "WHO khuyến cáo về phòng chống dịch bệnh mùa đông"
    ]
    
    results = predictor.predict_batch(examples)
    
    print("\n📊 KẾT QUẢ CHI TIẾT:")
    print("-" * 80)
    
    for i, result in enumerate(results):
        status = "✅" if result['predicted_label'] == 'clickbait' and i < 5 else "✅" if result['predicted_label'] == 'non-clickbait' and i >= 5 else "❌"
        
        print(f"\n{i+1}. {status} [{result['predicted_label'].upper()}] (confidence: {result['confidence']:.3f})")
        print(f"   Text: {result['text']}")
        print(f"   Probabilities: Non-clickbait: {result['probabilities']['non-clickbait']:.3f}, "
              f"Clickbait: {result['probabilities']['clickbait']:.3f}")
    
    # Analyze results
    predictor.analyze_predictions(results)

def main():
    parser = argparse.ArgumentParser(description="Enhanced LSTM Demo")
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory chứa trained model')
    parser.add_argument('--mode', type=str, default='interactive',
                       choices=['interactive', 'examples', 'both'],
                       help='Demo mode')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_dir):
        print(f"❌ Model directory không tồn tại: {args.model_dir}")
        return
    
    print("=== ENHANCED LSTM DEMO ===")
    print()
    
    # Load predictor
    try:
        predictor = EnhancedLSTMPredictor(args.model_dir)
    except Exception as e:
        print(f"❌ Lỗi loading model: {e}")
        return
    
    if args.mode in ['examples', 'both']:
        demo_predefined_examples(predictor)
    
    if args.mode in ['interactive', 'both']:
        demo_interactive(predictor)

if __name__ == "__main__":
    main() 