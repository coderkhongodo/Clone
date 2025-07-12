#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Demo script TF-IDF + SVM với VnCoreNLP cho clickbait detection
Dataset: simple_dataset (title classification)
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict
import argparse
from tqdm import tqdm
import json
from datetime import datetime
import pickle
import re

# VnCoreNLP và sklearn
import vncorenlp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, precision_score, recall_score, f1_score
)
import seaborn as sns
import matplotlib.pyplot as plt

class VnCoreNLPProcessor:
    """Text processor sử dụng VnCoreNLP"""
    
    def __init__(self, vncorenlp_file: str = None):
        self.vncorenlp_file = vncorenlp_file or self._download_vncorenlp()
        self.annotator = None
        self._initialize_annotator()
    
    def _download_vncorenlp(self):
        """Download VnCoreNLP nếu chưa có"""
        import subprocess
        import urllib.request
        
        vncorenlp_dir = "VnCoreNLP"
        jar_file = "VnCoreNLP-1.1.1.jar"
        jar_path = os.path.join(vncorenlp_dir, jar_file)
        
        if not os.path.exists(jar_path):
            print("🔄 Downloading VnCoreNLP...")
            os.makedirs(vncorenlp_dir, exist_ok=True)
            
            # Download jar file
            url = "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar"
            try:
                urllib.request.urlretrieve(url, jar_path)
            except Exception as e:
                print(f"❌ Error downloading from {url}: {e}")
                # Alternative: manual download instruction
                print("🔧 Please download VnCoreNLP-1.1.1.jar manually:")
                print("   1. Go to: https://github.com/vncorenlp/VnCoreNLP/releases")
                print("   2. Download VnCoreNLP-1.1.1.jar")
                print(f"   3. Place it at: {jar_path}")
                raise
            print(f"✅ Downloaded VnCoreNLP to {jar_path}")
        
        return jar_path
    
    def _initialize_annotator(self):
        """Khởi tạo VnCoreNLP annotator"""
        try:
            print(f"🚀 Initializing VnCoreNLP from {self.vncorenlp_file}")
            self.annotator = vncorenlp.VnCoreNLP(
                self.vncorenlp_file, 
                annotators="wseg,pos", 
                max_heap_size='-Xmx2g'
            )
            print("✅ VnCoreNLP initialized successfully!")
        except Exception as e:
            print(f"❌ Error initializing VnCoreNLP: {e}")
            print("🔧 Please ensure Java is installed and VnCoreNLP jar file exists")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text với VnCoreNLP"""
        if not text or pd.isna(text):
            return ""
        
        # Basic cleaning
        text = str(text).strip()
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'[^\w\s\u00C0-\u1EF9]', ' ', text)  # Keep Vietnamese chars
        
        try:
            # VnCoreNLP word segmentation và POS tagging
            annotated = self.annotator.annotate(text)
            
            # Extract words (có thể filter theo POS nếu cần)
            words = []
            for sentence in annotated['sentences']:
                for word_info in sentence:
                    word = word_info['form']
                    pos = word_info['posTag']
                    
                    # Filter theo POS tags (optional)
                    # Giữ lại: Noun, Verb, Adjective, etc.
                    if len(word) > 1 and not word.isdigit():
                        words.append(word.lower())
            
            return ' '.join(words)
        
        except Exception as e:
            print(f"⚠️ Error processing text: {text[:50]}... - {e}")
            # Fallback: basic tokenization
            return text.lower()
    
    def close(self):
        """Đóng VnCoreNLP annotator"""
        if self.annotator:
            self.annotator.close()

class TfIdfSvmClassifier:
    """TF-IDF + SVM classifier với VnCoreNLP"""
    
    def __init__(self, vncorenlp_file: str = None):
        self.vncorenlp_file = vncorenlp_file
        self.processor = VnCoreNLPProcessor(vncorenlp_file)
        self.pipeline = None
        self.label_names = ['non-clickbait', 'clickbait']
        
        print(f"🚀 Khởi tạo TF-IDF + SVM Classifier với VnCoreNLP")
    
    def load_data(self, data_path: str) -> Dict:
        """Load dữ liệu từ simple_dataset"""
        print(f"📂 Loading data từ {data_path}")
        
        # Load train, val, test
        train_df = pd.read_csv(os.path.join(data_path, 'train/train.csv'))
        val_df = pd.read_csv(os.path.join(data_path, 'val/val.csv'))
        test_df = pd.read_csv(os.path.join(data_path, 'test/test.csv'))
        
        # Convert labels
        label_map = {'non-clickbait': 0, 'clickbait': 1}
        
        data = {
            'train': {
                'texts': train_df['title'].tolist(),
                'labels': [label_map[label] for label in train_df['label']]
            },
            'val': {
                'texts': val_df['title'].tolist(),
                'labels': [label_map[label] for label in val_df['label']]
            },
            'test': {
                'texts': test_df['title'].tolist(),
                'labels': [label_map[label] for label in test_df['label']]
            }
        }
        
        print(f"✅ Data loaded successfully:")
        print(f"   Train: {len(data['train']['texts'])} samples")
        print(f"   Val: {len(data['val']['texts'])} samples")  
        print(f"   Test: {len(data['test']['texts'])} samples")
        
        # Hiển thị phân bố labels
        for split in ['train', 'val', 'test']:
            labels = data[split]['labels']
            clickbait_count = sum(labels)
            total = len(labels)
            print(f"   {split.capitalize()}: {clickbait_count}/{total} clickbait ({clickbait_count/total*100:.1f}%)")
        
        return data
    
    def preprocess_texts(self, texts: List[str]) -> List[str]:
        """Preprocess list of texts với VnCoreNLP"""
        print(f"🔄 Preprocessing {len(texts)} texts với VnCoreNLP...")
        
        processed_texts = []
        for text in tqdm(texts, desc="Processing texts"):
            processed = self.processor.preprocess_text(text)
            processed_texts.append(processed)
        
        return processed_texts
    
    def create_pipeline(self) -> Pipeline:
        """Tạo TF-IDF + SVM pipeline"""
        # TF-IDF Vectorizer
        tfidf = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),  # unigram, bigram, trigram
            min_df=2,            # Ignore terms xuất hiện < 2 times
            max_df=0.95,         # Ignore terms xuất hiện > 95% documents
            stop_words=None,     # Không dùng stop words (VnCoreNLP đã filter)
            lowercase=False,     # Đã lowercase trong preprocessing
            token_pattern=r'\S+' # Split by whitespace
        )
        
        # SVM Classifier
        svm = SVC(
            kernel='linear',
            C=1.0,
            probability=True,    # Enable probability estimates
            random_state=42
        )
        
        # Pipeline
        pipeline = Pipeline([
            ('tfidf', tfidf),
            ('svm', svm)
        ])
        
        return pipeline
    
    def train(self, data: Dict, output_dir: str = './tfidf_svm_results', 
              tune_hyperparams: bool = True):
        """Train TF-IDF + SVM model"""
        print(f"\n🔄 Bắt đầu training TF-IDF + SVM...")
        
        # Preprocess texts
        train_texts = self.preprocess_texts(data['train']['texts'])
        val_texts = self.preprocess_texts(data['val']['texts'])
        
        # Combine train + val for training (traditional approach)
        all_texts = train_texts + val_texts
        all_labels = data['train']['labels'] + data['val']['labels']
        
        print(f"📊 Training data: {len(all_texts)} samples")
        
        # Create pipeline
        self.pipeline = self.create_pipeline()
        
        if tune_hyperparams:
            print("🔧 Tuning hyperparameters với GridSearch...")
            
            # Hyperparameter grid
            param_grid = {
                'tfidf__max_features': [3000, 5000, 10000],
                'tfidf__ngram_range': [(1, 2), (1, 3)],
                'svm__C': [0.1, 1.0, 10.0],
                'svm__kernel': ['linear', 'rbf']
            }
            
            # GridSearchCV
            grid_search = GridSearchCV(
                self.pipeline,
                param_grid,
                cv=5,
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1
            )
            
            # Fit
            grid_search.fit(all_texts, all_labels)
            
            # Best model
            self.pipeline = grid_search.best_estimator_
            
            print(f"✅ Best parameters: {grid_search.best_params_}")
            print(f"✅ Best CV F1-score: {grid_search.best_score_:.4f}")
            
            # Save grid search results
            grid_results = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            
            os.makedirs(output_dir, exist_ok=True)
            with open(f'{output_dir}/grid_search_results.json', 'w') as f:
                json.dump(grid_results, f, indent=2, default=str)
        
        else:
            print("🔄 Training với default parameters...")
            self.pipeline.fit(all_texts, all_labels)
        
        # Save model
        os.makedirs(output_dir, exist_ok=True)
        with open(f'{output_dir}/tfidf_svm_model.pkl', 'wb') as f:
            pickle.dump(self.pipeline, f)
        
        print(f"✅ Training completed! Model saved to {output_dir}")
        
        return self.pipeline
    
    def evaluate(self, test_data: Dict, output_dir: str = './tfidf_svm_results'):
        """Evaluate model trên test set với chi tiết đầy đủ"""
        print(f"\n📊 Evaluating trên test set...")
        
        # Load model nếu chưa có
        if self.pipeline is None:
            model_path = f'{output_dir}/tfidf_svm_model.pkl'
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.pipeline = pickle.load(f)
                print(f"📥 Loaded model from {model_path}")
            else:
                raise ValueError("No trained model found!")
        
        # Preprocess test texts
        test_texts = self.preprocess_texts(test_data['test']['texts'])
        true_labels = test_data['test']['labels']
        
        # Predict
        predictions = self.pipeline.predict(test_texts)
        prediction_probs = self.pipeline.predict_proba(test_texts)
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(true_labels, predictions)
        
        # Overall metrics (weighted average)
        precision_weighted = precision_score(true_labels, predictions, average='weighted')
        recall_weighted = recall_score(true_labels, predictions, average='weighted')
        f1_weighted = f1_score(true_labels, predictions, average='weighted')
        
        # Per-class metrics
        precision_per_class = precision_score(true_labels, predictions, average=None)
        recall_per_class = recall_score(true_labels, predictions, average=None)
        f1_per_class = f1_score(true_labels, predictions, average=None)
        
        # Macro average
        precision_macro = precision_score(true_labels, predictions, average='macro')
        recall_macro = recall_score(true_labels, predictions, average='macro')
        f1_macro = f1_score(true_labels, predictions, average='macro')
        
        # Support (số samples thực tế cho mỗi class)
        unique, counts = np.unique(true_labels, return_counts=True)
        support_per_class = counts.tolist()
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # Classification report
        class_report = classification_report(
            true_labels, predictions, 
            target_names=self.label_names,
            output_dict=True
        )
        
        # TF-IDF feature analysis
        feature_names = self.pipeline['tfidf'].get_feature_names_out()
        svm_coef = self.pipeline['svm'].coef_[0]
        
        # Top features for each class
        top_features = {
            'clickbait': [],
            'non_clickbait': []
        }
        
        # Clickbait features (positive coefficients)
        clickbait_indices = np.argsort(svm_coef)[-20:]
        top_features['clickbait'] = [
            (feature_names[i], float(svm_coef[i])) 
            for i in clickbait_indices[::-1]
        ]
        
        # Non-clickbait features (negative coefficients)
        nonclickbait_indices = np.argsort(svm_coef)[:20]
        top_features['non_clickbait'] = [
            (feature_names[i], float(svm_coef[i])) 
            for i in nonclickbait_indices
        ]
        
        # Detailed results structure
        results = {
            "accuracy": accuracy,
            "precision": precision_weighted,
            "recall": recall_weighted,
            "f1": f1_weighted,
            "precision_per_class": precision_per_class.tolist(),
            "recall_per_class": recall_per_class.tolist(),
            "f1_per_class": f1_per_class.tolist(),
            "support_per_class": support_per_class,
            "confusion_matrix": cm.tolist(),
            "classification_report": {
                self.label_names[0]: {
                    "precision": float(precision_per_class[0]),
                    "recall": float(recall_per_class[0]),
                    "f1-score": float(f1_per_class[0]),
                    "support": float(support_per_class[0])
                },
                self.label_names[1]: {
                    "precision": float(precision_per_class[1]),
                    "recall": float(recall_per_class[1]),
                    "f1-score": float(f1_per_class[1]),
                    "support": float(support_per_class[1])
                },
                "accuracy": accuracy,
                "macro avg": {
                    "precision": precision_macro,
                    "recall": recall_macro,
                    "f1-score": f1_macro,
                    "support": float(sum(support_per_class))
                },
                "weighted avg": {
                    "precision": precision_weighted,
                    "recall": recall_weighted,
                    "f1-score": f1_weighted,
                    "support": float(sum(support_per_class))
                }
            },
            "label_mapping": {
                self.label_names[0]: 0,
                self.label_names[1]: 1
            },
            "model_info": {
                "model_name": "TF-IDF + SVM + VnCoreNLP",
                "model_path": f"{output_dir}/tfidf_svm_model.pkl",
                "test_samples": len(true_labels),
                "feature_count": len(feature_names),
                "vncorenlp_file": self.vncorenlp_file
            },
            "top_features": top_features
        }
        
        # Print detailed results
        print(f"\n📈 DETAILED TEST RESULTS:")
        print(f"=" * 60)
        print(f"📊 Overall Metrics:")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Precision (weighted): {precision_weighted:.4f}")
        print(f"   Recall (weighted): {recall_weighted:.4f}")
        print(f"   F1-score (weighted): {f1_weighted:.4f}")
        
        print(f"\n🎯 Per-Class Metrics:")
        for i, label in enumerate(self.label_names):
            print(f"   {label.upper()}:")
            print(f"     Precision: {precision_per_class[i]:.4f}")
            print(f"     Recall: {recall_per_class[i]:.4f}")
            print(f"     F1-score: {f1_per_class[i]:.4f}")
            print(f"     Support: {support_per_class[i]} samples")
        
        print(f"\n📈 Macro Average:")
        print(f"   Precision: {precision_macro:.4f}")
        print(f"   Recall: {recall_macro:.4f}")
        print(f"   F1-score: {f1_macro:.4f}")
        
        print(f"\n🔢 Confusion Matrix:")
        print(f"   Predicted →")
        print(f"   Actual ↓     {self.label_names[0]:<15} {self.label_names[1]:<15}")
        print(f"   {self.label_names[0]:<12} {cm[0][0]:<15} {cm[0][1]:<15}")
        print(f"   {self.label_names[1]:<12} {cm[1][0]:<15} {cm[1][1]:<15}")
        
        # Feature analysis
        print(f"\n🔍 Top Features Analysis:")
        print(f"   📊 Total features: {len(feature_names)}")
        print(f"   🎯 Top Clickbait indicators:")
        for feature, coef in top_features['clickbait'][:10]:
            print(f"     '{feature}': {coef:.3f}")
        
        print(f"   📰 Top Non-clickbait indicators:")
        for feature, coef in top_features['non_clickbait'][:10]:
            print(f"     '{feature}': {coef:.3f}")
        
        # Calculate additional insights
        true_positives = cm[1][1]
        false_positives = cm[0][1]
        false_negatives = cm[1][0]
        true_negatives = cm[0][0]
        
        print(f"\n🔍 Detailed Analysis:")
        print(f"   True Positives (Correct clickbait): {true_positives}")
        print(f"   False Positives (Wrong clickbait): {false_positives}")
        print(f"   False Negatives (Missed clickbait): {false_negatives}")
        print(f"   True Negatives (Correct non-clickbait): {true_negatives}")
        
        print(f"\n💾 Model Info:")
        print(f"   Model: TF-IDF + SVM + VnCoreNLP")
        print(f"   Test samples: {len(true_labels)}")
        print(f"   Features: {len(feature_names)}")
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        with open(f'{output_dir}/test_results_detailed.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_dir}/test_results_detailed.json")
        
        # Plot confusion matrix
        self.plot_confusion_matrix(true_labels, predictions, output_dir)
        
        return results
    
    def plot_confusion_matrix(self, true_labels, predictions, output_dir):
        """Vẽ confusion matrix chi tiết"""
        cm = confusion_matrix(true_labels, predictions)
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        plt.figure(figsize=(10, 8))
        
        # Create annotations with both counts and percentages
        annotations = []
        for i in range(cm.shape[0]):
            row = []
            for j in range(cm.shape[1]):
                count = cm[i, j]
                percent = cm_percent[i, j]
                row.append(f'{count}\n({percent:.1f}%)')
            annotations.append(row)
        
        # Plot heatmap
        sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', 
                   xticklabels=[f'{label}\n(Predicted)' for label in self.label_names], 
                   yticklabels=[f'{label}\n(Actual)' for label in self.label_names],
                   cbar_kws={'label': 'Number of Samples'})
        
        plt.title('Confusion Matrix - TF-IDF + SVM + VnCoreNLP\n' + 
                 f'Total Samples: {cm.sum()}, Accuracy: {accuracy_score(true_labels, predictions):.3f}', 
                 fontsize=14, pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # Add performance metrics text
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='weighted')
        recall = recall_score(true_labels, predictions, average='weighted')
        f1 = f1_score(true_labels, predictions, average='weighted')
        
        metrics_text = f'Accuracy: {accuracy:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}'
        plt.figtext(0.5, 0.02, metrics_text, ha='center', fontsize=10, 
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/confusion_matrix_detailed.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Confusion matrix saved: {output_dir}/confusion_matrix_detailed.png")
    
    def predict_single(self, text: str, output_dir: str = './tfidf_svm_results') -> Dict:
        """Predict cho một text"""
        # Load model nếu chưa load
        if self.pipeline is None:
            model_path = f'{output_dir}/tfidf_svm_model.pkl'
            with open(model_path, 'rb') as f:
                self.pipeline = pickle.load(f)
        
        # Preprocess
        processed_text = self.processor.preprocess_text(text)
        
        # Predict
        prediction = self.pipeline.predict([processed_text])[0]
        probabilities = self.pipeline.predict_proba([processed_text])[0]
        
        result = {
            'text': text,
            'processed_text': processed_text,
            'predicted_label': self.label_names[prediction],
            'predicted_class': int(prediction),
            'confidence': float(max(probabilities)),
            'probabilities': {
                'non-clickbait': float(probabilities[0]),
                'clickbait': float(probabilities[1])
            }
        }
        
        return result
    
    def close(self):
        """Cleanup resources"""
        if self.processor:
            self.processor.close()

def demo_training(args):
    """Demo training TF-IDF + SVM với VnCoreNLP"""
    print("="*60)
    print("🚀 DEMO TRAINING TF-IDF + SVM + VnCoreNLP")
    print("="*60)
    
    # Initialize classifier
    classifier = TfIdfSvmClassifier(args.vncorenlp_file)
    
    try:
        # Load data
        data = classifier.load_data(args.data_path)
        
        # Train
        classifier.train(
            data, 
            output_dir=args.output_dir,
            tune_hyperparams=args.tune_hyperparams
        )
        
        # Evaluate
        results = classifier.evaluate(data, args.output_dir)
        
        print("\n" + "="*60)
        print("✅ DEMO TRAINING HOÀN THÀNH!")
        print("="*60)
        print(f"📁 Results saved in: {args.output_dir}/")
        print(f"   📊 JSON: test_results_detailed.json")
        print(f"   📈 Chart: confusion_matrix_detailed.png")
        print(f"   🤖 Model: tfidf_svm_model.pkl")
        if args.tune_hyperparams:
            print(f"   🔧 Grid search: grid_search_results.json")
        
    finally:
        # Cleanup
        classifier.close()
    
    return classifier, results

def demo_interactive(args):
    """Demo interactive prediction"""
    print("="*60)
    print("🎮 DEMO INTERACTIVE - TF-IDF + SVM + VnCoreNLP")
    print("="*60)
    
    classifier = TfIdfSvmClassifier(args.vncorenlp_file)
    
    try:
        print("Nhập text để phân loại clickbait (gõ 'quit' để thoát)")
        print()
        
        while True:
            text = input("📝 Nhập text: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("👋 Tạm biệt!")
                break
            
            if not text:
                continue
            
            try:
                result = classifier.predict_single(text, args.output_dir)
                
                print(f"\n📊 Kết quả:")
                print(f"   Original: {result['text']}")
                print(f"   Processed: {result['processed_text']}")
                print(f"   Label: {result['predicted_label'].upper()}")
                print(f"   Confidence: {result['confidence']:.3f}")
                print(f"   Probabilities:")
                print(f"     Non-clickbait: {result['probabilities']['non-clickbait']:.3f}")
                print(f"     Clickbait: {result['probabilities']['clickbait']:.3f}")
                
                # Interpretation
                if result['confidence'] > 0.8:
                    conf_level = "RẤT TIN CẬY"
                elif result['confidence'] > 0.6:
                    conf_level = "TIN CẬY"
                else:
                    conf_level = "KHÔNG CHẮC CHẮN"
                
                print(f"   Đánh giá: {conf_level}")
                print()
                
            except Exception as e:
                print(f"❌ Lỗi: {e}")
                print("Hãy đảm bảo model đã được train trước!")
    
    finally:
        classifier.close()

def demo_predefined_examples(args):
    """Demo với examples có sẵn"""
    print("="*60)
    print("📋 DEMO VỚI EXAMPLES CÓ SẴN - TF-IDF + SVM + VnCoreNLP")
    print("="*60)
    
    classifier = TfIdfSvmClassifier(args.vncorenlp_file)
    
    try:
        examples = [
            "5 bí mật gây sốc mà bạn chưa bao giờ biết về smartphone",
            "Nghiên cứu mới về tác động của AI đến giáo dục",
            "Cô gái 20 tuổi kiếm được 100 triệu/tháng bằng cách này",
            "Chính phủ công bố chính sách mới về thuế môi trường",
            "Bạn sẽ không tin những gì xảy ra khi cô ấy mở cửa",
            "Báo cáo tài chính quý 3 của các doanh nghiệp lớn",
            "10 thói quen buổi sáng giúp bạn thành công",
            "Ủy ban nhân dân thành phố họp bàn về quy hoạch đô thị"
        ]
        
        print("🔍 Đang phân tích các examples...")
        results = []
        
        for i, text in enumerate(examples, 1):
            try:
                result = classifier.predict_single(text, args.output_dir)
                results.append(result)
                
                print(f"\n{i}. [{result['predicted_label'].upper()}] (confidence: {result['confidence']:.3f})")
                print(f"   Original: {text}")
                print(f"   Processed: {result['processed_text']}")
                
            except Exception as e:
                print(f"❌ Lỗi cho example {i}: {e}")
        
        # Tổng kết chi tiết
        if results:
            clickbait_count = sum(1 for r in results if r['predicted_class'] == 1)
            high_conf_clickbait = sum(1 for r in results if r['predicted_class'] == 1 and r['confidence'] > 0.8)
            high_conf_nonclickbait = sum(1 for r in results if r['predicted_class'] == 0 and r['confidence'] > 0.8)
            avg_confidence = sum(r['confidence'] for r in results) / len(results)
            
            print(f"\n📊 Tổng kết chi tiết:")
            print(f"   Total examples: {len(results)}")
            print(f"   Clickbait predictions: {clickbait_count}/{len(results)} ({clickbait_count/len(results)*100:.1f}%)")
            print(f"   Non-clickbait predictions: {len(results)-clickbait_count}/{len(results)} ({(len(results)-clickbait_count)/len(results)*100:.1f}%)")
            print(f"   Average confidence: {avg_confidence:.3f}")
            print(f"   High confidence (>0.8):")
            print(f"     Clickbait: {high_conf_clickbait}/{clickbait_count} ({high_conf_clickbait/max(clickbait_count,1)*100:.1f}%)")
            print(f"     Non-clickbait: {high_conf_nonclickbait}/{len(results)-clickbait_count} ({high_conf_nonclickbait/max(len(results)-clickbait_count,1)*100:.1f}%)")
    
    finally:
        classifier.close()

def main():
    parser = argparse.ArgumentParser(description='TF-IDF + SVM + VnCoreNLP Clickbait Detection Demo')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'interactive', 'examples'],
                       help='Demo mode')
    parser.add_argument('--data_path', type=str, default='simple_dataset',
                       help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='./tfidf_svm_results',
                       help='Output directory')
    parser.add_argument('--vncorenlp_file', type=str, default=None,
                       help='Path to VnCoreNLP jar file')
    parser.add_argument('--tune_hyperparams', action='store_true',
                       help='Tune hyperparameters với GridSearchCV')
    
    args = parser.parse_args()
    
    print(f"🎯 Mode: {args.mode}")
    print(f"📂 Data path: {args.data_path}")
    print(f"🤖 Model: TF-IDF + SVM + VnCoreNLP")
    print(f"🔧 Tune hyperparams: {args.tune_hyperparams}")
    print()
    
    if args.mode == 'train':
        demo_training(args)
    elif args.mode == 'interactive':
        demo_interactive(args)
    elif args.mode == 'examples':
        demo_predefined_examples(args)

if __name__ == "__main__":
    main() 