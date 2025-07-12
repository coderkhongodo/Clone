#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import pickle
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# ML libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import joblib

# Vietnamese NLP
try:
    from pyvi import ViTokenizer, ViPosTagger
except ImportError:
    print("ERROR: pyvi không có sẵn. Cài đặt: pip install pyvi")
    exit(1)

# Visualization
import matplotlib
matplotlib.use('Agg')  # Fix multiprocessing ResourceTracker errors
import matplotlib.pyplot as plt
import seaborn as sns

class VietnameseTextPreprocessor:
    """
    Text preprocessor cho tiếng Việt với pyvi
    """
    
    def __init__(self):
        """
        Khởi tạo preprocessor với pyvi
        """
        print("PROCESS: Đang khởi tạo pyvi...")
        
        try:
            # Test pyvi functionality
            test_text = "Xin chào"
            _ = ViTokenizer.tokenize(test_text)
            _ = ViPosTagger.postagging(ViTokenizer.tokenize(test_text))
            print("SUCCESS: pyvi đã sẵn sàng!")
            self.nlp_available = True
        except Exception as e:
            print(f"ERROR: Lỗi khởi tạo pyvi: {e}")
            print("TIP: Cài đặt: pip install pyvi")
            self.nlp_available = False
    
    def remove_punctuation_and_special_chars(self, text: str) -> str:
        """
        Bỏ dấu câu và ký tự đặc biệt
        """
        # Giữ lại chữ cái tiếng Việt, số, và khoảng trắng
        text = re.sub(r'[^\w\s\u00C0-\u024F\u1E00-\u1EFF]', ' ', text)
        # Bỏ nhiều khoảng trắng liên tiếp
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def word_segment_and_pos_tag(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Tách từ và phân tích từ loại với pyvi
        
        Returns:
            words: Danh sách từ đã tách
            pos_tags: Danh sách từ loại tương ứng
        """
        if not self.nlp_available:
            # Fallback: chỉ split theo khoảng trắng
            words = text.split()
            pos_tags = ['X'] * len(words)  # Unknown POS
            return words, pos_tags
        
        try:
            # pyvi word segmentation
            segmented_text = ViTokenizer.tokenize(text)
            
            # pyvi POS tagging
            pos_result = ViPosTagger.postagging(segmented_text)
            words = pos_result[0]  # List of words
            pos_tags = pos_result[1]  # List of POS tags
            
            return words, pos_tags
        
        except Exception as e:
            print(f"WARNING: Lỗi word segmentation: {e}")
            # Fallback
            words = text.split()
            pos_tags = ['X'] * len(words)
            return words, pos_tags
    
    def filter_words_by_pos(self, words: List[str], pos_tags: List[str], 
                           keep_pos: List[str] = None) -> List[str]:
        """
        Lọc từ theo từ loại (pyvi POS tags)
        
        Args:
            words: Danh sách từ
            pos_tags: Danh sách từ loại
            keep_pos: Danh sách từ loại cần giữ (None = giữ tất cả)
        """
        if keep_pos is None:
            # Mặc định cho pyvi: giữ danh từ, động từ, tính từ, trạng từ
            # pyvi sử dụng tag set tương tự VnCoreNLP nhưng có thể khác một chút
            keep_pos = ['N', 'V', 'A', 'R', 'Np', 'Ny', 'Nc', 'NN', 'VB', 'JJ', 'RB']
        
        filtered_words = []
        for word, pos in zip(words, pos_tags):
            # Kiểm tra nếu từ loại bắt đầu bằng một trong các prefix được giữ
            # hoặc khớp chính xác
            if any(pos.startswith(prefix) or pos == prefix for prefix in keep_pos):
                # Bỏ qua từ quá ngắn hoặc chỉ là ký tự đặc biệt
                if len(word.strip()) > 1 and not word.strip().isdigit():
                    filtered_words.append(word.strip())
        
        return filtered_words
    
    def preprocess_text(self, text: str, filter_pos: bool = True) -> str:
        """
        Pipeline preprocessing hoàn chỉnh
        
        Args:
            text: Text cần xử lý
            filter_pos: Có lọc theo từ loại không
        
        Returns:
            Text đã được preprocessing
        """
        # 1. Lowercase
        text = text.lower()
        
        # 2. Bỏ dấu câu và ký tự đặc biệt
        text = self.remove_punctuation_and_special_chars(text)
        
        # 3. Word segmentation và POS tagging
        words, pos_tags = self.word_segment_and_pos_tag(text)
        
        # 4. Lọc từ theo từ loại (optional)
        if filter_pos:
            words = self.filter_words_by_pos(words, pos_tags)
        
        # 5. Kết hợp lại thành text
        processed_text = ' '.join(words)
        
        return processed_text

class TfIdfSVMClassifier:
    """
    TF-IDF + SVM Classifier cho phân loại clickbait tiếng Việt
    """
    
    def __init__(self, 
                 max_features: int = 10000,
                 min_df: int = 2,
                 max_df: float = 0.95,
                 ngram_range: Tuple[int, int] = (1, 2),
                 svm_params: Dict = None):
        """
        Khởi tạo classifier
        
        Args:
            max_features: Số feature tối đa cho TF-IDF
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            ngram_range: N-gram range
            svm_params: Hyperparameters cho SVM
        """
        self.preprocessor = VietnameseTextPreprocessor()
        
        # TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            sublinear_tf=True,
            strip_accents=None,  # Đã xử lý ở preprocessing
            stop_words=None      # Đã xử lý ở POS filtering
        )
        
        # SVM default parameters
        default_svm_params = {
            'C': 1.0,
            'kernel': 'linear',
            'gamma': 'scale',
            'class_weight': 'balanced',
            'random_state': 42
        }
        
        if svm_params:
            default_svm_params.update(svm_params)
        
        self.svm = SVC(**default_svm_params, probability=True)
        
        # Label encoder
        self.label_encoder = LabelEncoder()
        
        # Pipeline
        self.pipeline = Pipeline([
            ('tfidf', self.vectorizer),
            ('svm', self.svm)
        ])
        
        # Statistics
        self.training_stats = {}
        self.feature_names = []
    
    def load_data(self, data_dir: str) -> Dict[str, Tuple[List[str], List[str]]]:
        """
        Load dữ liệu từ CSV files
        
        Args:
            data_dir: Thư mục chứa train/val/test
            
        Returns:
            Dictionary chứa texts và labels cho mỗi split
        """
        data = {}
        splits = ['train', 'val', 'test']
        
        for split in splits:
            csv_path = os.path.join(data_dir, split, f'{split}.csv')
            
            if not os.path.exists(csv_path):
                print(f"WARNING: Không tìm thấy {csv_path}")
                continue
            
            texts, labels = [], []
            
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    texts.append(row['title'])
                    labels.append(row['label'])
            
            data[split] = (texts, labels)
            print(f"STATS: Loaded {split}: {len(texts)} samples")
        
        return data
    
    def preprocess_texts(self, texts: List[str], show_progress: bool = True) -> List[str]:
        """
        Preprocess danh sách texts
        
        Args:
            texts: Danh sách texts cần xử lý
            show_progress: Hiển thị progress bar
            
        Returns:
            Danh sách texts đã preprocessing
        """
        processed_texts = []
        
        iterator = tqdm(texts, desc="Preprocessing") if show_progress else texts
        
        for text in iterator:
            processed_text = self.preprocessor.preprocess_text(text, filter_pos=True)
            processed_texts.append(processed_text)
        
        return processed_texts
    
    def train(self, train_texts: List[str], train_labels: List[str],
              val_texts: List[str] = None, val_labels: List[str] = None,
              grid_search: bool = True):
        """
        Training với optional grid search
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts (optional)
            val_labels: Validation labels (optional)
            grid_search: Có sử dụng grid search không
        """
        print("PROCESS: Bắt đầu training...")
        
        # Preprocess texts
        print("PROCESS: Preprocessing training data...")
        processed_train_texts = self.preprocess_texts(train_texts)
        
        # Encode labels
        encoded_train_labels = self.label_encoder.fit_transform(train_labels)
        
        # Update class weights in SVM if provided
        if hasattr(self.svm, 'class_weight') and isinstance(self.svm.class_weight, dict):
            # Convert string-based class weights to encoded class weights
            encoded_class_weight = {}
            for class_name, weight in self.svm.class_weight.items():
                if class_name in self.label_encoder.classes_:
                    encoded_idx = self.label_encoder.transform([class_name])[0]
                    encoded_class_weight[encoded_idx] = weight
            
            # Update SVM class weight
            self.svm.class_weight = encoded_class_weight
            # Also update in pipeline
            self.pipeline.named_steps['svm'].class_weight = encoded_class_weight
            
            print(f"⚖️ Updated class weights: {encoded_class_weight}")
            print(f"   Class mapping: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
        # Grid search nếu được yêu cầu
        if grid_search:
            print("SEARCH: Performing grid search...")
            
            param_grid = {
                'tfidf__max_features': [5000, 10000, 15000],
                'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
                'svm__C': [0.1, 1.0, 10.0],
                'svm__kernel': ['linear', 'rbf']
            }
            
            grid_search_cv = GridSearchCV(
                self.pipeline,
                param_grid,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search_cv.fit(processed_train_texts, encoded_train_labels)
            
            self.pipeline = grid_search_cv.best_estimator_
            
            print("SUCCESS: Best parameters:")
            for param, value in grid_search_cv.best_params_.items():
                print(f"   {param}: {value}")
            print(f"SUCCESS: Best CV score: {grid_search_cv.best_score_:.4f}")
            
        else:
            # Training thông thường
            self.pipeline.fit(processed_train_texts, encoded_train_labels)
        
        # Get feature names
        self.feature_names = self.pipeline.named_steps['tfidf'].get_feature_names_out()
        
        # Validation nếu có
        if val_texts and val_labels:
            print("PROCESS: Evaluating on validation set...")
            val_metrics = self.evaluate(val_texts, val_labels, split_name="Validation")
            self.training_stats['validation'] = val_metrics
        
        # Training statistics
        train_predictions = self.predict(train_texts)
        train_accuracy = accuracy_score(train_labels, train_predictions)
        
        self.training_stats['train_accuracy'] = train_accuracy
        self.training_stats['num_features'] = len(self.feature_names)
        self.training_stats['label_mapping'] = {
            label: idx for idx, label in enumerate(self.label_encoder.classes_)
        }
        
        print(f"SUCCESS: Training completed!")
        print(f"STATS: Training accuracy: {train_accuracy:.4f}")
        print(f"STATS: Number of features: {len(self.feature_names)}")
    
    def predict(self, texts: List[str]) -> List[str]:
        """
        Predict labels
        """
        processed_texts = self.preprocess_texts(texts, show_progress=False)
        encoded_predictions = self.pipeline.predict(processed_texts)
        predictions = self.label_encoder.inverse_transform(encoded_predictions)
        return predictions.tolist()
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict probabilities
        """
        processed_texts = self.preprocess_texts(texts, show_progress=False)
        probabilities = self.pipeline.predict_proba(processed_texts)
        return probabilities
    
    def evaluate(self, texts: List[str], labels: List[str], 
                split_name: str = "Test") -> Dict:
        """
        Comprehensive evaluation
        """
        print(f"PROCESS: Evaluating on {split_name} set...")
        
        # Predictions
        predictions = self.predict(texts)
        probabilities = self.predict_proba(texts)
        
        # Metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support = \
            precision_recall_fscore_support(labels, predictions, average=None)
        
        # Classification report
        report = classification_report(
            labels, predictions,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        # Confusion matrix
        encoded_labels = self.label_encoder.transform(labels)
        encoded_predictions = self.label_encoder.transform(predictions)
        cm = confusion_matrix(encoded_labels, encoded_predictions)
        
        # AUC nếu có 2 classes
        auc = None
        if len(self.label_encoder.classes_) == 2:
            try:
                auc = roc_auc_score(
                    encoded_labels,
                    probabilities[:, 1]
                )
            except:
                pass
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'support_per_class': support,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': predictions,
            'probabilities': probabilities
        }
        
        # Print results
        print(f"SUCCESS: {split_name} Results:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-score: {f1:.4f}")
        if auc:
            print(f"   AUC: {auc:.4f}")
        
        return results
    
    def get_feature_importance(self, top_k: int = 20) -> Dict:
        """
        Lấy feature importance từ SVM
        """
        if self.pipeline.named_steps['svm'].kernel != 'linear':
            print("WARNING: Feature importance chỉ available cho linear kernel")
            return {}
        
        # Get coefficients
        coef = self.pipeline.named_steps['svm'].coef_[0]
        
        # Convert to array if sparse matrix
        if hasattr(coef, 'toarray'):
            coef = coef.toarray().flatten()
        
        # Get top positive và negative features
        top_positive_idx = np.argsort(coef)[-top_k:][::-1]
        top_negative_idx = np.argsort(coef)[:top_k]
        
        feature_importance = {
            'top_positive': [
                (self.feature_names[idx], float(coef[idx])) 
                for idx in top_positive_idx
            ],
            'top_negative': [
                (self.feature_names[idx], float(coef[idx])) 
                for idx in top_negative_idx
            ]
        }
        
        return feature_importance
    
    def save_model(self, output_path: str):
        """
        Lưu model và metadata
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        model_data = {
            'pipeline': self.pipeline,
            'label_encoder': self.label_encoder,
            'training_stats': self.training_stats,
            'feature_names': self.feature_names,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, output_path)
        print(f"SUCCESS: Model saved: {output_path}")
    
    def load_model(self, model_path: str):
        """
        Load model đã training
        """
        model_data = joblib.load(model_path)
        
        self.pipeline = model_data['pipeline']
        self.label_encoder = model_data['label_encoder']
        self.training_stats = model_data['training_stats']
        self.feature_names = model_data['feature_names']
        
        print(f"SUCCESS: Model loaded: {model_path}")

def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

def plot_confusion_matrix(cm, class_names, title="Confusion Matrix", save_path=None):
    """
    Vẽ confusion matrix
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"STATS: Confusion matrix saved: {save_path}")
        plt.close()  # Close instead of show for Agg backend
    else:
        plt.close()

def plot_feature_importance(feature_importance, save_path=None):
    """
    Vẽ feature importance
    """
    if not feature_importance:
        return
    
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Top positive features
        pos_features, pos_scores = zip(*feature_importance['top_positive'])
        # Convert scores to float to avoid sparse matrix issues
        pos_scores = [float(score) for score in pos_scores]
        ax1.barh(range(len(pos_features)), pos_scores)
        ax1.set_yticks(range(len(pos_features)))
        ax1.set_yticklabels(pos_features)
        ax1.set_title('Top Positive Features (Clickbait)')
        ax1.set_xlabel('SVM Coefficient')
        
        # Top negative features
        neg_features, neg_scores = zip(*feature_importance['top_negative'])
        # Convert scores to float to avoid sparse matrix issues
        neg_scores = [float(score) for score in neg_scores]
        ax2.barh(range(len(neg_features)), neg_scores)
        ax2.set_yticks(range(len(neg_features)))
        ax2.set_yticklabels(neg_features)
        ax2.set_title('Top Negative Features (Non-clickbait)')
        ax2.set_xlabel('SVM Coefficient')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"STATS: Feature importance saved: {save_path}")
        
        plt.close()  # Close instead of show for non-interactive
        
    except Exception as e:
        print(f"WARNING: Error plotting feature importance: {e}")
        print("Feature importance data saved to JSON file.")

def main():
    parser = argparse.ArgumentParser(description="TF-IDF + SVM Classifier for Vietnamese Clickbait Detection")
    
    parser.add_argument('--data_dir', type=str, default='/home/huflit/NCKH/simple_dataset',
                       help='Directory containing train/val/test data')
    parser.add_argument('--output_dir', type=str, default='models_tfidf_svm_v3',
                       help='Output directory for models and results')
    parser.add_argument('--clickbait_weight', type=float, default=2.0,
                       help='Class weight for clickbait class (higher = more focus on clickbait)')
    parser.add_argument('--max_features', type=int, default=10000,
                       help='Maximum number of TF-IDF features')
    parser.add_argument('--ngram_range', type=str, default='1,2',
                       help='N-gram range (format: min,max)')
    parser.add_argument('--grid_search', action='store_true',
                       help='Perform grid search for hyperparameters')
    parser.add_argument('--test_only', type=str, default=None,
                       help='Path to trained model for testing only')
    
    args = parser.parse_args()
    
    # Parse ngram_range
    ngram_min, ngram_max = map(int, args.ngram_range.split(','))
    ngram_range = (ngram_min, ngram_max)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=== TF-IDF + SVM VIETNAMESE CLICKBAIT CLASSIFIER ===")
    print(f"📁 Data directory: {args.data_dir}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"CONFIG: Max features: {args.max_features}")
    print(f"CONFIG: N-gram range: {ngram_range}")
    print(f"⚖️ Clickbait weight: {args.clickbait_weight}x")
    print(f"SEARCH: Grid search: {args.grid_search}")
    print()
    
    # Create class weights
    class_weights = {
        'non-clickbait': 1.0,
        'clickbait': args.clickbait_weight
    }
    
    # Initialize classifier
    classifier = TfIdfSVMClassifier(
        max_features=args.max_features,
        ngram_range=ngram_range,
        svm_params={'class_weight': class_weights}
    )
    
    if args.test_only:
        # Load trained model và test
        print("PROCESS: Loading trained model...")
        classifier.load_model(args.test_only)
        
        # Load test data
        data = classifier.load_data(args.data_dir)
        if 'test' not in data:
            print("ERROR: No test data found!")
            return
        
        test_texts, test_labels = data['test']
        
        # Evaluate
        test_results = classifier.evaluate(test_texts, test_labels, "Test")
        
        # Save results
        results_path = os.path.join(args.output_dir, 'test_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for key, value in test_results.items():
                if isinstance(value, np.ndarray):
                    serializable_results[key] = value.tolist()
                elif isinstance(value, np.float64):
                    serializable_results[key] = float(value)
                else:
                    serializable_results[key] = value
            
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"SUCCESS: Test results saved: {results_path}")
        return
    
    # Load data
    data = classifier.load_data(args.data_dir)
    
    if 'train' not in data:
        print("ERROR: No training data found!")
        return
    
    train_texts, train_labels = data['train']
    
    val_texts, val_labels = None, None
    if 'val' in data:
        val_texts, val_labels = data['val']
    
    # Training
    classifier.train(
        train_texts, train_labels,
        val_texts, val_labels,
        grid_search=args.grid_search
    )
    
    # Test evaluation nếu có
    if 'test' in data:
        test_texts, test_labels = data['test']
        test_results = classifier.evaluate(test_texts, test_labels, "Test")
        
        # Save test results
        results_path = os.path.join(args.output_dir, 'test_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for key, value in test_results.items():
                if isinstance(value, np.ndarray):
                    serializable_results[key] = value.tolist()
                elif isinstance(value, np.float64):
                    serializable_results[key] = float(value)
                else:
                    serializable_results[key] = value
            
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        # Plot confusion matrix
        cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
        plot_confusion_matrix(
            test_results['confusion_matrix'],
            classifier.label_encoder.classes_,
            title="TF-IDF + SVM Confusion Matrix",
            save_path=cm_path
        )
        
        # Feature importance
        feature_importance = classifier.get_feature_importance(top_k=20)
        if feature_importance:
            fi_path = os.path.join(args.output_dir, 'feature_importance.png')
            plot_feature_importance(feature_importance, save_path=fi_path)
            
            # Save feature importance as JSON
            fi_json_path = os.path.join(args.output_dir, 'feature_importance.json')
            with open(fi_json_path, 'w', encoding='utf-8') as f:
                json.dump(feature_importance, f, indent=2, ensure_ascii=False)
    
    # Save model
    model_path = os.path.join(args.output_dir, 'tfidf_svm_model.pkl')
    classifier.save_model(model_path)
    
    # Save training statistics
    stats_path = os.path.join(args.output_dir, 'training_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        # Convert numpy types recursively for JSON serialization
        serializable_stats = convert_numpy_types(classifier.training_stats)
        json.dump(serializable_stats, f, indent=2, ensure_ascii=False)
    
    print("SUCCESS: Training completed successfully!")
    print(f"📁 Results saved in: {args.output_dir}")

if __name__ == "__main__":
    main() 