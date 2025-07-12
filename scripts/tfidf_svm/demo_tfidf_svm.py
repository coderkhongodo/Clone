#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Demo script cho TF-IDF + SVM classifier
Test với sample data để kiểm tra tính năng
"""

import os
import sys
from train_tfidf_svm import TfIdfSVMClassifier, VietnameseTextPreprocessor

def test_preprocessor():
    """Test text preprocessing"""
    print("TEST: Testing Vietnamese Text Preprocessor...")
    
    # Sample texts
    sample_texts = [
        "Bài viết này THẬT SỰ sẽ khiến bạn bất ngờ!!!",
        "Tin HOT: 10 bí mật mà bạn KHÔNG THỂ TIN ĐƯỢC",
        "Báo cáo chính thức từ Bộ Y tế về tình hình dịch bệnh",
        "Nghiên cứu mới cho thấy hiệu quả của vaccine"
    ]
    
    try:
        preprocessor = VietnameseTextPreprocessor()
        
        print("\nLOG: Sample preprocessing results:")
        for i, text in enumerate(sample_texts, 1):
            print(f"\n{i}. Original: {text}")
            
            # Test preprocessing steps
            processed = preprocessor.preprocess_text(text, filter_pos=True)
            print(f"   Processed: {processed}")
            
    except Exception as e:
        print(f"ERROR: Preprocessing test failed: {e}")
        print("TIP: Có thể VnCoreNLP chưa được cài đặt hoặc configured đúng")
        return False
    
    print("SUCCESS: Preprocessing test passed!")
    return True

def test_classifier_with_sample_data():
    """Test classifier với sample data"""
    print("\nTEST: Testing TF-IDF + SVM Classifier...")
    
    # Sample data
    sample_texts = [
        "Tin sốc sẽ khiến bạn bất ngờ về scandal nghệ sĩ nổi tiếng",
        "10 bí mật mà bạn không thể tin được về cuộc sống",
        "Sự thật đằng sau vụ việc sẽ làm bạn rùng mình",
        "Báo cáo chính thức từ Bộ Y tế về tình hình dịch COVID-19",
        "Nghiên cứu khoa học mới về hiệu quả vaccine phòng COVID-19",
        "Thông tin chính thức từ Chính phủ về chính sách kinh tế",
        "Kết quả điều tra của cơ quan chức năng về vụ tai nạn",
        "Dữ liệu thống kê về tình hình kinh tế trong quý 3",
        "Tin thể thao: đội tuyển Việt Nam thắng trận đấu quan trọng",
        "Thông báo về thay đổi giờ làm việc của các cơ quan hành chính"
    ]
    
    sample_labels = [
        "clickbait", "clickbait", "clickbait",
        "non-clickbait", "non-clickbait", "non-clickbait",
        "non-clickbait", "non-clickbait", "non-clickbait", "non-clickbait"
    ]
    
    try:
        # Initialize classifier
        classifier = TfIdfSVMClassifier(
            max_features=1000,  # Smaller cho demo
            ngram_range=(1, 2)
        )
        
        print(f"STATS: Sample data: {len(sample_texts)} texts")
        print(f"   Clickbait: {sample_labels.count('clickbait')}")
        print(f"   Non-clickbait: {sample_labels.count('non-clickbait')}")
        
        # Train với sample data
        print("\nPROCESS: Training với sample data...")
        classifier.train(
            sample_texts, sample_labels,
            val_texts=None, val_labels=None,
            grid_search=False  # Skip grid search cho demo
        )
        
        # Test predictions
        print("\n🔮 Testing predictions:")
        test_texts = [
            "Điều bất ngờ sẽ xảy ra với bạn hôm nay",
            "Báo cáo từ Bộ Giáo dục về kế hoạch năm học mới",
            "10 sự thật sốc về cuộc sống mà bạn chưa biết"
        ]
        
        predictions = classifier.predict(test_texts)
        probabilities = classifier.predict_proba(test_texts)
        
        for i, (text, pred, prob) in enumerate(zip(test_texts, predictions, probabilities)):
            print(f"\n{i+1}. Text: {text}")
            print(f"   Prediction: {pred}")
            print(f"   Probabilities: {prob}")
        
        # Feature importance (nếu linear kernel)
        try:
            feature_importance = classifier.get_feature_importance(top_k=10)
            if feature_importance:
                print("\nSEARCH: Top features:")
                print("   Clickbait indicators:")
                for feature, score in feature_importance['top_positive'][:5]:
                    print(f"     {feature}: {score:.3f}")
                
                print("   Non-clickbait indicators:")
                for feature, score in feature_importance['top_negative'][:5]:
                    print(f"     {feature}: {score:.3f}")
        except Exception as e:
            print(f"WARNING: Feature importance error: {e}")
        
        print("SUCCESS: Classifier test passed!")
        return True
        
    except Exception as e:
        print(f"ERROR: Classifier test failed: {e}")
        return False

def main():
    print("=== TF-IDF + SVM DEMO ===")
    print("Demo script để test các tính năng cơ bản")
    print()
    
    # Test 1: Preprocessor
    preprocessor_ok = test_preprocessor()
    
    if not preprocessor_ok:
        print("\nWARNING: Preprocessor test failed. Bạn vẫn có thể tiếp tục nhưng")
        print("   preprocessing có thể không hoạt động optimal.")
        
        response = input("\nTiếp tục với classifier test? (y/n): ").lower()
        if response not in ['y', 'yes']:
            return
    
    # Test 2: Classifier
    test_classifier_with_sample_data()
    
    print("\nTARGET: Demo completed!")
    print("\nTIP: Next steps:")
    print("   1. Prepare your data in data/train/, data/val/, data/test/")
    print("   2. Run: python run_tfidf_svm.py")
    print("   3. Or: python train_tfidf_svm.py --data_dir data")

if __name__ == "__main__":
    # Kiểm tra dependencies
    missing_deps = []
    
    try:
        import sklearn
    except ImportError:
        missing_deps.append("scikit-learn")
    
    try:
        import pyvi
    except ImportError:
        missing_deps.append("pyvi")
    
    try:
        import matplotlib
        import seaborn
    except ImportError:
        missing_deps.append("matplotlib seaborn")
    
    if missing_deps:
        print("ERROR: Missing dependencies:")
        for dep in missing_deps:
            print(f"   pip install {dep}")
        print()
        print("Install dependencies và chạy lại script này.")
        sys.exit(1)
    
    main() 