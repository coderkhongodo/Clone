# 🎯 CLICKBAIT DETECTION PROJECT

Dự án phát hiện clickbait sử dụng 3 model chính trên dataset `simple_dataset`.

## 📊 **CẤU TRÚC DỰ ÁN**

```
📁 NCKH/
├── 📊 simple_dataset/          # Dữ liệu gốc (train/val/test)
├── 📊 data_alpaca_v2/         # Dữ liệu Alpaca format từ simple_dataset
├── 🏆 evaluation_results/      # Kết quả đánh giá các models
├── 🤖 models/                  # Các models đã train
│   ├── phobert/               # PhoBERT models + data
│   ├── tfidf_svm/             # TF-IDF SVM models
│   └── vistral/               # Vistral models
├── 📝 scripts/                 # Scripts theo từng model
│   ├── phobert/               # PhoBERT scripts
│   ├── lstm/                  # LSTM scripts
│   ├── tfidf_svm/             # TF-IDF SVM scripts
│   ├── vistral/               # Vistral scripts
│   ├── data_prep/             # Data preparation scripts
│   └── train_launcher.py      # Training launcher
└── 📋 requirements.txt         # Dependencies
```

## 🤖 **CÁC MODEL CHÍNH**

### 1. **PhoBERT** (BERT-based) 🏆
- **Models**: PhoBERT-base, PhoBERT-large
- **Location**: `models/phobert/checkpoints/`
- **Scripts**: `scripts/phobert/`
- **Best Performance**: PhoBERT-large (F1: 0.8290)

### 2. **LSTM Embedding** (Deep Learning)
- **Model**: LSTM với word embeddings
- **Performance**: F1-Score 0.7539
- **Approach**: Sequential neural network

### 3. **Vistral** (Large Language Model)
- **Models**: Few-shot fine-tune, Zero-shot fine-tune
- **Location**: `models/vistral/Vistral-v03/`
- **Scripts**: `scripts/vistral/`
- **Best Performance**: Few-shot (F1: 0.6898)

### 4. **TF-IDF + SVM** (Traditional ML)
- **Model**: TF-IDF Vectorizer + SVM Classifier
- **Location**: `models/tfidf_svm/models_tfidf_svm_v3/`
- **Scripts**: `scripts/tfidf_svm/`
- **Performance**: F1-Score 0.7426

## 🚀 **CÁCH SỬ DỤNG**

### **1. Chuẩn bị dữ liệu**
```bash
# Convert simple_dataset sang Alpaca format
python scripts/data_prep/convert_simple_to_alpaca_v2.py

# Data augmentation (optional)
python scripts/data_prep/data_augmentation_synonym_only.py
```

### **2. PhoBERT**
```bash
# Preprocess data
python scripts/phobert/preprocess_for_phobert.py

# Train model
python scripts/phobert/train_phobert.py

# Evaluate model
python scripts/phobert/evaluate_phobert.py \
    --model_path models/phobert/checkpoints/phobert-large-v2/latest_checkpoint.pth \
    --model_name vinai/phobert-large \
    --data_path models/phobert/data-bert-v2/phobert-large-v2/test_processed.pkl
```

### **3. TF-IDF SVM**
```bash
# Train model
python scripts/tfidf_svm/train_tfidf_svm.py --data_dir simple_dataset

# Demo model
python scripts/tfidf_svm/demo_tfidf_svm.py
```

### **4. LSTM Embedding**
```bash
# Train LSTM model
python scripts/lstm/train_lstm_enhanced.py

# Evaluate LSTM model  
python scripts/lstm/run_lstm_enhanced.py
```

### **5. Vistral**
```bash
# Fine-tune model
python scripts/vistral/fine_tune_clickbait_vistral_v03.py

# Test model
python scripts/vistral/test_base_vistral.py
```

## 📈 **HIỆU SUẤT MODELS**

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **PhoBERT-large** | **0.8281** | **0.8301** | **0.8281** | **0.8290** |
| PhoBERT-base | 0.8223 | 0.8254 | 0.8223 | 0.8235 |
| Vistral (few-shot) | 0.7832 | 0.8092 | 0.6719 | 0.6898 |
| LSTM Embedding | 0.7559 | 0.7524 | 0.7559 | 0.7539 |
| TF-IDF + SVM | 0.7422 | 0.7431 | 0.7422 | 0.7426 |
| Vistral (zero-shot) | 0.3145 | 0.4275 | 0.4929 | 0.2510 |

### 🏆 **Xếp hạng theo F1-Score:**
1. **PhoBERT-large**: 0.8290 ⭐
2. **PhoBERT-base**: 0.8235  
3. **LSTM Embedding**: 0.7539
4. **TF-IDF + SVM**: 0.7426
5. **Vistral (few-shot)**: 0.6898
6. **Vistral (zero-shot)**: 0.2510

### 📊 **Phân tích kết quả:**
- **🥇 PhoBERT models** dẫn đầu với F1-Score > 0.82, chứng tỏ sức mạnh của pre-trained BERT
- **🥈 LSTM Embedding** đạt kết quả tốt (0.7539) với approach đơn giản hơn
- **🥉 TF-IDF + SVM** traditional ML vẫn competitive với 0.7426  
- **⚡ Vistral few-shot** tốt hơn zero-shot đáng kể (0.6898 vs 0.2510)
- **📈 Precision cao nhất**: Vistral few-shot (0.8092) - ít false positive
- **📈 Recall cân bằng**: PhoBERT models có recall tốt nhất

## 📋 **YÊU CẦU HỆ THỐNG**

```bash
pip install -r requirements.txt
```

## 🏆 **ĐÁNH GIÁ MODELS**

Tất cả kết quả đánh giá được lưu trong `evaluation_results/`:
- Confusion matrices
- Classification reports  
- Detailed metrics (JSON)
- Performance comparisons

## 👥 **TEAM & CONTRIBUTION**

Dự án nghiên cứu phát hiện clickbait sử dụng multiple approaches:
- Deep Learning (PhoBERT)
- Traditional ML (TF-IDF + SVM)  
- Large Language Models (Vistral)

---
**Cập nhật cuối**: Đã tái cấu trúc và tối ưu hóa dự án (Jul 2024) 