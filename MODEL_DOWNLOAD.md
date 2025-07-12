# Hướng dẫn tải lại Models

## Các thư mục đã được xóa để giảm kích thước repository:

### 1. Model Weights (đã xóa):
- `checkpoints-vncorenlp/` - PhoBERT checkpoints (~36GB)
- `multilingual_bert_results/` - Multilingual BERT model (~11GB)  
- `xlm_roberta_results/` - XLM-RoBERTa model (~48GB)
- `models_lstm_enhanced/` - LSTM Enhanced model (~26MB)
- `VnCoreNLP/` - VnCoreNLP library và models (~114MB)
- `venv/` - Virtual environment (~6.1GB)

### 2. Các file model weights đã xóa:
- `*.pth` - PyTorch model files
- `*.safetensors` - SafeTensors model files  
- `*.pt` - PyTorch checkpoint files
- `*.bin` - Binary model files

## Cách tải lại models:

### 1. PhoBERT Models:
```bash
# Tạo thư mục
mkdir -p checkpoints-vncorenlp/phobert-large-v5
mkdir -p checkpoints-vncorenlp/phobert-large-v4
mkdir -p checkpoints-vncorenlp/phobert-large-v3
mkdir -p checkpoints-vncorenlp/phobert-large-v2
mkdir -p checkpoints-vncorenlp/phobert-base-v2

# Tải models từ Google Drive hoặc cloud storage
# (Cần upload models lên cloud storage trước)
```

### 2. Multilingual BERT:
```bash
mkdir -p multilingual_bert_results
# Tải model.safetensors và các checkpoint
```

### 3. XLM-RoBERTa:
```bash
mkdir -p xlm_roberta_results
# Tải model.safetensors và các checkpoint
```

### 4. LSTM Enhanced:
```bash
mkdir -p models_lstm_enhanced
# Tải best_model_enhanced.pth
```

### 5. VnCoreNLP:
```bash
# Clone repository VnCoreNLP
git clone https://github.com/vncorenlp/VnCoreNLP.git
# Hoặc tải từ trang chủ VnCoreNLP
```

### 6. Virtual Environment:
```bash
# Tạo virtual environment mới
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

## Lưu ý:
- Tổng kích thước models: ~101GB
- Repository hiện tại: ~91MB (không bao gồm .git)
- Các file config và kết quả evaluation vẫn được giữ lại
- File `.gitignore` đã được tạo để tránh commit models trong tương lai 