# So sánh XLM-RoBERTa vs Multilingual BERT cho Clickbait Detection

## Tổng quan

Đã tạo thành công script mới `demo_multilingual_bert.py` để thay thế script XLM-RoBERTa hiện tại. Dưới đây là so sánh chi tiết giữa hai mô hình:

## Sự khác biệt chính

### 1. Model Architecture
| Đặc điểm | XLM-RoBERTa | Multilingual BERT |
|----------|-------------|-------------------|
| **Base Model** | `xlm-roberta-base` | `bert-base-multilingual-cased` |
| **Kiến trúc** | RoBERTa (Robustly Optimized BERT) | BERT (Bidirectional Encoder Representations) |
| **Tokenizer** | SentencePiece | WordPiece |
| **Vocab Size** | ~250K tokens | ~119K tokens |
| **Max Length mặc định** | 256 tokens | 512 tokens |

### 2. Hyperparameters được điều chỉnh
| Tham số | XLM-RoBERTa | Multilingual BERT | Lý do thay đổi |
|---------|-------------|-------------------|----------------|
| **max_length** | 256 | 512 | BERT hỗ trợ tốt hơn với sequence dài |
| **batch_size** | 16 | 8 | Multilingual BERT lớn hơn, cần memory nhiều hơn |
| **output_dir** | `./xlm_roberta_results` | `./multilingual_bert_results` | Tránh conflict |

### 3. Điểm mạnh của từng model

#### XLM-RoBERTa
- ✅ **Training hiệu quả hơn**: Bỏ Next Sentence Prediction task
- ✅ **Robust hơn**: Training với dynamic masking
- ✅ **Performance tốt**: Thường đạt điểm cao hơn trên nhiều tasks
- ✅ **Memory efficient**: Batch size lớn hơn với cùng memory

#### Multilingual BERT
- ✅ **Stable và proven**: Model đã được test rộng rãi
- ✅ **Good multilingual capability**: Hỗ trợ 104 ngôn ngữ
- ✅ **Longer sequences**: Max length 512 tokens
- ✅ **Community support**: Nhiều pre-trained models và resources

## Cách sử dụng

### Chạy Multilingual BERT

```bash
# Training mode
python demo_multilingual_bert.py --mode train --data_path simple_dataset

# Interactive mode
python demo_multilingual_bert.py --mode interactive --output_dir ./multilingual_bert_results

# Examples mode
python demo_multilingual_bert.py --mode examples --output_dir ./multilingual_bert_results
```

### Parameters quan trọng

```bash
# Custom parameters
python demo_multilingual_bert.py \
    --mode train \
    --model_name bert-base-multilingual-cased \
    --max_length 512 \
    --batch_size 8 \
    --epochs 3 \
    --learning_rate 2e-5 \
    --output_dir ./multilingual_bert_results
```

## So sánh Performance dự kiến

### Memory Usage
- **XLM-RoBERTa**: ~3-4GB VRAM (batch_size=16, max_length=256)
- **Multilingual BERT**: ~6-8GB VRAM (batch_size=8, max_length=512)

### Training Time
- **XLM-RoBERTa**: Nhanh hơn ~20-30% do batch size lớn hơn
- **Multilingual BERT**: Chậm hơn nhưng có thể xử lý text dài hơn

### Accuracy dự kiến
- **XLM-RoBERTa**: 85-90% (thường cao hơn một chút)
- **Multilingual BERT**: 82-88% (stable và consistent)

## Khuyến nghị

### Chọn XLM-RoBERTa nếu:
- ✅ Muốn performance cao nhất
- ✅ Resource bị hạn chế (GPU memory)
- ✅ Text ngắn (< 256 tokens)
- ✅ Cần training nhanh

### Chọn Multilingual BERT nếu:
- ✅ Muốn stability và proven results
- ✅ Text dài (> 256 tokens)
- ✅ Cần compatibility với ecosystem BERT
- ✅ Muốn nhiều pre-trained options

## Migration từ XLM-RoBERTa

Để migrate từ script cũ sang script mới:

1. **Backup results cũ**:
```bash
mv xlm_roberta_results xlm_roberta_results_backup
```

2. **Chạy script mới**:
```bash
python demo_multilingual_bert.py --mode train
```

3. **So sánh results**:
```bash
# Compare JSON results
diff xlm_roberta_results_backup/test_results_detailed.json multilingual_bert_results/test_results_detailed.json
```

## Các files được tạo

### Multilingual BERT outputs:
- `multilingual_bert_results/model.safetensors` - Trained model
- `multilingual_bert_results/test_results_detailed.json` - Metrics chi tiết
- `multilingual_bert_results/evaluation_report.txt` - Báo cáo text
- `multilingual_bert_results/confusion_matrix_detailed.png` - Confusion matrix
- `multilingual_bert_results/tokenizer_config.json` - Tokenizer config

### Script comparison:
- `demo_xlm_roberta.py` - Script gốc
- `demo_multilingual_bert.py` - Script mới
- `model_comparison.md` - File so sánh này

## Kết luận

Cả hai mô hình đều phù hợp cho clickbait detection. Multilingual BERT cung cấp stability và compatibility tốt, trong khi XLM-RoBERTa có performance cao hơn một chút. Lựa chọn tùy thuộc vào requirements cụ thể của project. 