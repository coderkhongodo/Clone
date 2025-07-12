"""
Fine-tune Vistral-7B-Chat for Clickbait Detection - VISTRAL V03 (Simple Dataset)
Enhanced version với Viet-Mistral/Vistral-7B-Chat model, optimized for Vietnamese
Sử dụng simple_dataset đã convert sang Alpaca format V2
Training với epochs=5, higher LoRA rank, output to Vistral-v03
✨ ENHANCED 4-bit Quantization: NF4 + Double Quant (Memory usage ~4x smaller)
"""

import os
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import json
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, precision_recall_fscore_support
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling,
    GenerationConfig,
    default_data_collator,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset
from tqdm import tqdm
import re
import warnings
warnings.filterwarnings("ignore")

def load_alpaca_data():
    """
    Load dữ liệu từ Alpaca JSON format V2 (simple_dataset converted)
    """
    print("DATA: Bước 1: Load dữ liệu từ Alpaca V2 JSON files (simple_dataset)")
    
    try:
        # Load JSON files từ data_alpaca_v2 (adjust path for script location)
        base_path = "../../data_alpaca_v2" if os.path.exists("../../data_alpaca_v2") else "data_alpaca_v2"
        
        with open(f"{base_path}/train_alpaca.json", 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        
        with open(f"{base_path}/val_alpaca.json", 'r', encoding='utf-8') as f:
            val_data = json.load(f)
            
        with open(f"{base_path}/test_alpaca.json", 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        print(f"SUCCESS: Train: {len(train_data)} mẫu")
        print(f"SUCCESS: Val: {len(val_data)} mẫu")  
        print(f"SUCCESS: Test: {len(test_data)} mẫu")
        
        # Convert to DataFrame for easier processing
        def alpaca_to_df(alpaca_data):
            titles = []
            labels = []
            
            for item in alpaca_data:
                # Extract title from input (format V2: "Phân loại tiêu đề sau: TITLE")
                input_text = item['input']
                title = input_text.replace("Phân loại tiêu đề sau: ", "")
                # Fallback for old format
                if title == input_text:  # không match
                    title = input_text.replace("Tiêu đề: ", "")
                titles.append(title)
                
                # Extract label from output
                output_text = item['output']
                if "clickbait" in output_text.lower() and "non-clickbait" not in output_text.lower():
                    label = "clickbait"
                else:
                    label = "non-clickbait"
                labels.append(label)
            
            return pd.DataFrame({"title": titles, "label": labels})
        
        df_train = alpaca_to_df(train_data)
        df_val = alpaca_to_df(val_data)
        df_test = alpaca_to_df(test_data)
        
        # Kiểm tra class distribution
        print("\nSTATS: Phân bố class trong train:")
        print(df_train['label'].value_counts(normalize=True))
        print("\nSTATS: Phân bố class trong val:")
        print(df_val['label'].value_counts(normalize=True))
        print("\nSTATS: Phân bố class trong test:")
        print(df_test['label'].value_counts(normalize=True))
        
        return df_train, df_val, df_test, train_data, val_data, test_data
        
    except FileNotFoundError as e:
        print(f"ERROR: Error: {e}")
        print("TIP: Chạy convert_simple_to_alpaca_v2.py trước để tạo dữ liệu Alpaca V2")
        return None, None, None, None, None, None

def balance_dataset(df, dataset_name="dataset"):
    """
    Cân bằng dataset bằng oversampling
    """
    print(f"\nPROCESS: Cân bằng {dataset_name}")
    
    clickbait_samples = df[df['label'] == 'clickbait']
    non_clickbait_samples = df[df['label'] == 'non-clickbait']
    
    n_clickbait = len(clickbait_samples)
    n_non_clickbait = len(non_clickbait_samples)
    
    print(f"STATS: Ban đầu - Clickbait: {n_clickbait}, Non-clickbait: {n_non_clickbait}")
    
    if n_clickbait < n_non_clickbait:
        # Oversample clickbait
        repeat_factor = n_non_clickbait // n_clickbait
        remaining = n_non_clickbait % n_clickbait
        
        clickbait_balanced = pd.concat([clickbait_samples] * repeat_factor + 
                                     ([clickbait_samples.sample(remaining, random_state=42)] if remaining > 0 else []), 
                                     ignore_index=True)
        df_balanced = pd.concat([non_clickbait_samples, clickbait_balanced], ignore_index=True)
    else:
        # Oversample non-clickbait  
        repeat_factor = n_clickbait // n_non_clickbait
        remaining = n_clickbait % n_non_clickbait
        
        non_clickbait_balanced = pd.concat([non_clickbait_samples] * repeat_factor +
                                         ([non_clickbait_samples.sample(remaining, random_state=42)] if remaining > 0 else []),
                                         ignore_index=True)
        df_balanced = pd.concat([clickbait_samples, non_clickbait_balanced], ignore_index=True)
    
    # Shuffle
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"STATS: Sau balancing - Total: {len(df_balanced)}")
    print(df_balanced['label'].value_counts(normalize=True))
    
    return df_balanced

def create_vistral_datasets(train_data, val_data, test_data, tokenizer, max_length=512):
    """
    Tạo datasets từ Alpaca format cho training với Vistral chat format
    """
    print("\nLOG: Tạo Vistral datasets cho training")
    
    def format_vistral_training(item):
        """Format Alpaca item cho Vistral training"""
        instruction = item['instruction']
        input_text = item['input']
        output_text = item['output']
        
        # Vistral chat format - giống Mistral nhưng được tối ưu cho tiếng Việt
        prompt = f"<s>[INST] {instruction}\n\n{input_text} [/INST]{output_text}</s>"
        return prompt
    
    def tokenize_function(examples):
        # Tokenize full sequences with padding
        model_inputs = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors=None
        )
        # For causal LM, labels are the same as input_ids
        labels = []
        for input_ids in model_inputs["input_ids"]:
            label = input_ids.copy()
            # Set padding tokens to -100 so they're ignored in loss calculation
            for i, token_id in enumerate(label):
                if token_id == tokenizer.pad_token_id:
                    label[i] = -100
            labels.append(label)
        
        model_inputs["labels"] = labels
        return model_inputs
    
    def prepare_dataset(alpaca_data):
        # Format examples cho generation
        formatted_texts = []
        for item in alpaca_data:
            formatted_text = format_vistral_training(item)
            formatted_texts.append(formatted_text)
        
        dataset_dict = {"text": formatted_texts}
        dataset = Dataset.from_dict(dataset_dict)
        dataset = dataset.map(
            tokenize_function, 
            batched=True,
            remove_columns=["text"]
        )
        return dataset
    
    train_dataset = prepare_dataset(train_data)
    val_dataset = prepare_dataset(val_data)
    test_dataset = prepare_dataset(test_data)
    
    print(f"SUCCESS: Train dataset: {len(train_dataset)} mẫu")
    print(f"SUCCESS: Val dataset: {len(val_dataset)} mẫu") 
    print(f"SUCCESS: Test dataset: {len(test_dataset)} mẫu")
    
    return train_dataset, val_dataset, test_dataset

def setup_vistral_model_v03():
    """
    Setup Vistral model V03 với enhanced parameters cho Vietnamese
    """
    print("\n🇻🇳 Setup Vistral V03 - Vietnamese Optimized Model")
    
    model_name = "Viet-Mistral/Vistral-7B-Chat"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # ENHANCED 4-bit quantization config để giảm memory ~4x
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,      # Double quantization để tiết kiệm thêm memory
        bnb_4bit_quant_type="nf4",           # NormalFloat4 - optimal cho neural networks
        bnb_4bit_compute_dtype=torch.bfloat16  # Compute dtype cho inference
    )
    
    # Load model for causal language modeling với optimized 4-bit
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config,      # Sử dụng BitsAndBytesConfig thay vì load_in_4bit=True
        trust_remote_code=True
    )
    
    # Make sure pad token is set correctly
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # ENHANCED LoRA configuration V03 for Vietnamese
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,                          # High rank for better adaptation
        lora_alpha=64,                 # High alpha for strong adaptation
        lora_dropout=0.1,              # Dropout for regularization
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    
    # Apply LoRA
    model = get_peft_model(model, peft_config)
    
    print("SUCCESS: Vistral V03 và Enhanced LoRA setup hoàn tất")
    print(f"STATS: Trainable parameters: {model.num_parameters(only_trainable=True):,}")
    print(f"STATS: Total parameters: {model.num_parameters():,}")
    print(f"CONFIG: LoRA Rank: 32 (enhanced)")
    print(f"CONFIG: LoRA Alpha: 64 (enhanced)")
    print(f"⚡ 4-bit Quantization: NF4 + Double Quant (Memory ~4x smaller)")
    print(f"🇻🇳 Base Model: Viet-Mistral/Vistral-7B-Chat (Vietnamese Optimized)")
    
    return model, tokenizer

def train_vistral_v03(model, tokenizer, train_dataset, val_dataset):
    """
    Train Vistral V03 với enhanced parameters for Vietnamese
    """
    print("\nSTART: Bắt đầu training Vistral V03 - Vietnamese Enhanced")
    
    # Use default data collator
    data_collator = default_data_collator
    
    # ENHANCED Training arguments V03 for Vistral
    training_args = TrainingArguments(
        output_dir="./Vistral-v03",              # NEW output directory for Vistral
        num_train_epochs=5,                      # 5 epochs for better convergence
        per_device_train_batch_size=1,           # Memory optimized
        per_device_eval_batch_size=2,            # Evaluation batch size
        gradient_accumulation_steps=32,          # Effective batch size = 32
        warmup_steps=150,                        # Warmup for stable training
        learning_rate=1.5e-5,                    # Slightly lower for stability
        weight_decay=0.01,                       # L2 regularization
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=5,                      # Keep more checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,
        bf16=True,                               # Use bf16 for better precision
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to="none",
        seed=42,
        dataloader_num_workers=0,
    )
    
    # Enhanced early stopping
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.001
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[early_stopping],
    )
    
    # Train
    print("🔥 Training Vistral V03 started...")
    print(f"TARGET: Epochs: 5")
    print(f"TARGET: LoRA Rank: 32")
    print(f"TARGET: Output: ./Vistral-v03/")
    print(f"🇻🇳 Model: Viet-Mistral/Vistral-7B-Chat")
    
    trainer.train()
    
    print("SUCCESS: Training Vistral V03 completed!")
    return trainer

def test_vistral_prompting(model, tokenizer, test_data, sample_size=None):
    """
    Test Vistral model với Simple Few-Shot prompting - 4 ví dụ tổng (như ban đầu)
    """
    print(f"\nTEST: Testing Vistral V03 Model - Simple Few-Shot (4 examples total)")
    
    if sample_size:
        test_data = test_data[:sample_size]
    
    print(f"STATS: Testing trên {len(test_data)} samples")
    
    predictions = []
    actual_labels = []
    
    for idx, item in tqdm(enumerate(test_data), total=len(test_data), desc="Vistral Simple Few-Shot Testing"):
        try:
            # Create simple Vietnamese prompt
            input_text = item['input']
            simple_instruction = """Phân loại tiêu đề: clickbait hoặc non-clickbait

VÍ DỤ:
- "Bí mật không ai dám nói về ngôi sao này" → clickbait
- "Sự thật gây sốc về món ăn yêu thích" → clickbait  
- "Chính phủ công bố báo cáo kinh tế quý 3" → non-clickbait
- "Giá xăng tăng 500 đồng từ ngày mai" → non-clickbait

Chỉ trả lời MỘT TỪ: clickbait hoặc non-clickbait"""
            
            prompt = f"<s>[INST] {simple_instruction}\n\n{input_text}\n\nTrả lời: [/INST]"
            
            # Generate với parameters tối ưu cho tiếng Việt
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,             # Đủ để generate full "non-clickbait" (có thể 4-6 tokens)
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1,       # Tránh lặp lại
                )
            
            # Decode prediction
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text.split("[/INST]")[-1].strip()
            
            # Enhanced parsing cho tiếng Việt
            prediction = parse_vistral_response(response)
            predictions.append(prediction)
            
            # Extract actual label
            output_text = item['output']
            if "clickbait" in output_text.lower() and "non-clickbait" not in output_text.lower():
                actual = "clickbait"
            else:
                actual = "non-clickbait"
            actual_labels.append(actual)
            
            # Debug first few
            if idx < 3:
                print(f"\nVistral Simple Few-Shot Example {idx+1}:")
                print(f"Input: {input_text}")
                print(f"Generated: '{response}'")
                print(f"Predicted: {prediction}")
                print(f"Actual: {actual}")
                print(f"Match: {'SUCCESS:' if prediction == actual else 'ERROR:'}")
                
        except Exception as e:
            print(f"ERROR: Error processing sample {idx}: {e}")
            predictions.append("non-clickbait")
            actual_labels.append("non-clickbait")
    
    return predictions, actual_labels

def parse_vistral_response(response):
    """
    Enhanced parsing cho Vistral response - handle longer generation
    """
    text = response.lower().strip()
    
    # Direct matches (exact)
    if text == "clickbait":
        return "clickbait"
    elif text == "non-clickbait":
        return "non-clickbait"
    
    # Contains check - ưu tiên non-clickbait trước vì nó dài hơn
    if "non-clickbait" in text:
        return "non-clickbait"
    elif "clickbait" in text:
        return "clickbait"
    
    # Check từng word trong response
    words = text.split()
    for word in words:
        if word == "clickbait":
            return "clickbait"
        elif word == "non-clickbait":
            return "non-clickbait"
        # Handle incomplete words
        elif word.startswith("clickb") and len(word) >= 5:
            return "clickbait" 
        elif word.startswith("non-") and len(word) >= 4:
            return "non-clickbait"
    
    # Check first word cho partial matches
    if words:
        first_word = words[0]
        if "clickbait" in first_word and "non" not in first_word:
            return "clickbait"
        elif "non" in first_word or first_word in ["không", "no"]:
            return "non-clickbait"
    
    # Default fallback
    return "non-clickbait"

def test_vistral_zero_shot(model, tokenizer, test_data, sample_size=None):
    """
    Test với Simple Zero Shot prompting - định nghĩa đơn giản như ban đầu
    """
    print(f"\nTARGET: Testing với VISTRAL SIMPLE ZERO-SHOT - Basic Definitions")
    
    if sample_size:
        test_data = test_data[:sample_size]
    
    print(f"STATS: Testing trên {len(test_data)} samples")
    
    predictions = []
    actual_labels = []
    
    for idx, item in tqdm(enumerate(test_data), total=len(test_data), desc="Vistral Simple Zero-Shot Testing"):
        try:
            input_text = item['input']
            
            # Simple Zero Shot Prompt - trở lại phiên bản đơn giản
            zero_shot_prompt = f"""<s>[INST] Phân loại tiêu đề tin tức: clickbait hoặc non-clickbait

CLICKBAIT: Tiêu đề câu view, từ ngữ cảm xúc mạnh, tạo tò mò quá mức
NON-CLICKBAIT: Tiêu đề thông tin rõ ràng, khách quan, cụ thể

{input_text}

Chỉ trả lời MỘT TỪ: [/INST]"""
            
            # Generate
            inputs = tokenizer(zero_shot_prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,             # Consistency với few-shot: đủ để generate full word
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1,       # Tránh lặp lại
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text.split("[/INST]")[-1].strip()
            
            # Simple parsing
            prediction = parse_simple_viet_response(response)
            predictions.append(prediction)
            
            # Extract actual label
            output_text = item['output']
            if "clickbait" in output_text.lower() and "non-clickbait" not in output_text.lower():
                actual = "clickbait"
            else:
                actual = "non-clickbait"
            actual_labels.append(actual)
            
            # Debug first few
            if idx < 3:
                print(f"\nVistral Simple Zero-Shot Example {idx+1}:")
                print(f"Input: {input_text}")
                print(f"Generated: '{response}'")
                print(f"Predicted: {prediction}")
                print(f"Actual: {actual}")
                print(f"Match: {'SUCCESS:' if prediction == actual else 'ERROR:'}")
                
        except Exception as e:
            print(f"ERROR: Error processing sample {idx}: {e}")
            predictions.append("non-clickbait")
            actual_labels.append("non-clickbait")
    
    return predictions, actual_labels

def parse_simple_viet_response(response):
    """
    Enhanced parsing cho Vietnamese responses - consistency với parse_vistral_response
    """
    text = response.lower().strip()
    
    # Direct matches (exact)
    if text == "clickbait":
        return "clickbait"
    elif text == "non-clickbait":
        return "non-clickbait"
    
    # Contains check - ưu tiên non-clickbait trước
    if "non-clickbait" in text:
        return "non-clickbait"
    elif "clickbait" in text:
        return "clickbait"
    
    # Check từng word trong response
    words = text.split()
    for word in words:
        if word == "clickbait":
            return "clickbait"
        elif word == "non-clickbait":
            return "non-clickbait"
        # Handle incomplete words
        elif word.startswith("clickb") and len(word) >= 5:
            return "clickbait" 
        elif word.startswith("non-") and len(word) >= 4:
            return "non-clickbait"
    
    # Check first word cho partial matches
    if words:
        first_word = words[0]
        if "clickbait" in first_word and "non" not in first_word:
            return "clickbait"
        elif "non" in first_word or first_word in ["không", "no"]:
            return "non-clickbait"
    
    # Default
    return "non-clickbait"

def evaluate_vistral_results(predictions, actual_labels, method_name):
    """
    Đánh giá kết quả Vistral methods với comprehensive metrics
    """
    print(f"\nSTATS: {method_name.upper()} EVALUATION RESULTS - VISTRAL V03")
    print("=" * 70)
    
    # Calculate basic metrics
    acc = accuracy_score(actual_labels, predictions)
    f1_macro = f1_score(actual_labels, predictions, average='macro')
    f1_weighted = f1_score(actual_labels, predictions, average='weighted')
    
    # Calculate per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        actual_labels, predictions, 
        labels=['clickbait', 'non-clickbait'],
        average=None
    )
    
    clickbait_precision = precision[0]
    clickbait_f1 = f1[0]
    non_clickbait_precision = precision[1] 
    non_clickbait_f1 = f1[1]
    
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Macro: {f1_macro:.4f}")
    print(f"F1-Weighted: {f1_weighted:.4f}")
    print(f"Clickbait - Precision: {clickbait_precision:.4f}, F1: {clickbait_f1:.4f}")
    print(f"Non-clickbait - Precision: {non_clickbait_precision:.4f}, F1: {non_clickbait_f1:.4f}")
    
    # Detailed classification report
    print("\n📋 Detailed Classification Report:")
    report = classification_report(
        actual_labels, predictions,
        target_names=['clickbait', 'non-clickbait'],
        digits=4
    )
    print(report)
    
    # Comprehensive metrics table
    print("\n" + "="*100)
    print("📊 COMPREHENSIVE EVALUATION METRICS")
    print("="*100)
    
    # Table header
    print(f"{'Model':<25} {'Clickbait':<20} {'Non-Clickbait':<20} {'Macro F1':<10} {'Weighted F1':<12} {'Accuracy':<10}")
    print(f"{'':<25} {'Precision':<10} {'F1':<10} {'Precision':<10} {'F1':<10} {'':<10} {'':<12} {'':<10}")
    print("-" * 100)
    
    # Data row
    data_row = (f"{method_name:<25} "
               f"{clickbait_precision:<10.4f} {clickbait_f1:<10.4f} "
               f"{non_clickbait_precision:<10.4f} {non_clickbait_f1:<10.4f} "
               f"{f1_macro:<10.4f} {f1_weighted:<12.4f} {acc:<10.4f}")
    print(data_row)
    print("="*100)
    
    # Confusion Matrix
    cm = confusion_matrix(actual_labels, predictions, labels=['clickbait', 'non-clickbait'])
    print(f"\n📉 Confusion Matrix:")
    print("         Pred: clickbait  non-clickbait")
    print(f"True clickbait:      {cm[0][0]:3d}         {cm[0][1]:3d}")
    print(f"True non-clickbait:  {cm[1][0]:3d}         {cm[1][1]:3d}")
    
    # Save results với comprehensive metrics
    results_df = pd.DataFrame({
        'actual': actual_labels,
        'predicted': predictions
    })
    
    # Save detailed metrics
    detailed_metrics = {
        'method': method_name,
        'accuracy': acc,
        'macro_f1': f1_macro,
        'weighted_f1': f1_weighted,
        'clickbait_precision': clickbait_precision,
        'clickbait_f1': clickbait_f1,
        'non_clickbait_precision': non_clickbait_precision,
        'non_clickbait_f1': non_clickbait_f1,
        'confusion_matrix': cm.tolist()
    }
    
    filename = f"vistral_v03_{method_name.lower().replace(' ', '_')}_results.csv"
    metrics_filename = f"vistral_v03_{method_name.lower().replace(' ', '_')}_metrics.json"
    
    results_df.to_csv(filename, index=False)
    with open(metrics_filename, 'w', encoding='utf-8') as f:
        json.dump(detailed_metrics, f, indent=2, ensure_ascii=False)
    
    print(f"SAVE: Results saved to {filename}")
    print(f"SAVE: Detailed metrics saved to {metrics_filename}")
    
    return acc, f1_macro, f1_weighted, detailed_metrics

def main(mode="train"):
    """
    Main function cho Vistral V03
    """
    if mode == "train":
        print("START: Fine-tune Vistral-7B-Chat V03 - VIETNAMESE OPTIMIZED")
        print("🔥 Mode: TRAINING WITH VIETNAMESE ENHANCED PARAMETERS")
        print("🇻🇳 Model: Viet-Mistral/Vistral-7B-Chat")
        print("=" * 80)
        
        try:
            # Load Alpaca data
            df_train, df_val, df_test, train_data, val_data, test_data = load_alpaca_data()
            if df_train is None:
                return
            
            # Skip balancing - use original imbalanced data
            print(f"\nSTATS: Sử dụng dữ liệu gốc (không cân bằng)")
            print(f"🔢 Train - Clickbait: {len(df_train[df_train['label'] == 'clickbait'])}, Non-clickbait: {len(df_train[df_train['label'] == 'non-clickbait'])}")
            print(f"🔢 Val - Clickbait: {len(df_val[df_val['label'] == 'clickbait'])}, Non-clickbait: {len(df_val[df_val['label'] == 'non-clickbait'])}")
            
            # Keep original dataframes (no balancing)
            df_train_balanced = df_train
            df_val_balanced = df_val
            
            # Setup Vistral model V03
            model, tokenizer = setup_vistral_model_v03()
            
            # Create Vistral datasets
            train_dataset, val_dataset, test_dataset = create_vistral_datasets(
                train_data, val_data, test_data, tokenizer
            )
            
            # Train Vistral V03
            trainer = train_vistral_v03(model, tokenizer, train_dataset, val_dataset)
            
            print(f"\nCOMPLETE: VISTRAL V03 TRAINING COMPLETED!")
            print("TIP: Để test Vistral V03, chạy: python fine_tune_clickbait_vistral_v03.py --mode test")
            
        except Exception as e:
            print(f"ERROR: Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    elif mode == "test":
        print("START: Testing Vistral V03 Model với Vietnamese-Optimized Prompting")
        print("🔥 Mode: COMPREHENSIVE VIETNAMESE TESTING")
        print("🇻🇳 Model: Viet-Mistral/Vistral-7B-Chat")
        print("=" * 80)
        
        try:
            # Load test data
            _, _, _, _, _, test_data = load_alpaca_data()
            if test_data is None:
                return
            
            # Load trained Vistral model
            model_path = "/home/huflit/NCKH/models/vistral/Vistral-v03/checkpoint-375"
            
            if not os.path.exists(model_path):
                print(f"ERROR: Không tìm thấy Vistral V03 model tại {model_path}!")
                print("TIP: Kiểm tra đường dẫn model")
                return
            
            print(f"PROCESS: Loading Vistral V03 từ {model_path}")
            
            # Load tokenizer và model
            tokenizer = AutoTokenizer.from_pretrained("Viet-Mistral/Vistral-7B-Chat")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # ENHANCED 4-bit quantization config cho testing
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
            base_model = AutoModelForCausalLM.from_pretrained(
                "Viet-Mistral/Vistral-7B-Chat",
                torch_dtype=torch.bfloat16,
                device_map="auto",
                quantization_config=bnb_config,      # Sử dụng optimized 4-bit config
                trust_remote_code=True
            )
            
            model = PeftModel.from_pretrained(base_model, model_path)
            
            print("SUCCESS: Vistral V03 loaded successfully!")
            
            # Test với Vietnamese-optimized prompting methods
            results = {}
            
            print(f"\n{'='*80}")
            print("TEST: TESTING 1: VISTRAL SIMPLE FEW-SHOT PROMPTING")
            print(f"{'='*80}")
            predictions_vistral, actual_labels = test_vistral_prompting(model, tokenizer, test_data)
            acc_vistral, f1_vistral, f1_weighted_vistral, metrics_vistral = evaluate_vistral_results(predictions_vistral, actual_labels, "Vistral Simple Few-Shot")
            results['Vistral Simple Few-Shot'] = (acc_vistral, f1_vistral, f1_weighted_vistral, metrics_vistral)
            
            print(f"\n{'='*80}")
            print("TEST: TESTING 2: VISTRAL SIMPLE ZERO-SHOT PROMPTING")
            print(f"{'='*80}")
            predictions_zero, actual_labels = test_vistral_zero_shot(model, tokenizer, test_data)
            acc_zero, f1_zero, f1_weighted_zero, metrics_zero = evaluate_vistral_results(predictions_zero, actual_labels, "Vistral Simple Zero-Shot")
            results['Vistral Simple Zero-Shot'] = (acc_zero, f1_zero, f1_weighted_zero, metrics_zero)
            
            # Final Comprehensive Comparison
            print(f"\n{'='*100}")
            print("📊 FINAL COMPREHENSIVE COMPARISON - VISTRAL V03 VIETNAMESE MODEL")
            print(f"{'='*100}")
            
            # Table header
            print(f"{'Method':<25} {'Clickbait':<20} {'Non-Clickbait':<20} {'Macro F1':<10} {'Weighted F1':<12} {'Accuracy':<10}")
            print(f"{'':<25} {'Precision':<10} {'F1':<10} {'Precision':<10} {'F1':<10} {'':<10} {'':<12} {'':<10}")
            print("-" * 100)
            
            # Data rows
            for method, (acc, f1_macro, f1_weighted, metrics) in results.items():
                data_row = (f"{method:<25} "
                           f"{metrics['clickbait_precision']:<10.4f} {metrics['clickbait_f1']:<10.4f} "
                           f"{metrics['non_clickbait_precision']:<10.4f} {metrics['non_clickbait_f1']:<10.4f} "
                           f"{f1_macro:<10.4f} {f1_weighted:<12.4f} {acc:<10.4f}")
                print(data_row)
            
            print("="*100)
            
            # Best method
            best_method = max(results.items(), key=lambda x: x[1][2])  # Sort by weighted F1
            print(f"\n🥇 BEST VISTRAL METHOD: {best_method[0]}")
            print(f"   Accuracy: {best_method[1][0]:.4f}")
            print(f"   Macro F1: {best_method[1][1]:.4f}")
            print(f"   Weighted F1: {best_method[1][2]:.4f}")
            print(f"   Clickbait F1: {best_method[1][3]['clickbait_f1']:.4f}")
            print(f"   Non-clickbait F1: {best_method[1][3]['non_clickbait_f1']:.4f}")
            print(f"🇻🇳 Vietnamese-optimized model performance achieved!")
            
        except Exception as e:
            print(f"ERROR: Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    else:
        print("ERROR: Invalid mode. Use 'train' or 'test'")

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    if len(sys.argv) > 1 and "--mode" in sys.argv:
        mode_index = sys.argv.index("--mode") + 1
        if mode_index < len(sys.argv):
            mode = sys.argv[mode_index]
        else:
            mode = "train"
    else:
        mode = "train"
    
    main(mode) 