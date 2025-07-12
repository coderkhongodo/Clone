"""
Test Base Vistral-7B-Chat Model (Chưa Fine-tune) - Clickbait Detection V2
Sử dụng simple_dataset đã được convert sang Alpaca format
So sánh hiệu suất với model đã fine-tune
"""

import os
import pandas as pd
import torch
import numpy as np
import json
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def load_test_data():
    """
    Load dữ liệu test từ Alpaca JSON format V2 (simple_dataset converted)
    """
    print("Load dữ liệu test từ Alpaca V2 JSON files (simple_dataset)")
    
    try:
        with open("data_alpaca_v2/test_alpaca.json", 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        print(f"Test: {len(test_data)} mẫu")
        
        # Show data distribution
        labels = []
        for item in test_data:
            if "clickbait" in item['output'].lower() and "non-clickbait" not in item['output'].lower():
                labels.append("clickbait")
            else:
                labels.append("non-clickbait")
        
        from collections import Counter
        label_counts = Counter(labels)
        total = len(labels)
        print(f"Test data distribution:")
        for label, count in label_counts.items():
            print(f"   {label}: {count} ({count/total*100:.1f}%)")
        
        return test_data
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Chạy convert_simple_to_alpaca_v2.py trước để tạo dữ liệu Alpaca V2")
        return None

def setup_base_vistral_model():
    """
    Setup Base Vistral model (chưa fine-tune)
    """
    print("\nSetup Base Vistral Model - Chưa Fine-tune")
    
    model_name = "Viet-Mistral/Vistral-7B-Chat"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 4-bit quantization config để giảm memory
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True
    )
    
    model.config.pad_token_id = tokenizer.pad_token_id
    
    print("Base Vistral Model setup hoàn tất")
    print(f"Base Model: Viet-Mistral/Vistral-7B-Chat (Chưa Fine-tune)")
    print(f"4-bit Quantization: NF4 + Double Quant")
    
    return model, tokenizer

def setup_finetuned_vistral_model():
    """
    Setup Fine-tuned Vistral model
    """
    print("\nSetup Fine-tuned Vistral Model")
    
    # Use specific fine-tuned model path
    model_path = "/home/huflit/NCKH/Vistral-v03/checkpoint-375"
    
    if not os.path.exists(model_path):
        print(f"Không tìm thấy fine-tuned model tại: {model_path}")
        print("Kiểm tra lại đường dẫn model")
        return None, None
    
    print(f"Sử dụng model: {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load fine-tuned model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True
    )
    
    model.config.pad_token_id = tokenizer.pad_token_id
    
    print("Fine-tuned Vistral Model setup hoàn tất")
    print(f"Fine-tuned Model: {model_path}")
    print(f"4-bit Quantization: NF4 + Double Quant")
    
    return model, tokenizer

def test_base_vistral_few_shot(model, tokenizer, test_data, sample_size=None):
    """
    Test Base Vistral model với Simple Few-Shot prompting
    """
    print(f"\nTesting Base Vistral Model - Simple Few-Shot")
    
    if sample_size:
        test_data = test_data[:sample_size]
    
    print(f"Testing trên {len(test_data)} samples")
    
    predictions = []
    actual_labels = []
    
    for idx, item in tqdm(enumerate(test_data), total=len(test_data), desc="Base Vistral Few-Shot Testing"):
        try:
            input_text = item['input']
            
            # Simple Few-Shot prompt
            few_shot_prompt = f"""<s>[INST] Phân loại tiêu đề: clickbait hoặc non-clickbait

Ví dụ:
"Bí mật mà không ai dám tiết lộ" → clickbait
"Thủ tướng ký quyết định tăng lương tối thiểu" → non-clickbait
"Sự thật gây sốc về ngôi sao này" → clickbait
"Giá xăng tăng 500 đồng từ ngày mai" → non-clickbait

Tiêu đề: {input_text}

Trả lời: [/INST]"""
            
            # Generate
            inputs = tokenizer(few_shot_prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text.split("[/INST]")[-1].strip()
            
            # Simple parsing
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
                print(f"\nBase Vistral Few-Shot Example {idx+1}:")
                print(f"Input: {input_text}")
                print(f"Generated: '{response}'")
                print(f"Predicted: {prediction}")
                print(f"Actual: {actual}")
                print(f"Match: {'YES' if prediction == actual else 'NO'}")
                
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            predictions.append("non-clickbait")
            actual_labels.append("non-clickbait")
    
    return predictions, actual_labels

def test_base_vistral_zero_shot(model, tokenizer, test_data, sample_size=None):
    """
    Test Base Vistral với Simple Zero Shot prompting
    """
    print(f"\nTesting Base Vistral với Simple Zero-Shot")
    
    if sample_size:
        test_data = test_data[:sample_size]
    
    print(f"Testing trên {len(test_data)} samples")
    
    predictions = []
    actual_labels = []
    
    for idx, item in tqdm(enumerate(test_data), total=len(test_data), desc="Base Vistral Zero-Shot Testing"):
        try:
            input_text = item['input']
            
            # Simple Zero Shot Prompt
            zero_shot_prompt = f"""<s>[INST] Phân loại tiêu đề sau đây là clickbait hoặc non-clickbait:

{input_text}

Trả lời: [/INST]"""
            
            # Generate
            inputs = tokenizer(zero_shot_prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text.split("[/INST]")[-1].strip()
            
            # Simple parsing
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
                print(f"\nBase Vistral Zero-Shot Example {idx+1}:")
                print(f"Input: {input_text}")
                print(f"Generated: '{response}'")
                print(f"Predicted: {prediction}")
                print(f"Actual: {actual}")
                print(f"Match: {'YES' if prediction == actual else 'NO'}")
                
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            predictions.append("non-clickbait")
            actual_labels.append("non-clickbait")
    
    return predictions, actual_labels

def test_finetuned_vistral_few_shot(model, tokenizer, test_data, sample_size=None):
    """
    Test Fine-tuned Vistral model với Simple Few-Shot prompting
    """
    print(f"\nTesting Fine-tuned Vistral Model - Simple Few-Shot")
    
    if sample_size:
        test_data = test_data[:sample_size]
    
    print(f"Testing trên {len(test_data)} samples")
    
    predictions = []
    actual_labels = []
    
    for idx, item in tqdm(enumerate(test_data), total=len(test_data), desc="Fine-tuned Vistral Few-Shot Testing"):
        try:
            input_text = item['input']
            
            # Simple Few-Shot prompt
            few_shot_prompt = f"""<s>[INST] Phân loại tiêu đề: clickbait hoặc non-clickbait

Ví dụ:
"Bí mật mà không ai dám tiết lộ" → clickbait
"Thủ tướng ký quyết định tăng lương tối thiểu" → non-clickbait
"Sự thật gây sốc về ngôi sao này" → clickbait
"Giá xăng tăng 500 đồng từ ngày mai" → non-clickbait

Tiêu đề: {input_text}

Trả lời: [/INST]"""
            
            # Generate
            inputs = tokenizer(few_shot_prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text.split("[/INST]")[-1].strip()
            
            # Simple parsing
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
                print(f"\nFine-tuned Vistral Few-Shot Example {idx+1}:")
                print(f"Input: {input_text}")
                print(f"Generated: '{response}'")
                print(f"Predicted: {prediction}")
                print(f"Actual: {actual}")
                print(f"Match: {'YES' if prediction == actual else 'NO'}")
                
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            predictions.append("non-clickbait")
            actual_labels.append("non-clickbait")
    
    return predictions, actual_labels

def test_finetuned_vistral_zero_shot(model, tokenizer, test_data, sample_size=None):
    """
    Test Fine-tuned Vistral với Simple Zero Shot prompting
    """
    print(f"\nTesting Fine-tuned Vistral với Simple Zero-Shot")
    
    if sample_size:
        test_data = test_data[:sample_size]
    
    print(f"Testing trên {len(test_data)} samples")
    
    predictions = []
    actual_labels = []
    
    for idx, item in tqdm(enumerate(test_data), total=len(test_data), desc="Fine-tuned Vistral Zero-Shot Testing"):
        try:
            input_text = item['input']
            
            # Simple Zero Shot Prompt
            zero_shot_prompt = f"""<s>[INST] Phân loại tiêu đề sau đây là clickbait hoặc non-clickbait:

{input_text}

Trả lời: [/INST]"""
            
            # Generate
            inputs = tokenizer(zero_shot_prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text.split("[/INST]")[-1].strip()
            
            # Simple parsing
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
                print(f"\nFine-tuned Vistral Zero-Shot Example {idx+1}:")
                print(f"Input: {input_text}")
                print(f"Generated: '{response}'")
                print(f"Predicted: {prediction}")
                print(f"Actual: {actual}")
                print(f"Match: {'YES' if prediction == actual else 'NO'}")
                
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            predictions.append("non-clickbait")
            actual_labels.append("non-clickbait")
    
    return predictions, actual_labels

def parse_vistral_response(response):
    """
    Simple parsing cho Vistral response
    """
    text = response.lower().strip()
    
    # Direct matches
    if text == "clickbait":
        return "clickbait"
    elif text == "non-clickbait":
        return "non-clickbait"
    
    # Contains check
    if "non-clickbait" in text:
        return "non-clickbait"
    elif "clickbait" in text:
        return "clickbait"
    
    # Default fallback
    return "non-clickbait"

def evaluate_results(predictions, actual_labels, method_name):
    """
    Đánh giá kết quả
    """
    print(f"\n{method_name.upper()} EVALUATION RESULTS")
    print("=" * 60)
    
    # Calculate metrics
    acc = accuracy_score(actual_labels, predictions)
    f1_macro = f1_score(actual_labels, predictions, average='macro')
    f1_clickbait = f1_score(actual_labels, predictions, pos_label='clickbait')
    f1_non_clickbait = f1_score(actual_labels, predictions, pos_label='non-clickbait')
    
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Macro: {f1_macro:.4f}")
    print(f"F1-Clickbait: {f1_clickbait:.4f}")
    print(f"F1-Non-Clickbait: {f1_non_clickbait:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(actual_labels, predictions, labels=['clickbait', 'non-clickbait'])
    print(f"\nConfusion Matrix:")
    print("         Pred: clickbait  non-clickbait")
    print(f"True clickbait:      {cm[0][0]:3d}         {cm[0][1]:3d}")
    print(f"True non-clickbait:  {cm[1][0]:3d}         {cm[1][1]:3d}")
    
    # Save results
    results_df = pd.DataFrame({
        'actual': actual_labels,
        'predicted': predictions
    })
    filename = f"{method_name.lower().replace(' ', '_')}_results.csv"
    results_df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")
    
    return acc, f1_macro

def main():
    """
    Main function cho Base và Fine-tuned Vistral Testing
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Base vs Fine-tuned Vistral Models for Clickbait Detection")
    parser.add_argument('--sample_size', type=int, default=None, 
                       help='Number of samples to test (default: all samples)')
    parser.add_argument('--test_base', action='store_true', 
                       help='Test base model')
    parser.add_argument('--test_finetuned', action='store_true',
                       help='Test fine-tuned model')
    args = parser.parse_args()
    
    # If no specific test is chosen, test both
    if not args.test_base and not args.test_finetuned:
        args.test_base = True
        args.test_finetuned = True
    
    print("Test Vistral-7B-Chat Models - BASE vs FINE-TUNED")
    print("Mode: MODEL COMPARISON TESTING")
    if args.sample_size:
        print(f"Sample Size: {args.sample_size} (for quick testing)")
    print("=" * 80)
    
    try:
        # Load test data
        test_data = load_test_data()
        if test_data is None:
            return
        
        all_results = {}
        
        # Test Base Model
        if args.test_base:
            print(f"\n{'='*60}")
            print("TESTING BASE VISTRAL MODEL (CHƯA FINE-TUNE)")
            print(f"{'='*60}")
            
            # Setup Base Vistral model
            print(f"\nLoading Base Vistral Model...")
            base_model, base_tokenizer = setup_base_vistral_model()
            print("Base Vistral Model loaded successfully!")
            
            # Test Base với simple prompting methods
            print(f"\n{'='*40}")
            print("BASE MODEL - FEW-SHOT PROMPTING")
            print(f"{'='*40}")
            predictions_few_shot, actual_labels = test_base_vistral_few_shot(base_model, base_tokenizer, test_data, args.sample_size)
            acc_few_shot, f1_few_shot = evaluate_results(predictions_few_shot, actual_labels, "Base Few-Shot")
            all_results['Base Few-Shot'] = (acc_few_shot, f1_few_shot)
            
            print(f"\n{'='*40}")
            print("BASE MODEL - ZERO-SHOT PROMPTING")
            print(f"{'='*40}")
            predictions_zero_shot, actual_labels = test_base_vistral_zero_shot(base_model, base_tokenizer, test_data, args.sample_size)
            acc_zero_shot, f1_zero_shot = evaluate_results(predictions_zero_shot, actual_labels, "Base Zero-Shot")
            all_results['Base Zero-Shot'] = (acc_zero_shot, f1_zero_shot)
            
            # Clean up base model
            del base_model, base_tokenizer
            torch.cuda.empty_cache()
        
        # Test Fine-tuned Model
        if args.test_finetuned:
            print(f"\n{'='*60}")
            print("TESTING FINE-TUNED VISTRAL MODEL")
            print(f"{'='*60}")
            
            # Setup Fine-tuned Vistral model
            print(f"\nLoading Fine-tuned Vistral Model...")
            ft_model, ft_tokenizer = setup_finetuned_vistral_model()
            
            if ft_model is not None:
                print("Fine-tuned Vistral Model loaded successfully!")
                
                # Test Fine-tuned với simple prompting methods
                print(f"\n{'='*40}")
                print("FINE-TUNED MODEL - FEW-SHOT PROMPTING")
                print(f"{'='*40}")
                predictions_few_shot_ft, actual_labels = test_finetuned_vistral_few_shot(ft_model, ft_tokenizer, test_data, args.sample_size)
                acc_few_shot_ft, f1_few_shot_ft = evaluate_results(predictions_few_shot_ft, actual_labels, "Fine-tuned Few-Shot")
                all_results['Fine-tuned Few-Shot'] = (acc_few_shot_ft, f1_few_shot_ft)
                
                print(f"\n{'='*40}")
                print("FINE-TUNED MODEL - ZERO-SHOT PROMPTING")
                print(f"{'='*40}")
                predictions_zero_shot_ft, actual_labels = test_finetuned_vistral_zero_shot(ft_model, ft_tokenizer, test_data, args.sample_size)
                acc_zero_shot_ft, f1_zero_shot_ft = evaluate_results(predictions_zero_shot_ft, actual_labels, "Fine-tuned Zero-Shot")
                all_results['Fine-tuned Zero-Shot'] = (acc_zero_shot_ft, f1_zero_shot_ft)
                
                # Clean up fine-tuned model
                del ft_model, ft_tokenizer
                torch.cuda.empty_cache()
            else:
                print("Không thể load fine-tuned model. Bỏ qua phần test này.")
        
        # Final Comparison
        if all_results:
            print(f"\n{'='*80}")
            print("FINAL COMPARISON - BASE vs FINE-TUNED VISTRAL MODELS")
            print(f"{'='*80}")
            print(f"{'Method':<25} {'Accuracy':<12} {'F1-Macro':<12}")
            print("-" * 55)
            
            for method, (acc, f1) in all_results.items():
                print(f"{method:<25} {acc:<12.4f} {f1:<12.4f}")
            
            # Best method overall
            best_method = max(all_results.items(), key=lambda x: x[1][1])
            print(f"\nBEST OVERALL METHOD: {best_method[0]}")
            print(f"Best F1-Macro: {best_method[1][1]:.4f}")
            print(f"Best Accuracy: {best_method[1][0]:.4f}")
            
            # Compare base vs fine-tuned
            base_methods = {k: v for k, v in all_results.items() if k.startswith('Base')}
            ft_methods = {k: v for k, v in all_results.items() if k.startswith('Fine-tuned')}
            
            if base_methods and ft_methods:
                print(f"\nCOMPARISON SUMMARY:")
                
                best_base = max(base_methods.items(), key=lambda x: x[1][1])
                best_ft = max(ft_methods.items(), key=lambda x: x[1][1])
                
                print(f"Best Base Model:      {best_base[0]} - F1: {best_base[1][1]:.4f}")
                print(f"Best Fine-tuned:      {best_ft[0]} - F1: {best_ft[1][1]:.4f}")
                
                improvement = best_ft[1][1] - best_base[1][1]
                print(f"Fine-tuning Improvement: {improvement:+.4f} F1-Macro")
                print(f"Relative Improvement: {improvement/best_base[1][1]*100:+.2f}%")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 