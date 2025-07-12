"""
Convert Simple Dataset to Alpaca Format V2
Chuyển đổi dữ liệu từ simple_dataset sang format Alpaca JSON để sử dụng với Vistral
"""

import pandas as pd
import json
import os
from tqdm import tqdm

def create_alpaca_format_v2(title, label):
    """
    Tạo format Alpaca V2 cho một sample clickbait với instruction tối ưu cho Vistral
    """
    instruction = """Bạn là chuyên gia phân tích báo chí Việt Nam. Hãy phân loại tiêu đề bài báo thành clickbait hoặc non-clickbait.

Tiêu chí phân loại:
- CLICKBAIT: Tiêu đề câu view, sử dụng từ ngữ cảm xúc mạnh, tạo tò mò quá mức, thiếu thông tin cụ thể, có từ khóa như "bí mật", "gây sốc", "không ai ngờ", "top X"
- NON-CLICKBAIT: Tiêu đề thông tin rõ ràng, khách quan, trung thực, có nội dung cụ thể, mang tính tin tức thực tế

Hãy phân tích và đưa ra kết luận chính xác."""

    input_text = f"Phân loại tiêu đề sau: {title}"
    
    # Tạo output với phân tích ngắn gọn nhưng rõ ràng
    if label == "clickbait":
        output_text = f"Đây là tiêu đề clickbait vì sử dụng ngôn ngữ câu view và tạo tò mò quá mức.\n\nKết quả: clickbait"
    else:
        output_text = f"Đây là tiêu đề non-clickbait vì cung cấp thông tin rõ ràng và khách quan.\n\nKết quả: non-clickbait"
    
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text
    }

def convert_simple_dataset_to_alpaca(data_dir, output_dir):
    """
    Convert simple_dataset to Alpaca JSON format
    """
    print(f"START: Converting {data_dir} to Alpaca V2 format...")
    print(f"📁 Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # File mappings từ simple_dataset structure
    splits = ['train', 'val', 'test']
    total_samples = 0
    
    for split in splits:
        csv_file = os.path.join(data_dir, split, f"{split}.csv")
        output_file = os.path.join(output_dir, f"{split}_alpaca.json")
        
        if os.path.exists(csv_file):
            print(f"\nDATA: Converting {split} split from {csv_file}")
            
            # Load CSV
            df = pd.read_csv(csv_file)
            print(f"SUCCESS: Loaded {len(df)} samples")
            
            # Show label distribution
            label_counts = df['label'].value_counts()
            print(f"STATS: Label distribution:")
            for label, count in label_counts.items():
                percentage = count / len(df) * 100
                print(f"   {label}: {count} ({percentage:.1f}%)")
            
            # Convert to Alpaca format
            alpaca_data = []
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Converting {split}"):
                alpaca_sample = create_alpaca_format_v2(row['title'], row['label'])
                alpaca_data.append(alpaca_sample)
            
            # Save to JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(alpaca_data, f, ensure_ascii=False, indent=2)
            
            print(f"SAVE: Saved {len(alpaca_data)} samples to {output_file}")
            total_samples += len(alpaca_data)
            
        else:
            print(f"WARNING: Warning: {csv_file} not found, skipping...")
    
    return total_samples

def show_conversion_summary(output_dir):
    """
    Hiển thị tổng kết quá trình convert
    """
    print(f"\nCOMPLETE: CONVERSION COMPLETED!")
    print(f"LOAD: Output directory: {output_dir}/")
    print(f"REPORT: Files created:")
    
    total_samples = 0
    for split in ['train', 'val', 'test']:
        json_file = os.path.join(output_dir, f"{split}_alpaca.json")
        if os.path.exists(json_file):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"   SUCCESS: {json_file} - {len(data):,} samples")
                total_samples += len(data)
    
    print(f"STATS: Total samples converted: {total_samples:,}")
    
    # Show example từ test file
    test_file = os.path.join(output_dir, "test_alpaca.json")
    if os.path.exists(test_file):
        print(f"\n📖 Example from test data:")
        with open(test_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if data:
                example = data[0]
                print(f"Input: {example['input']}")
                print(f"Output: {example['output']}")

def main():
    """
    Main conversion process
    """
    print("=" * 70)
    print("PROCESS: CONVERTING SIMPLE_DATASET TO ALPACA FORMAT V2")
    print("=" * 70)
    
    # Paths
    input_dir = "simple_dataset"
    output_dir = "data_alpaca_v2"
    
    # Validate input directory
    if not os.path.exists(input_dir):
        print(f"ERROR: Error: Input directory {input_dir} not found!")
        return
    
    print(f"📁 Input: {input_dir}")
    print(f"📁 Output: {output_dir}")
    
    # Convert
    total_samples = convert_simple_dataset_to_alpaca(input_dir, output_dir)
    
    # Show summary
    show_conversion_summary(output_dir)
    
    print(f"\nSUCCESS: Conversion completed successfully!")
    print(f"TARGET: Ready to use with test_base_vistral.py")
    print(f"   Update script to use: {output_dir}/test_alpaca.json")

if __name__ == "__main__":
    main() 