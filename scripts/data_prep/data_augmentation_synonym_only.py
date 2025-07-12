"""
Data Augmentation for Imbalanced Dataset using Synonym Replacement
Sử dụng Synonym Replacement để tăng cường dữ liệu cho class minority (clickbait)
Không cần external libraries phức tạp
"""

import os
import pandas as pd
import numpy as np
import json
import re
import random
from tqdm import tqdm
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings("ignore")

class VietnameseSynonymReplacer:
    """
    Vietnamese Synonym Replacement using comprehensive predefined synonyms
    """
    
    def __init__(self):
        print("LOG: Initializing Vietnamese Synonym Replacer...")
        
        # Extended Vietnamese synonyms dictionary
        self.synonyms = {
            # Clickbait-specific words (quan trọng nhất)
            "bí mật": ["điều bí ẩn", "bí ẩn", "điều kín", "điều ẩn giấu", "điều không ai biết", "điều thầm kín"],
            "gây sốc": ["gây shock", "làm choáng", "làm bất ngờ", "khiến sửng sốt", "gây ngỡ ngàng", "làm kinh ngạc"],
            "không ai ngờ": ["bất ngờ", "không ai biết", "ngoài dự đoán", "không lường trước", "bất thần", "đột ngột"],
            "khó tin": ["không thể tin", "phi thường", "bất thường", "lạ lùng", "kỳ lạ", "không tin nổi"],
            "kinh hoàng": ["khủng khiếp", "đáng sợ", "ghê rợn", "rùng rợn", "kinh dị", "khiếp sợ"],
            "bất ngờ": ["đột ngột", "không ngờ", "ngoài dự tính", "ngoài ý muốn", "không lường trước"],
            "sự thật": ["thực tế", "thật", "điều thật", "thực chất", "hiện thực", "sự thực"],
            "choáng váng": ["sững sờ", "ngỡ ngàng", "bất ngờ", "sốc", "kinh ngạc", "tá hỏa"],
            
            # Numbers and rankings (common in clickbait)
            "top": ["hàng đầu", "tốt nhất", "đỉnh", "cao nhất", "số 1", "đứng đầu"],
            "tốt nhất": ["hay nhất", "xuất sắc nhất", "đỉnh cao", "số một", "hàng đầu"],
            "xấu nhất": ["tệ nhất", "dở nhất", "kém nhất", "tồi tệ nhất", "kinh khủng nhất"],
            
            # Common adjectives
            "tốt": ["hay", "giỏi", "xuất sắc", "ưu việt", "tuyệt vời", "hoàn hảo"],
            "xấu": ["tệ", "dở", "kém", "tồi tệ", "kinh khủng", "thảm hại"],
            "lớn": ["to", "khổng lồ", "rộng lớn", "đồ sộ", "vĩ đại", "khủng"],
            "nhỏ": ["bé", "tí hon", "thu nhỏ", "tí xíu", "li ti", "nhít"],
            "đẹp": ["xinh", "lung linh", "quyến rũ", "hấp dẫn", "tuyệt đẹp", "rực rỡ"],
            "nhiều": ["đông", "mọi", "rất nhiều", "vô số", "hàng loạt", "vô vàn"],
            "ít": ["thiếu", "hiếm", "khan hiếm", "không nhiều", "số ít", "eo hẹp"],
            
            # News-related words
            "tin tức": ["thông tin", "báo tin", "tin báo", "thông báo", "bản tin", "tin bài"],
            "sự kiện": ["việc", "chuyện", "điều", "câu chuyện", "tình huống", "biến cố"],
            "người": ["cá nhân", "con người", "nhân vật", "ai đó", "đối tượng", "cư dân"],
            "nổi tiếng": ["nổi bật", "được biết đến", "có tiếng", "danh tiếng", "nổi danh", "nức tiếng"],
            "thành công": ["chiến thắng", "đạt được", "hoàn thành", "giành được", "đạt thành tích", "thắng lợi"],
            
            # Time-related words
            "mới": ["mới mẻ", "tươi mới", "mới lạ", "hiện đại", "cập nhật", "vừa mới"],
            "cũ": ["lỗi thời", "xưa", "cổ", "quá khứ", "đã qua", "lạc hậu"],
            "nhanh": ["chớp nhoáng", "thần tốc", "tức thì", "nhanh chóng", "mau lẹ", "tốc độ"],
            "chậm": ["từ từ", "chậm chạp", "ì ạch", "chậm rãi", "lề mề", "ậm ạch"],
            
            # Emotional words
            "vui": ["vui vẻ", "hạnh phúc", "phấn khích", "hào hứng", "thích thú", "vui mừng"],
            "buồn": ["u sầu", "thất vọng", "đau khổ", "chán nản", "ủ rũ", "sầu muộn"],
            "giận": ["tức giận", "bực tức", "phẫn nộ", "khó chịu", "cáu kỉnh", "cáu gắt"],
            "sợ": ["lo sợ", "hoảng sợ", "kinh sợ", "e ngại", "lo lắng", "sợ hãi"],
            
            # Action words
            "làm": ["thực hiện", "tiến hành", "tạo ra", "thực thi", "cải thiện", "phát triển"],
            "có": ["sở hữu", "tồn tại", "hiện có", "chứa đựng", "bao gồm", "mang"],
            "đi": ["di chuyển", "tiến hành", "rời khỏi", "bước đi", "chuyển động", "hành trình"],
            "đến": ["tới", "đạt tới", "tiếp cận", "tiến đến", "tới nơi", "về"],
            
            # Vietnamese-specific words
            "việt nam": ["việt", "vn", "trong nước", "đất nước", "quê hương", "tổ quốc"],
            "thế giới": ["toàn cầu", "quốc tế", "địa cầu", "trái đất", "nhân loại", "toàn thế giới"],
            "hà nội": ["thủ đô", "hà thành", "thăng long", "khu vực hà nội"],
            "tp hcm": ["sài gòn", "thành phố hồ chí minh", "hcm", "thương cảng"],
            
            # Technology words
            "công nghệ": ["kỹ thuật", "khoa học", "tiến bộ", "đổi mới", "innovation", "tech"],
            "internet": ["mạng", "online", "web", "trực tuyến", "mạng lưới", "digital"],
            "điện thoại": ["di động", "smartphone", "mobile", "thiết bị", "máy"],
            
            # Business words
            "công ty": ["doanh nghiệp", "tập đoàn", "tổ chức", "firm", "công ty", "business"],
            "tiền": ["đồng tiền", "tài chính", "thu nhập", "lương", "kinh tế", "ngân sách"],
            "giá": ["mức giá", "chi phí", "cost", "price", "giá cả", "phí"],
            
            # Sports words
            "bóng đá": ["football", "soccer", "thể thao", "môn thể thao", "túc cầu"],
            "vô địch": ["chiến thắng", "đăng quang", "thắng lớn", "giành cup", "champion"],
        }
        
        # Create reverse mapping for better coverage
        self.reverse_synonyms = {}
        for word, synonyms in self.synonyms.items():
            for synonym in synonyms:
                if synonym not in self.reverse_synonyms:
                    self.reverse_synonyms[synonym] = []
                self.reverse_synonyms[synonym].append(word)
        
        print(f"SUCCESS: Vietnamese Synonym Replacer initialized with {len(self.synonyms)} main words!")
    
    def get_synonyms(self, word: str) -> List[str]:
        """Get synonyms for a word"""
        word_lower = word.lower().strip()
        
        # Check main dictionary
        if word_lower in self.synonyms:
            return self.synonyms[word_lower]
        
        # Check reverse dictionary
        if word_lower in self.reverse_synonyms:
            return self.reverse_synonyms[word_lower]
        
        return []
    
    def synonym_replacement(self, text: str, replacement_prob: float = 0.4) -> str:
        """
        Replace words with their synonyms with higher probability for clickbait keywords
        """
        words = text.split()
        augmented_words = []
        
        # Clickbait keywords get higher replacement probability
        clickbait_keywords = ["bí mật", "gây sốc", "không ai ngờ", "khó tin", "kinh hoàng", 
                             "bất ngờ", "sự thật", "choáng váng", "top", "tốt nhất", "xấu nhất"]
        
        for word in words:
            # Clean word (remove punctuation for lookup)
            clean_word = re.sub(r'[^\w\s]', '', word.lower())
            synonyms = self.get_synonyms(clean_word)
            
            # Higher probability for clickbait keywords
            prob = 0.7 if clean_word in clickbait_keywords else replacement_prob
            
            if synonyms and random.random() < prob:
                # Replace with random synonym
                synonym = random.choice(synonyms)
                
                # Preserve capitalization
                if len(word) > 0 and word[0].isupper():
                    synonym = synonym.capitalize()
                
                # Preserve punctuation
                punctuation = re.findall(r'[^\w\s]', word)
                if punctuation:
                    synonym += ''.join(punctuation)
                
                augmented_words.append(synonym)
            else:
                augmented_words.append(word)
        
        return ' '.join(augmented_words)

class DataAugmenter:
    """
    Data Augmenter using only Synonym Replacement
    """
    
    def __init__(self):
        print("START: Initializing Data Augmenter (Synonym Replacement Only)...")
        self.synonym_replacer = VietnameseSynonymReplacer()
        print("SUCCESS: Data Augmenter ready!")
    
    def augment_text(self, text: str, num_variants: int = 1) -> List[str]:
        """
        Generate multiple variants of the same text
        """
        variants = []
        
        for _ in range(num_variants):
            # Apply synonym replacement with some randomness
            variant = self.synonym_replacer.synonym_replacement(text, replacement_prob=random.uniform(0.3, 0.6))
            
            # Make sure it's different from original
            if variant != text and len(variant.strip()) > 0:
                variants.append(variant)
        
        return variants
    
    def create_alpaca_format(self, title: str, label: str) -> Dict:
        """
        Create Alpaca format for a sample
        """
        instruction = """Bạn là chuyên gia phân tích báo chí Việt Nam. Hãy phân loại tiêu đề bài báo thành clickbait hoặc non-clickbait.

Tiêu chí phân loại:
- CLICKBAIT: Tiêu đề câu view, sử dụng từ ngữ cảm xúc mạnh, tạo tò mò quá mức, thiếu thông tin cụ thể, có từ khóa như "bí mật", "gây sốc", "không ai ngờ", "top X"
- NON-CLICKBAIT: Tiêu đề thông tin rõ ràng, khách quan, trung thực, có nội dung cụ thể, mang tính tin tức thực tế

Hãy phân tích và đưa ra kết luận chính xác."""

        input_text = f"Phân loại tiêu đề sau: {title}"
        
        if label == "clickbait":
            output_text = f"Đây là tiêu đề clickbait vì sử dụng ngôn ngữ câu view và tạo tò mò quá mức.\n\nKết quả: clickbait"
        else:
            output_text = f"Đây là tiêu đề non-clickbait vì cung cấp thông tin rõ ràng và khách quan.\n\nKết quả: non-clickbait"
        
        return {
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        }
    
    def convert_csv_to_alpaca_json(self, csv_file: str, json_file: str):
        """
        Convert CSV file to Alpaca JSON format
        """
        df = pd.read_csv(csv_file)
        alpaca_data = []
        
        for _, row in df.iterrows():
            alpaca_sample = self.create_alpaca_format(row['title'], row['label'])
            alpaca_data.append(alpaca_sample)
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(alpaca_data, f, ensure_ascii=False, indent=2)
        
        print(f"REPORT: Alpaca JSON saved to: {json_file}")
        return len(alpaca_data)
    
    def augment_dataset(self, 
                       input_dir: str, 
                       output_dir: str, 
                       target_class: str = "clickbait",
                       augmentation_ratio: float = 2.0):
        """
        Augment dataset for minority class
        """
        print(f"\nPROCESS: Starting dataset augmentation...")
        print(f"📁 Input: {input_dir}")
        print(f"📁 Output: {output_dir}")
        print(f"TARGET: Target class: {target_class}")
        print(f"METRICS: Augmentation ratio: {augmentation_ratio}x")
        print(f"🛠️ Method: Synonym Replacement Only")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process only train split (val and test should remain original)
        for split in ["train"]:
            print(f"\nSTATS: Processing {split} split...")
            
            input_file = os.path.join(input_dir, split, f"{split}.csv")
            if not os.path.exists(input_file):
                print(f"WARNING: File not found: {input_file}")
                continue
            
            # Load data
            df = pd.read_csv(input_file)
            print(f"SUCCESS: Loaded {len(df)} samples")
            
            # Show class distribution
            class_counts = df['label'].value_counts()
            print(f"STATS: Class distribution:")
            for label, count in class_counts.items():
                print(f"   {label}: {count} ({count/len(df)*100:.1f}%)")
            
            # Generate augmented data for minority class
            if target_class in class_counts.index:
                minority_data = df[df['label'] == target_class].copy()
                n_minority = len(minority_data)
                n_to_generate = int(n_minority * (augmentation_ratio - 1))
                
                print(f"TARGET: Generating {n_to_generate} augmented samples for {target_class}")
                
                augmented_samples = []
                generated_count = 0
                
                # Generate multiple variants per sample
                samples_per_original = max(1, n_to_generate // n_minority)
                
                for idx, (_, sample) in tqdm(enumerate(minority_data.iterrows()), 
                                          total=len(minority_data), 
                                          desc=f"Augmenting {split}"):
                    original_title = sample['title']
                    
                    # Generate variants
                    variants = self.augment_text(original_title, num_variants=samples_per_original + 1)
                    
                    for variant in variants:
                        if generated_count >= n_to_generate:
                            break
                            
                        if variant != original_title and len(variant.strip()) > 0:
                            augmented_samples.append({
                                'title': variant,
                                'label': target_class,
                                'original_title': original_title,
                                'augmentation_method': 'synonym_replacement'
                            })
                            generated_count += 1
                    
                    if generated_count >= n_to_generate:
                        break
                
                print(f"SUCCESS: Generated {len(augmented_samples)} valid augmented samples")
                
                # Show some examples
                if len(augmented_samples) > 0:
                    print(f"\n📖 Examples of augmented data:")
                    for i in range(min(3, len(augmented_samples))):
                        sample = augmented_samples[i]
                        print(f"   Original: {sample['original_title']}")
                        print(f"   Augmented: {sample['title']}")
                        print()
                
                # Combine original and augmented data
                augmented_df = pd.DataFrame(augmented_samples)
                combined_df = pd.concat([df, augmented_df[['title', 'label']]], ignore_index=True)
                
                # Shuffle
                combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
                
                print(f"STATS: Final dataset size: {len(combined_df)} samples")
                
                # Show new class distribution
                new_class_counts = combined_df['label'].value_counts()
                print(f"STATS: New class distribution:")
                for label, count in new_class_counts.items():
                    percentage = count / len(combined_df) * 100
                    print(f"   {label}: {count} ({percentage:.1f}%)")
                
                # Save combined dataset
                output_split_dir = os.path.join(output_dir, split)
                os.makedirs(output_split_dir, exist_ok=True)
                
                # Save CSV format
                output_csv_file = os.path.join(output_split_dir, f"{split}.csv")
                combined_df.to_csv(output_csv_file, index=False)
                print(f"SAVE: CSV saved to: {output_csv_file}")
                
                # Save Alpaca JSON format
                output_json_file = os.path.join(output_split_dir, f"{split}_alpaca.json")
                json_samples = self.convert_csv_to_alpaca_json(output_csv_file, output_json_file)
                
                # Save augmentation details
                if len(augmented_samples) > 0:
                    details_file = os.path.join(output_split_dir, f"{split}_augmentation_details.csv")
                    augmented_df.to_csv(details_file, index=False)
                    print(f"REPORT: Augmentation details saved to: {details_file}")
            
            else:
                print(f"WARNING: Target class '{target_class}' not found in {split}")
                # Just copy original files
                output_split_dir = os.path.join(output_dir, split)
                os.makedirs(output_split_dir, exist_ok=True)
                
                # Save CSV format
                output_csv_file = os.path.join(output_split_dir, f"{split}.csv")
                df.to_csv(output_csv_file, index=False)
                print(f"REPORT: CSV copied to: {output_csv_file}")
                
                # Save Alpaca JSON format
                output_json_file = os.path.join(output_split_dir, f"{split}_alpaca.json")
                json_samples = self.convert_csv_to_alpaca_json(output_csv_file, output_json_file)

def main():
    """
    Main augmentation process
    """
    print("=" * 80)
    print("PROCESS: DATA AUGMENTATION: VIETNAMESE SYNONYM REPLACEMENT")
    print("REPORT: Creates both CSV and Alpaca JSON formats")
    print("WARNING:  ONLY augments TRAIN set - Val/Test remain original")
    print("=" * 80)
    
    # Configuration
    input_dir = "simple_dataset"
    output_dir = "data_genUpsampling"
    target_class = "clickbait"  # Minority class to augment
    augmentation_ratio = 2.2    # 2.2x tăng class clickbait
    
    print(f"📁 Input directory: {input_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"TARGET: Target class: {target_class}")
    print(f"METRICS: Augmentation ratio: {augmentation_ratio}x")
    print(f"🛠️ Method: Vietnamese Synonym Replacement")
    print(f"REPORT: Output formats: CSV + Alpaca JSON")
    print(f"WARNING:  Processing: TRAIN only (Val/Test keep original)")
    
    # Validate input directory
    if not os.path.exists(input_dir):
        print(f"ERROR: Input directory not found: {input_dir}")
        return
    
    try:
        # Initialize augmenter
        augmenter = DataAugmenter()
        
        # Run augmentation
        augmenter.augment_dataset(
            input_dir=input_dir,
            output_dir=output_dir,
            target_class=target_class,
            augmentation_ratio=augmentation_ratio
        )
        
        print(f"\nCOMPLETE: DATA AUGMENTATION COMPLETED!")
        print(f"📁 Augmented data saved to: {output_dir}")
        print(f"REPORT: Generated both CSV and Alpaca JSON formats:")
        print(f"   - *.csv files for traditional ML models")
        print(f"   - *_alpaca.json files for LLM fine-tuning")
        print(f"SEARCH: Check augmentation details in *_augmentation_details.csv files")
        print(f"TIP: To use with models:")
        print(f"   - Train: use data_genUpsampling/train/ (augmented)")
        print(f"   - Val: use original simple_dataset/val/val.csv (NO augmentation)")
        print(f"   - Test: use original simple_dataset/test/test.csv (NO augmentation)")
        print(f"   - Only TRAIN set is augmented, Val/Test remain original for proper evaluation")
        
    except Exception as e:
        print(f"ERROR: Error during augmentation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 