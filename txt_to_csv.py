import pandas as pd
from pathlib import Path
import re

class GPUProcessor:
    def __init__(self):
        self.all_data = []
        
    def process_files(self, input_pattern='gpu_*.txt', output_file='gpu_database_2015-2025.csv'):
        """Process all GPU files matching the pattern and save combined output"""
        input_files = self._find_input_files(input_pattern)
        
        if not input_files:
            print("No matching files found. Expected format: gpu_YYYY.txt")
            return False
            
        print(f"Found {len(input_files)} files to process:")
        for file in input_files:
            print(f"- {file.name}")
            
        for file in input_files:
            self._process_single_file(file)
            
        if not self.all_data:
            print("No valid GPU data found in any files")
            return False
            
        return self._save_combined_output(output_file)
        
    def _find_input_files(self, pattern):
        """Find all GPU files from 2015-2025 matching the pattern"""
        files = []
        for file in Path('.').glob(pattern):
            # Extract year from filename (gpu_2015.txt -> 2015)
            year_match = re.search(r'gpu_(\d{4})\.txt', file.name.lower())
            if year_match:
                year = int(year_match.group(1))
                if 2015 <= year <= 2025:
                    files.append(file)
                    
        # Sort files by year
        return sorted(files, key=lambda x: int(re.search(r'gpu_(\d{4})\.txt', x.name.lower()).group(1)))
        
    def _process_single_file(self, file_path):
        """Process a single GPU file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().splitlines()
                
            file_data = []
            current_brand = None
            headers = []
            year = self._extract_year(file_path.name)
            
            for line in content:
                line = line.strip()
                if not line:
                    continue
                    
                # Detect brand sections
                if line in ['AMD', 'Intel', 'NVIDIA']:
                    current_brand = line
                    headers = []
                    continue
                    
                if current_brand:
                    # First non-empty line after brand is headers
                    if not headers:
                        headers = [h.strip() for h in line.split('\t')]
                        continue
                        
                    # Process data rows
                    values = [v.strip() for v in line.split('\t')]
                    if len(values) == len(headers):
                        row = dict(zip(headers, values))
                        row['Brand'] = current_brand
                        row['Year'] = year
                        row['SourceFile'] = file_path.name
                        file_data.append(row)
            
            if file_data:
                self.all_data.extend(file_data)
                print(f"Processed {len(file_data)} entries from {file_path.name}")
            else:
                print(f"No valid data found in {file_path.name}")
                
        except Exception as e:
            print(f"Error processing {file_path.name}: {str(e)}")
            
    def _extract_year(self, filename):
        """Extract year from filename"""
        match = re.search(r'gpu_(\d{4})\.txt', filename.lower())
        return match.group(1) if match else 'Unknown'
        
    def _save_combined_output(self, output_file):
        """Save all processed data to a combined CSV"""
        df = pd.DataFrame(self.all_data)
        
        # Apply brand classification
        df['Brand'] = df.apply(self._classify_brand, axis=1)
        
        # Standard column order
        base_columns = ['Year', 'Brand', 'SourceFile']
        other_columns = [col for col in df.columns if col not in base_columns]
        df = df[base_columns + other_columns]
        
        # Save to CSV
        temp_file = output_file.replace('.csv', '_TEMP.csv')
        df.to_csv(temp_file, index=False, encoding='utf-8-sig')
        
        # Verify before final save
        try:
            test_df = pd.read_csv(temp_file)
            Path(temp_file).replace(output_file)
            self._generate_report(df, output_file)
            return True
        except Exception as e:
            print(f"Failed to save final output: {str(e)}")
            if Path(temp_file).exists():
                Path(temp_file).unlink()
            return False
            
    def _classify_brand(self, row):
        """Final brand verification"""
        product_name = str(row['Product Name']).lower()
        
        if 'radeon' in product_name or 'aero' in product_name or 'rx' in product_name or 'atari' in product_name or 'playstation' in product_name or 'xbox' in product_name or 'firepro' in product_name:
            return 'AMD'
        elif 'intel' in product_name or 'arc' in product_name or 'hd graphics' in product_name or 'iris' in product_name or 'data center' in product_name:
            return 'Intel'
        elif 'geforce' in product_name or 'rtx' in product_name or 'jetson' in product_name or 'titan' in product_name or 'quadro' in product_name or 'tesla' in product_name:
            return 'Nvidia'
        return row['Brand']
        
    def _generate_report(self, df, output_file):
        """Generate comprehensive processing report"""
        print(f"\n{' Processing Complete ':=^50}")
        print(f"Total GPUs Processed: {len(df)}")
        print(f"Output saved to: {output_file}")
        
        # Yearly summary
        print("\nYearly Distribution:")
        year_counts = df['Year'].value_counts().sort_index()
        for year, count in year_counts.items():
            print(f"{year}: {count} GPUs")
            
        # Brand summary
        print("\nBrand Distribution:")
        brand_counts = df['Brand'].value_counts()
        for brand, count in brand_counts.items():
            print(f"{brand}: {count} products")
            
        # Sample output
        print("\nSample Data:")
        print(df.head(3).to_string(index=False))

if __name__ == "__main__":
    processor = GPUProcessor()
    success = processor.process_files()
    
    if not success:
        print("\nProcessing completed with errors. Please check the input files.")
    else:
        print("\nAll files processed successfully!")
