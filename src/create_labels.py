import pandas as pd
from pathlib import Path

# Paths
dataset_dir = Path("data/test")  
input_csv = dataset_dir / "test_labels.csv"  
output_csv = dataset_dir / "labels.csv"  

# 1. Read the original CSV
df = pd.read_csv(input_csv)

# 2. Keep only image name and medicine name
df = df[['IMAGE', 'MEDICINE_NAME']]

# 3. Rename columns to TroCR standard
df.columns = ['file_name', 'text']

# 4. Save to labels.csv
df.to_csv(output_csv, index=False)

print(f"✅ labels.csv saved to: {output_csv}")
print(df.head())
