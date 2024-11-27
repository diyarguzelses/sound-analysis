import pandas as pd
import os

# Metadata ve dosyaların olduğu klasör yolları
METADATA_FILE = "./data/raw_sounds/UrbanSound8K.csv"
PROCESSED_DATA_DIR = "./data/processed_sounds"

# Metadata dosyasını oku
metadata = pd.read_csv(METADATA_FILE)
slice_files = metadata["slice_file_name"].values

# Processed klasöründeki dosyaları al
processed_files = [file_name.split("_")[-1].replace(".npy", ".wav") for file_name in os.listdir(PROCESSED_DATA_DIR) if file_name.endswith(".npy")]

# Eksik dosyaları bul
missing_files = [file for file in processed_files if file not in slice_files]

print("Processed klasöründe olup metadata dosyasında bulunmayan dosyalar:")
print(missing_files)
