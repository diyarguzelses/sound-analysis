import os
import shutil
import pandas as pd

# UrbanSound8K veri seti yolları
DATA_DIR = "./data/raw_sounds"  # fold1, fold2 vb. klasörlerin bulunduğu ana klasör
METADATA_FILE = "./data/raw_sounds/UrbanSound8K.csv"  # Metadata dosyasının tam yolu
OUTPUT_DIR = "./data/raw_sounds/tehlikeli_sesler"  # Tehlikeli seslerin taşınacağı klasör

# Hedef kategoriler (tehlikeli sesler)
TARGET_CLASSES = ["gun_shot", "siren", "car_horn","dog_bark"]

# Metadata dosyasını oku
try:
    metadata = pd.read_csv(METADATA_FILE)
    print(f"Metadata dosyası başarıyla yüklendi: {METADATA_FILE}")
except FileNotFoundError:
    print(f"Hata: Metadata dosyası bulunamadı: {METADATA_FILE}")
    exit(1)

# Tehlikeli sesleri seç
filtered_metadata = metadata[metadata["class"].isin(TARGET_CLASSES)]
if filtered_metadata.empty:
    print("Hedef sınıflar için hiçbir veri bulunamadı.")
    exit(1)

# Hedef klasör oluştur
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Ses dosyalarını taşı
success_count = 0
for _, row in filtered_metadata.iterrows():
    src_file = os.path.join(DATA_DIR, f"fold{row['fold']}", row['slice_file_name'])
    dest_file = os.path.join(OUTPUT_DIR, row['slice_file_name'])
    try:
        shutil.copyfile(src_file, dest_file)
        success_count += 1
    except FileNotFoundError:
        print(f"Hata: Kaynak dosya bulunamadı: {src_file}")
    except Exception as e:
        print(f"Hata: {src_file} kopyalanırken bir hata oluştu: {e}")

print(f"Tehlikeli ses dosyaları başarıyla taşındı! Toplam: {success_count}")
print(f"Taşınan dosyalar: {OUTPUT_DIR}")
