import librosa
import numpy as np
import os

# Augmented dosyaların olduğu klasör
RAW_DATA_DIR = "./data/augmented_sounds"

# MFCC özelliklerinin kaydedileceği klasör
PROCESSED_DATA_DIR = "./data/processed_sounds"

# MFCC çıkarma fonksiyonu
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)  # Ses dosyasını yükle
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)  # MFCC çıkar
        return np.mean(mfcc.T, axis=0)  # Ortalama MFCC değerini dön
    except Exception as e:
        print(f"Hata: {file_path} işlenemedi. Sebep: {e}")
        return None

if __name__ == "__main__":
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)

    for file_name in os.listdir(RAW_DATA_DIR):
        file_path = os.path.join(RAW_DATA_DIR, file_name)
        if file_name.endswith(".wav"):
            print(f"İşleniyor: {file_name}")
            features = extract_features(file_path)
            if features is not None:
                output_file = os.path.join(PROCESSED_DATA_DIR, f"{file_name.split('.')[0]}.npy")
                np.save(output_file, features)

    print("\nData preprocessing (MFCC çıkarma) işlemi tamamlandı!")
    print("İşlenmiş dosyalar ./data/processed_sounds klasörüne kaydedildi.")
    print("Şimdi model eğitimine geçebilirsiniz.")
