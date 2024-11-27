import os
import librosa
import numpy as np
import soundfile as sf

# Ham ses dosyalarının olduğu klasör
RAW_DATA_DIR = "./data/raw_sounds/tehlikeli_sesler"

# Augmented dosyaların kaydedileceği klasör
AUGMENTED_DATA_DIR = "./data/augmented_sounds"

# Veri artırma fonksiyonu
def augment_sound(file_path, output_dir):
    y, sr = librosa.load(file_path)

    # Gürültü ekleme
    noise = np.random.normal(0, 0.005, len(y))
    noisy_data = y + noise
    sf.write(os.path.join(output_dir, "noisy_" + os.path.basename(file_path)), noisy_data, sr)

    # Hız değiştirme
    faster_data = librosa.effects.time_stretch(y, rate=1.2)
    sf.write(os.path.join(output_dir, "faster_" + os.path.basename(file_path)), faster_data, sr)

    # Ton değiştirme (Pitch shifting)
    pitched_data = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=2)
    sf.write(os.path.join(output_dir, "pitched_" + os.path.basename(file_path)), pitched_data, sr)

if __name__ == "__main__":
    if not os.path.exists(AUGMENTED_DATA_DIR):
        os.makedirs(AUGMENTED_DATA_DIR)

    for file_name in os.listdir(RAW_DATA_DIR):
        file_path = os.path.join(RAW_DATA_DIR, file_name)
        augment_sound(file_path, AUGMENTED_DATA_DIR)

    print("\nVeri artırma işlemi tamamlandı! Artırılmış dosyalar ./data/augmented_sounds klasörüne kaydedildi.")
    print("Şimdi data preprocessing (MFCC çıkarma) adımına geçebilirsiniz.")
