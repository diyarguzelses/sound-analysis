import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# İşlenmiş veri klasörü
PROCESSED_DATA_DIR = "./data/processed_sounds"
METADATA_FILE = "./data/raw_sounds/UrbanSound8K.csv"

# Sınıflar (etiketler)
CLASSES =["gun_shot", "siren", "car_horn",'dog_bark']

# Metadata'dan sınıf etiketlerini yükleme
def load_metadata():
    metadata = pd.read_csv(METADATA_FILE)
    # Sadece CLASSES listesine uyan sınıfları filtrele
    metadata = metadata[metadata["class"].isin(CLASSES)]
    if metadata.empty:
        raise ValueError("Metadata içinde belirtilen sınıflara uyan veri bulunamadı!")
    print(f"Metadata yüklendi. Toplam sınıflandırılabilir dosya: {len(metadata)}")
    return metadata

# Verileri yükleme fonksiyonu
def load_data(metadata):
    X, y = [], []
    processed_count = 0
    for file_name in os.listdir(PROCESSED_DATA_DIR):
        if file_name.endswith(".npy"):
            # Augmentation'dan gelen dosya öneklerini temizle
            cleaned_file_name = file_name.split("_")[-1].replace(".npy", ".wav")  # noisy_148837 -> 148837.wav
            try:
                label_row = metadata[metadata["slice_file_name"] == cleaned_file_name]
                if not label_row.empty:
                    label = label_row["class"].values[0]
                    if label in CLASSES:
                        file_path = os.path.join(PROCESSED_DATA_DIR, file_name)
                        features = np.load(file_path)
                        X.append(features)
                        y.append(CLASSES.index(label))  # Sınıfın indeksini ekle
                        processed_count += 1
                else:
                    print(f"Metadata ile eşleşmeyen dosya: {file_name}")
            except (ValueError, IndexError):
                print(f"Geçersiz dosya adı veya sınıf bulunamadı: {file_name}")
    if processed_count == 0:
        raise ValueError("Hiçbir dosya işlenmedi! Metadata veya dosya isimleriyle ilgili bir sorun olabilir.")
    print(f"Toplam işlenmiş dosya: {processed_count}")
    return np.array(X), np.array(y)

# Metadata'yı yükle
print("Metadata yükleniyor...")
metadata = load_metadata()

# Veriyi yükle
print("Veri yükleniyor...")
X, y = load_data(metadata)

# Veriyi eğitim ve test olarak böl
if len(X) == 0 or len(y) == 0:
    raise ValueError("Veri yüklenirken hata oluştu! X veya y boş.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Etiketleri one-hot encoding formatına çevir
y_train = to_categorical(y_train, num_classes=len(CLASSES))
y_test = to_categorical(y_test, num_classes=len(CLASSES))

# Model yapısı
model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(len(CLASSES), activation="softmax")
])

# Modeli derle
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Modeli eğit
print("Model eğitiliyor...")
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), batch_size=16)

# Modeli kaydet
if not os.path.exists("./models"):
    os.makedirs("./models")
model.save("./models/sound_classifier.h5")

print("\nModel eğitimi tamamlandı ve kaydedildi: ./models/sound_classifier.h5")
print(f"Toplam veri sayısı: {len(X)}")
print(f"Eğitim veri sayısı: {len(X_train)}")
print(f"Test veri sayısı: {len(X_test)}")

# Metadata'daki sınıfları kontrol et
unique_classes = metadata["class"].unique()
print(f"Metadata'daki sınıflar: {unique_classes}")
print(f"CLASSES listesi: {CLASSES}")
