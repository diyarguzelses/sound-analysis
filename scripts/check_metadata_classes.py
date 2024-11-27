import pandas as pd

# Metadata dosyasının yolu
METADATA_FILE = "./data/raw_sounds/UrbanSound8K.csv"

# Metadata dosyasını oku
metadata = pd.read_csv(METADATA_FILE)

# Benzersiz sınıfları kontrol et
unique_classes = metadata["class"].unique()
print(f"Metadata'daki mevcut sınıflar: {unique_classes}")

# Her sınıftan kaç veri olduğunu kontrol et
class_counts = metadata["class"].value_counts()
print("\nHer sınıfa ait veri sayısı:")
print(class_counts)
