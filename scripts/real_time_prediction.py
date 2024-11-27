import sounddevice as sd
import librosa
import numpy as np
from tensorflow.keras.models import load_model

# Model ve sınıf bilgileri
MODEL_PATH = "./models/sound_classifier.h5"
CLASSES = ['gun_shot', 'siren', 'car_horn','dog_bark']
model = load_model(MODEL_PATH)

# Mikrofon ayarları
DURATION = 2  # Kaydedilecek sesin süresi (saniye)
SAMPLE_RATE = 22050

def predict_from_mic():
    print("Mikrofondan ses kaydediliyor...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    audio = audio.flatten()

    # MFCC çıkarma
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)

    # Tahmin yapma
    mfcc = mfcc.reshape(1, -1)  # Model girişine uygun şekil
    prediction = model.predict(mfcc)
    predicted_class = CLASSES[np.argmax(prediction)]

    print(f"Tahmin edilen sınıf: {predicted_class}")

if __name__ == "__main__":
    while True:
        input("Mikrofondan ses tahmini yapmak için ENTER'a bas...")
        predict_from_mic()
