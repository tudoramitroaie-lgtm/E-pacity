import os
import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier
import wave

def extract_features(wav_file):
    with wave.open(wav_file, 'r') as wf:
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        audio = audio / 32768.0
        chunk_size = 512
        features = []
        for i in range(0, len(audio) - chunk_size, chunk_size):
            chunk = audio[i:i+chunk_size]
            features.extend([
                np.mean(chunk),
                np.std(chunk),
                np.max(np.abs(chunk)),
                np.mean(np.abs(chunk))
            ])
        features = features[:200]
        while len(features) < 200:
            features.append(0.0)
        return features

print("Loading positive samples...")
X = []
y = []

sample_dir = "wake_word_samples/epacity"
for f in os.listdir(sample_dir):
    if f.endswith(".wav"):
        features = extract_features(os.path.join(sample_dir, f))
        X.append(features)
        y.append(1)

print(f"Loaded {len(X)} positive samples")
print("Generating negative samples...")

for _ in range(len(X) * 2):
    fake = np.random.normal(0, 0.01, 200).tolist()
    X.append(fake)
    y.append(0)

print("Training model...")
clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)
clf.fit(X, y)

with open("epacity_model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("E-pacity wake word model trained and saved!")
