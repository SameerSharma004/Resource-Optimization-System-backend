# backend/model/train.py

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# 1. LOAD DATA
# -----------------------------
data_path = "../data/laptop_data.csv"
df = pd.read_csv(data_path)

print("Data loaded:", df.shape)

# -----------------------------
# 2. SELECT FEATURES
# -----------------------------
FEATURE_COLUMNS = [
    "cpu_usage",
    "memory_usage",
    "net_upload_mbps",
    "net_download_mbps",
    "disk_read_mbps",
    "disk_write_mbps"
]

features = df[FEATURE_COLUMNS]

# -----------------------------
# 3. CREATE TARGET (LABEL)
# -----------------------------
target = df["idle"]

# -----------------------------
# 4. NORMALIZE FEATURES
# -----------------------------
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# -----------------------------
# 5. CREATE SEQUENCES (LSTM INPUT)
# -----------------------------
SEQUENCE_LENGTH = 10

X = []
y = []

for i in range(len(features_scaled) - SEQUENCE_LENGTH):
    X.append(features_scaled[i:i + SEQUENCE_LENGTH])
    y.append(target.iloc[i + SEQUENCE_LENGTH])

X = np.array(X)
y = np.array(y)

print("Sequences shape:", X.shape)
print("Labels shape:", y.shape)

# -----------------------------
# 6. TRAIN-TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 7. BUILD LSTM MODEL
# -----------------------------
model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(SEQUENCE_LENGTH, X.shape[2])),
    Dropout(0.3),
    LSTM(32, return_sequences=False),
    Dropout(0.3),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# 8. TRAIN MODEL
# -----------------------------
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping]
)

# -----------------------------
# 9. EVALUATE MODEL
# -----------------------------
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# -----------------------------
# 10. SAVE MODEL
# -----------------------------
model.save("model.h5")
print("Model saved as model.h5")