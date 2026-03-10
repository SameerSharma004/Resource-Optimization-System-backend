# backend/model/train.py

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
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
# 2. SELECT FEATURES & CREATE RESOURCE STATES
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

# Dynamic state generation based on resource usage for more professional prediction
def determine_state(row):
    cpu = row['cpu_usage']
    mem = row['memory_usage']
    
    if cpu > 85 or mem > 90:
        return 4  # Critical Load
    elif cpu > 65 or mem > 80:
        return 3  # High Load
    elif cpu > 30 or mem > 60:
        return 2  # Normal Load
    elif cpu > 10 or mem > 40:
        return 1  # Low Load
    else:
        return 0  # Idle

df['system_state'] = df.apply(determine_state, axis=1)
target = df["system_state"]

# -----------------------------
# 3. NORMALIZE FEATURES
# -----------------------------
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# -----------------------------
# 4. CREATE SEQUENCES (LSTM INPUT) & ONE-HOT TARGETS
# -----------------------------
SEQUENCE_LENGTH = 10

X = []
y = []

# Convert target to one-hot encoding for categorical crossentropy
target_onehot = to_categorical(target, num_classes=5)

for i in range(len(features_scaled) - SEQUENCE_LENGTH):
    X.append(features_scaled[i:i + SEQUENCE_LENGTH])
    y.append(target_onehot[i + SEQUENCE_LENGTH])

X = np.array(X)
y = np.array(y)

print("Sequences shape:", X.shape)
print("Labels shape:", y.shape)

# -----------------------------
# 5. TRAIN-TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 6. BUILD LSTM MODEL
# -----------------------------
model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(SEQUENCE_LENGTH, X.shape[2])),
    Dropout(0.3),
    LSTM(32, return_sequences=False),
    Dropout(0.3),
    Dense(16, activation="relu"),
    Dense(5, activation="softmax")  # 5 classes for resource states
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# 7. TRAIN MODEL
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
# 8. EVALUATE MODEL
# -----------------------------
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# -----------------------------
# 9. SAVE MODEL
# -----------------------------
model.save("model.h5")
print("Model saved as model.h5")