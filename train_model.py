import random
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
import matplotlib.pyplot as plt

# --- 1. LOAD & PROCESS DATA ---
print("Loading data...")
load_files = sorted(os.listdir('data/loads'))
struct_files = sorted(os.listdir('data/structures'))

X = []
y = []

# Load data
for f_load, f_struct in zip(load_files, struct_files):
    if f_load.endswith('.npy'):
        load_grid = np.load(f'data/loads/{f_load}')
        struct_grid = np.load(f'data/structures/{f_struct}')

        # --- TRICK 1: THICKEN THE INPUT ---
        # The single pixel is too small. Let's dilate it to a 3x3 block.
        # Find where the 1.0 is
        rows, cols = np.where(load_grid == 1.0)
        if len(rows) > 0:
            r, c = rows[0], cols[0]
            # Set neighbors to 1.0 too (clamping to boundaries)
            r_min, r_max = max(0, r-1), min(19, r+1)
            c_min, c_max = max(0, c-1), min(59, c+1)
            load_grid[r_min:r_max+1, c_min:c_max+1] = 1.0

        X.append(load_grid)
        y.append(struct_grid)

X = np.array(X).reshape(-1, 20, 60, 1)
y = np.array(y).reshape(-1, 20, 60, 1)

# Split Data
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Data Loaded: {len(X)} samples. Input Dot thickened.")

# --- 2. BUILD THE TRUE U-NET (Functional API) ---


def build_unet():
    # Input
    inputs = Input(shape=(20, 60, 1))

    # --- ENCODER (Downsampling) ---
    # Layer 1
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    # Layer 2
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # --- BOTTLENECK ---
    b = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    b = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(b)

    # --- DECODER (Upsampling + SKIP CONNECTIONS) ---
    # Up 1
    u1 = layers.UpSampling2D((2, 2))(b)
    cat1 = layers.Concatenate()([u1, c2])  # <--- THE MAGIC BRIDGE
    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(cat1)

    # Up 2
    u2 = layers.UpSampling2D((2, 2))(c3)
    cat2 = layers.Concatenate()([u2, c1])  # <--- THE MAGIC BRIDGE
    c4 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(cat2)

    # Output
    outputs = layers.Conv2D(
        1, (3, 3), activation='sigmoid', padding='same')(c4)

    return models.Model(inputs, outputs)


model = build_unet()
model.compile(optimizer='adam', loss='mae', metrics=['acc'])
model.summary()

# --- 3. TRAIN ---
print("\nStarting Training (V3 - U-Net)...")
# Training for 60 epochs should be enough with U-Net
history = model.fit(X_train, y_train, epochs=60,
                    batch_size=16, validation_data=(X_test, y_test))

# --- 4. TEST ---
model.save('neuro_topopt.keras')

# Pick a random sample
test_idx = random.randint(0, len(X_test)-1)
sample_input = X_test[test_idx].reshape(1, 20, 60, 1)
ground_truth = y_test[test_idx].reshape(20, 60)

prediction = model.predict(sample_input).reshape(20, 60)

# Plot
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Input (Thickened)")
plt.imshow(sample_input.reshape(20, 60), cmap='gray')

plt.subplot(1, 3, 2)
plt.title("Ground Truth")
plt.imshow(1 - ground_truth, cmap='gray')

plt.subplot(1, 3, 3)
plt.title("U-Net Prediction")
# Simple threshold
plt.imshow(1 - (prediction > 0.3).astype(float), cmap='gray')

plt.savefig('ai_result_v3.png')
print("Saved 'ai_result_v3.png'")
plt.show()
