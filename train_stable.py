import random
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input, optimizers
import matplotlib.pyplot as plt

# --- 1. LOAD DATA ---
print("Loading data...")
load_files = sorted(os.listdir('data/loads'))
struct_files = sorted(os.listdir('data/structures'))

X = []
y = []

for f_load, f_struct in zip(load_files, struct_files):
    if f_load.endswith('.npy'):
        load_grid = np.load(f'data/loads/{f_load}')
        struct_grid = np.load(f'data/structures/{f_struct}')

        # --- THICKEN INPUT (Make the dot visible) ---
        rows, cols = np.where(load_grid == 1.0)
        if len(rows) > 0:
            r, c = rows[0], cols[0]
            r_min, r_max = max(0, r-1), min(19, r+1)
            c_min, c_max = max(0, c-1), min(59, c+1)
            load_grid[r_min:r_max+1, c_min:c_max+1] = 1.0

        X.append(load_grid)
        y.append(struct_grid)

X = np.array(X).reshape(-1, 20, 60, 1)
y = np.array(y).reshape(-1, 20, 60, 1)

# Double check data isn't broken
print(f"Data Stats -> Min: {np.min(X)}, Max: {np.max(X)}")
if np.isnan(X).any() or np.isnan(y).any():
    print("CRITICAL WARNING: Data contains NaNs!")

# Split
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# --- 2. BUILD STABLE U-NET ---


def build_unet():
    inputs = Input(shape=(20, 60, 1))

    # Encoder
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(
        inputs)  # Reduced filters for stability
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Bottleneck
    b = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p2)

    # Decoder
    u1 = layers.UpSampling2D((2, 2))(b)
    cat1 = layers.Concatenate()([u1, c2])
    c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(cat1)

    u2 = layers.UpSampling2D((2, 2))(c3)
    cat2 = layers.Concatenate()([u2, c1])
    c4 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(cat2)

    outputs = layers.Conv2D(
        1, (3, 3), activation='sigmoid', padding='same')(c4)

    return models.Model(inputs, outputs)


model = build_unet()

# --- THE FIX IS HERE ---
# We use a smaller learning rate (0.001 -> 0.0001)
# We add 'clipnorm=1.0' to prevent exploding gradients (NaNs)
opt = optimizers.Adam(learning_rate=0.0005, clipnorm=1.0)

model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['mae'])
# Switched to binary_crossentropy as it's more stable for black/white images than MAE

# --- 3. TRAIN ---
print("\nStarting Stable Training...")
history = model.fit(X_train, y_train, epochs=200,
                    batch_size=16, validation_data=(X_test, y_test))

# --- 4. TEST ---
model.save('neuro_topopt.keras')

# Visualization
test_idx = random.randint(0, len(X_test)-1)
sample_input = X_test[test_idx].reshape(1, 20, 60, 1)
ground_truth = y_test[test_idx].reshape(20, 60)
prediction = model.predict(sample_input).reshape(20, 60)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Input")
plt.imshow(sample_input.reshape(20, 60), cmap='gray')
plt.subplot(1, 3, 2)
plt.title("Ground Truth")
plt.imshow(1 - ground_truth, cmap='gray')
plt.subplot(1, 3, 3)
plt.title("Stable AI Prediction")
plt.imshow(1 - prediction, cmap='gray')
plt.savefig('ai_result_stable.png')
print("Saved 'ai_result_stable.png'")
plt.show()
