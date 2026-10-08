import tensorflow as tf
import matplotlib.pyplot as plt


# ==========================================
# 1. LOAD TRAINING DATA
# ==========================================

train_data = tf.keras.preprocessing.image_dataset_from_directory(
    "Training",
    image_size=(128, 128),
    batch_size=32
)


# ==========================================
# 2. LOAD TESTING DATA
# ==========================================

test_data = tf.keras.preprocessing.image_dataset_from_directory(
    "Testing",
    image_size=(128, 128),
    batch_size=32
)


# ==========================================
# 3. BUILD CNN MODEL
# ==========================================

model = tf.keras.Sequential([

    # Normalize pixel values from 0-255 to 0-1
    tf.keras.layers.Rescaling(1./255),

    # First Convolution Block
    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation='relu'
    ),

    tf.keras.layers.MaxPooling2D(),

    # Second Convolution Block
    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation='relu'
    ),

    tf.keras.layers.MaxPooling2D(),

    # Flatten feature maps
    tf.keras.layers.Flatten(),

    # Fully Connected Layer
    tf.keras.layers.Dense(
        128,
        activation='relu'
    ),

    # Output Layer - 4 Classes
    tf.keras.layers.Dense(
        4,
        activation='softmax'
    )
])


# ==========================================
# 4. COMPILE MODEL
# ==========================================

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# ==========================================
# 5. DISPLAY MODEL SUMMARY
# ==========================================

print("\n===== CNN MODEL SUMMARY =====")

model.summary()


# ==========================================
# 6. TRAIN CNN
# ==========================================

print("\n===== STARTING TRAINING =====")

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=5
)


# ==========================================
# 7. FINAL TRAINING RESULTS
# ==========================================

final_train_loss = history.history['loss'][-1]
final_train_accuracy = history.history['accuracy'][-1]

final_val_loss = history.history['val_loss'][-1]
final_val_accuracy = history.history['val_accuracy'][-1]


print("\n======================================")
print("       FINAL TRAINING RESULTS")
print("======================================")

print(f"Training Loss       : {final_train_loss:.4f}")
print(f"Training Accuracy   : {final_train_accuracy * 100:.2f}%")

print(f"Validation Loss     : {final_val_loss:.4f}")
print(f"Validation Accuracy : {final_val_accuracy * 100:.2f}%")

print("======================================")


# ==========================================
# 8. ACCURACY GRAPH
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title("CNN Training and Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()
plt.grid()

plt.tight_layout()

plt.show()


# ==========================================
# 9. LOSS GRAPH
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history['loss'],
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.title("CNN Training and Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid()

plt.tight_layout()

plt.show()


# ==========================================
# 10. RESULTS TABLE
# ==========================================

print("\n======================================")
print("           RESULTS TABLE")
print("======================================")

print(f"{'Metric':<25} {'Value'}")
print("--------------------------------------")

print(
    f"{'Training Loss':<25} "
    f"{final_train_loss:.4f}"
)

print(
    f"{'Training Accuracy':<25} "
    f"{final_train_accuracy * 100:.2f}%"
)

print(
    f"{'Validation Loss':<25} "
    f"{final_val_loss:.4f}"
)

print(
    f"{'Validation Accuracy':<25} "
    f"{final_val_accuracy * 100:.2f}%"
)

print("======================================")


# ==========================================
# 11. SAVE MODEL
# ==========================================

model.save("brain_tumor_model.h5")

print("\nModel saved successfully!")
print("File: brain_tumor_model.h5")