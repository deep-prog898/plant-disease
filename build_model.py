import tensorflow as tf
from load_dataset import train_dataset, val_dataset

# Load MobileNetV2 with pre-trained ImageNet weights
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

print("MobileNetV2 loaded successfully!")
# Freeze the MobileNetV2 base model
base_model.trainable = False

# Add classification layers
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(7, activation="softmax")
])

# Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Model compiled successfully!")
print("Classification head added!")
print("MobileNetV2 base model frozen!")
# Display model summary
model.summary()
# Training configuration
epochs = 10

print("Training configuration ready!")
print("Epochs:", epochs)
print("Batch size: 32")
print("Training images: 7000")
print("Validation images: 700")
# Train the model
print("Starting training...")

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs
)

print("Training completed!")
# Display final training results
print("\nFinal Results:")
print("Training Accuracy:", history.history["accuracy"][-1])
print("Validation Accuracy:", history.history["val_accuracy"][-1])
print("Training Loss:", history.history["loss"][-1])
print("Validation Loss:", history.history["val_loss"][-1])