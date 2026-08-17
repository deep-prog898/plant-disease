import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1)
])
# Dataset paths
train_path = r"C:\Users\deep\Downloads\photo\tomato\train"
val_path = r"C:\Users\deep\Downloads\photo\tomato\val"

# Load training dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=(224, 224),
    batch_size=32,
    label_mode="int"
)

# Load validation dataset
val_dataset = tf.keras.utils.image_dataset_from_directory(
    val_path,
    image_size=(224, 224),
    batch_size=32,
    label_mode="int"
)
    # Save class names before preprocessing
class_names = train_dataset.class_names

# Apply augmentation to training images
def augment_images(images, labels):
    images = data_augmentation(images, training=True)
    return images, labels

train_dataset = train_dataset.map(augment_images)

# Apply MobileNetV2 preprocessing
def preprocess_images(images, labels):
    images = preprocess_input(images)
    return images, labels

train_dataset = train_dataset.map(preprocess_images)
val_dataset = val_dataset.map(preprocess_images)

print("MobileNetV2 preprocessing applied!")

# Display class names
print("Classes:", class_names)

print("Training dataset loaded successfully!")
print("Validation dataset loaded successfully!")

# Get one batch of images and labels
for images, labels in train_dataset.take(1):
    print("Image shape:", images.shape)
    print("Label shape:", labels.shape)
    print("Pixel value range:", images.numpy().min(), "to", images.numpy().max())
    print("First 10 labels:", labels.numpy()[:10])
import matplotlib.pyplot as plt

# Take one batch from the training dataset
for images, labels in train_dataset.take(1):
  
  original_image = images[0:1]

# Apply augmentation
augmented_image = data_augmentation(original_image, training=True)

    # Display both images
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow((original_image[0] + 1) / 2)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow((augmented_image[0] + 1) / 2)
plt.title("Augmented")
plt.axis("off")

plt.show()