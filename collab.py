import kagglehub
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam

# Download the dataset
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
print("Path to dataset files:", path)

# Dataset paths
train_dir = os.path.join(path, 'chest_xray', 'train')
val_dir = os.path.join(path, 'chest_xray', 'val')
test_dir = os.path.join(path, 'chest_xray', 'test')

# Image parameters
img_height, img_width = 224, 224  # VGG16 expects 224x224
batch_size = 32

# Data generators with augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Function to plot training history
def plot_history(history, model_name):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} Loss')
    plt.legend()
    
    plt.show()

# 1. Custom CNN Model
def build_cnn():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

cnn_model = build_cnn()
print("Custom CNN Model Summary:")
cnn_model.summary()

# Train CNN
epochs = 1  # Adjust as needed for better results
cnn_history = cnn_model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator
)

# Evaluate CNN on test set
cnn_test_loss, cnn_test_acc = cnn_model.evaluate(test_generator)
print(f"CNN Test Accuracy: {cnn_test_acc:.4f}")
print(f"CNN Test Loss: {cnn_test_loss:.4f}")

# Predictions for confusion matrix
cnn_predictions = cnn_model.predict(test_generator)
cnn_pred_classes = (cnn_predictions > 0.5).astype(int).flatten()
cnn_true_classes = test_generator.classes

# Confusion Matrix for CNN
cm_cnn = confusion_matrix(cnn_true_classes, cnn_pred_classes)
plt.figure(figsize=(6, 6))
sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
plt.title('CNN Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Plot CNN history
plot_history(cnn_history, 'Custom CNN')

# 2. VGG16 Model with Transfer Learning
def build_vgg16():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    base_model.trainable = False  # Freeze base layers
    
    inputs = Input(shape=(img_height, img_width, 3))
    x = base_model(inputs, training=False)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

vgg_model = build_vgg16()
print("VGG16 Model Summary:")
vgg_model.summary()

# Train VGG16
vgg_history = vgg_model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator
)

# Evaluate VGG16 on test set
vgg_test_loss, vgg_test_acc = vgg_model.evaluate(test_generator)
print(f"VGG16 Test Accuracy: {vgg_test_acc:.4f}")
print(f"VGG16 Test Loss: {vgg_test_loss:.4f}")

# Predictions for confusion matrix
vgg_predictions = vgg_model.predict(test_generator)
vgg_pred_classes = (vgg_predictions > 0.5).astype(int).flatten()
vgg_true_classes = test_generator.classes

# Confusion Matrix for VGG16
cm_vgg = confusion_matrix(vgg_true_classes, vgg_pred_classes)
plt.figure(figsize=(6, 6))
sns.heatmap(cm_vgg, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
plt.title('VGG16 Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Plot VGG16 history
plot_history(vgg_history, 'VGG16')

# Summary of Results
print("\n=== Model Comparison ===")
print(f"Custom CNN - Test Accuracy: {cnn_test_acc:.4f}, Test Loss: {cnn_test_loss:.4f}")
print(f"VGG16 - Test Accuracy: {vgg_test_acc:.4f}, Test Loss: {vgg_test_loss:.4f}")

# Classification Reports
print("\nCNN Classification Report:")
print(classification_report(cnn_true_classes, cnn_pred_classes, target_names=['Normal', 'Pneumonia']))

print("\nVGG16 Classification Report:")
print(classification_report(vgg_true_classes, vgg_pred_classes, target_names=['Normal', 'Pneumonia']))
