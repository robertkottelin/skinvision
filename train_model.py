import os
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout

# Set up directories
base_dir = 'data'
train_benign_dir = os.path.join(base_dir, 'train', 'benign')
train_malignant_dir = os.path.join(base_dir, 'train', 'malignant')
validation_benign_dir = os.path.join(base_dir, 'validation', 'benign')
validation_malignant_dir = os.path.join(base_dir, 'validation', 'malignant')

# Set up ImageDataGenerators
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(base_dir, 'train'),  # directory should contain subdirectories 'benign' and 'malignant'
    target_size=(299, 299),
    batch_size=32,
    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
    os.path.join(base_dir, 'validation'),  # directory should contain subdirectories 'benign' and 'malignant'
    target_size=(299, 299),
    batch_size=32,
    class_mode='binary')


# Set up model architecture
model = Sequential([
    Conv2D(32, 3, padding='same', activation='relu', input_shape=(299, 299 ,3)),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(128, 3, padding='same', activation='relu'),
    Conv2D(128, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(256, 3, padding='same', activation='relu'),
    Conv2D(256, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])


# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=4,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)

# Save your final model after training on all CSV files
model.save('mark1_skinvision.h5')

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.tight_layout()
plt.show()