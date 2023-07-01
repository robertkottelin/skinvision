import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

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
    target_size=(150, 150),
    batch_size=5,
    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
    os.path.join(base_dir, 'validation'),  # directory should contain subdirectories 'benign' and 'malignant'
    target_size=(150, 150),
    batch_size=5,
    class_mode='binary')


# Set up model architecture
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(150, 150 ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
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
    epochs=10,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)
