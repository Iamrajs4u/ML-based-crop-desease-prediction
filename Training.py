from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import os

# Initialize the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Add a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=10, activation='sigmoid'))

# Compile the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data Augmentation and Preprocessing
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training data
training_set = train_datagen.flow_from_directory(
    'Dataset/train',
    target_size=(128, 128),
    batch_size=6,
    class_mode='categorical'
)

# Load validation data
valid_set = test_datagen.flow_from_directory(
    'Dataset/val',
    target_size=(128, 128),
    batch_size=3,
    class_mode='categorical'
)

# Train the model
classifier.fit_generator(
    training_set,
    steps_per_epoch=20,
    epochs=50,
    validation_data=valid_set
)

# Save the model
classifier_json = classifier.to_json()
with open("model/model1.json", "w") as json_file:
    json_file.write(classifier_json)

# Save the weights
classifier.save_weights("model/my_model_weights.h5")
classifier.save("model/model.h5")
print("Model saved to disk")