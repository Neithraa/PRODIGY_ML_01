#__Load Data__
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths
train_dir = 'path/to/your/train'
test_dir = 'path/to/your/test'

# Data generators
train_datagen = ImageDataGenerator(rescale=0.255, validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=0.255)

# Load and preprocess the data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical')

#__Feature Extraction__
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input

# Load pre-trained MobileNetV2 model
base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')

# Freeze the base model
base_model.trainable = False

# Add custom layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Classification output
num_classes = train_generator.num_classes
output_labels = Dense(num_classes, activation='softmax', name='labels')(x)

# Define the model
model = Model(inputs=base_model.input, outputs=output_labels)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#__Training and Evaluation__
# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test Accuracy: {test_accuracy}')

# Predict an example image
import numpy as np
from tensorflow.keras.preprocessing import image

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    
    return predicted_class

# Example usage
img_path = 'path/to/your/image.jpg'
predicted_class = predict_image(img_path)
print(f'Predicted Class: {predicted_class}')
