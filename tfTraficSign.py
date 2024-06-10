# NOT WORKING YET

import tensorflow as tf
import tensorflow_hub as hub
import cv2

# Load the pre-trained MobileNetV2 model from TensorFlow Hub
model_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
feature_extractor_layer = hub.KerasLayer(model_url, input_shape=(224, 224, 3), trainable=False)

# Define the rest of the model using the functional API
inputs = tf.keras.Input(shape=(224, 224, 3))
x = feature_extractor_layer(inputs)
outputs = tf.keras.layers.Dense(8, activation='softmax')(x)  # Assuming 8 classes for speed limit signs
model = tf.keras.Model(inputs, outputs)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display model summary
model.summary()

# Define a function to preprocess the input image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Resize to the input shape required by the model
    img = img / 255.0  # Normalize to [0, 1]
    img = tf.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Define a function to decode the predictions
def decode_predictions(preds):
    class_names = ['Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 'Speed limit (60km/h)',
                   'Speed limit (70km/h)', 'Speed limit (80km/h)', 'Speed limit (100km/h)', 'Speed limit (120km/h)']
    predicted_class = tf.argmax(preds, axis=1)
    return class_names[predicted_class[0]]

# Load and preprocess an image
image_path = "/Users/sofianurobert/Projects/tensorflow/images/images.png"
input_image = preprocess_image(image_path)

# Make predictions
predictions = model.predict(input_image)
predicted_label = decode_predictions(predictions)

print(f"Predicted Speed Limit: {predicted_label}")
