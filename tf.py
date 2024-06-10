import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# Load the pre-trained object detection model
model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
model = hub.load(model_url)

# Function to preprocess the image
def preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.uint8)
    return img

# Path to the input image
image_path = "D:\Programming\Python\Simulator - Denis\Simulator\images\images.jpeg" # UPDATE WITH YOUR LOCAL PATH
input_image = preprocess_image(image_path)

# Add batch dimension
input_tensor = tf.expand_dims(input_image, axis=0)

# Perform object detection
detector_output = model(input_tensor)

# Extract detection results
boxes = detector_output['detection_boxes'][0].numpy()
scores = detector_output['detection_scores'][0].numpy()
classes = detector_output['detection_classes'][0].numpy().astype(np.int32)

# COCO labels
coco_labels = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train',
    8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter',
    15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant',
    23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie',
    33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
    40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass',
    47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich',
    55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake',
    62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv',
    73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
    80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
    88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'
}

# Function to draw bounding boxes on the image
def draw_boxes(image, boxes, scores, classes, labels, score_threshold=0.5):
    image = np.array(image)
    height, width, _ = image.shape
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    
    for i in range(len(boxes)):
        if scores[i] >= score_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            ymin, xmin, ymax, xmax = int(ymin * height), int(xmin * width), int(ymax * height), int(xmax * width)
            draw.rectangle([(xmin, ymin), (xmax, ymax)], outline='red', width=2)
            label = labels.get(classes[i], 'N/A')
            draw.text((xmin, ymin), f"{label}: {scores[i]:.2f}", fill='red')
    
    return image

# Draw bounding boxes on the original image
output_image = draw_boxes(input_image, boxes, scores, classes, coco_labels)

# Print detected objects
for i in range(len(boxes)):
    if scores[i] >= 0.5:
        ymin, xmin, ymax, xmax = boxes[i]
        label = coco_labels.get(classes[i], 'N/A')
        print(f"Object: {label}, Score: {scores[i]:.2f}, Box: ({ymin:.2f}, {xmin:.2f}, {ymax:.2f}, {xmax:.2f})")

# Save and display the resulting image
output_image_path = "output_image.jpg"
output_image.save(output_image_path)

print(f"Annotated image saved to {output_image_path}")
