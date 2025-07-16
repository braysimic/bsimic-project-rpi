import os
import cv2
import numpy as np
import tensorflow as tf

# Load labels
with open('labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path='model4.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

images = ('1.jpeg', '2.jpg', '3.jpg', '4.jpg')

for img in images:
    print(f"Processing {img}...")
    image = cv2.imread(img)
    if image is None:
        print(f"Could not load image {img}")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height))

    # Check dtype expected by model
    if input_details[0]['dtype'] == np.uint8:
        input_data = np.expand_dims(image_resized, axis=0).astype(np.uint8)
    else:
        input_data = np.expand_dims(image_resized / 255.0, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])[0]

    # Attempt to get probabilities
    try:
        probabilities = tf.nn.softmax(output).numpy()
    except Exception:
        probabilities = output

    # Clamp confidence between 0 and 1
    class_id = np.argmax(probabilities)
    confidence = probabilities[class_id]
    confidence = max(0.0, min(confidence, 1.0))

    label = labels[class_id]
    print(f"{img} → Prediction: {label} ({confidence * 100:.2f}%)")

print("Done processing images.")
