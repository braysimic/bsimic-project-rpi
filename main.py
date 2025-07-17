import os
import cv2
import numpy as np
import tensorflow as tf
import datetime
from gpiozero import LED
from time import sleep

# Initialize LEDs
green_led = LED(17)
red_led = LED(27)

# Load labels (make sure labels.txt has one label per line: e.g., "Mask", "No Mask")
with open('labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path='Model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

height = input_details[0]['shape'][1]   # should be 96
width = input_details[0]['shape'][2]    # should be 96
channels = input_details[0]['shape'][3] # should be 1 for grayscale
floating_model = input_details[0]['dtype'] == np.float32

def capture_image():
    # Take picture using libcamera-jpeg
    os.system("libcamera-jpeg -o cam.jpg --width 640 --height 480")

def predict_mask(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None, None

    # Convert to grayscale if model expects 1 channel
    if channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (width, height))
        input_data = np.expand_dims(image, axis=(0, -1))  # shape: (1, 96, 96, 1)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (width, height))
        input_data = np.expand_dims(image, axis=0)        # shape: (1, 96, 96, 3)

    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5
    else:
        input_data = np.uint8(input_data)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    if output_data.dtype == np.uint8:
        output_data = output_data.astype(np.float32)

    try:
        probabilities = tf.nn.softmax(output_data).numpy()
    except Exception:
        probabilities = output_data

    top_idx = np.argmax(probabilities)
    label = labels[top_idx]
    confidence = probabilities[top_idx]

    return label, confidence

def led_indicator(label):
    if "No Mask" in label:
        red_led.on()
        green_led.off()
    else:
        green_led.on()
        red_led.off()

def main():
    print("Capturing image...")
    capture_image()

    print("Running mask prediction...")
    label, confidence = predict_mask("cam.jpg")

    if label is None:
        print("Prediction failed.")
        return

    print(f"Prediction: {label} ({confidence * 100:.2f}%)")
    led_indicator(label)
    sleep(5)
    green_led.off()
    red_led.off()

if __name__ == "__main__":
    main()
