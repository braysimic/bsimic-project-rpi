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
floating_model = input_details[0]['dtype'] == np.float32

def capture_image():
    # Take picture using libcamera-jpeg
    os.system("libcamera-jpeg -o cam.jpg --width 640 --height 480")

def predict_mask(image_path):
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None, None

    # Preprocess image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    # If output is logits or scores, convert to probabilities
    try:
        probabilities = tf.nn.softmax(output_data).numpy()
    except Exception:
        # If output is already probabilities
        probabilities = output_data

    top_idx = np.argmax(probabilities)
    label = labels[top_idx]
    confidence = probabilities[top_idx]

    return label, confidence

def led_indicator(label):
    if "No Mask" in label:
        red_led.off()
        green_led.on()
    else:
        green_led.off()
        red_led.on()

def main():
    print("Capturing image...")
    capture_image()

    print("Running mask prediction...")
    label, confidence = predict_mask("cam.jpg")

    if label is None:
        print("Prediction failed.")
        return

    print(f"Prediction: {label} ({confidence * 100:.2f}%)")

    # Turn on LED accordingly
    led_indicator(label)

    # Keep LED on for 5 seconds
    sleep(5)

    # Turn off LEDs
    green_led.off()
    red_led.off()

if __name__ == "__main__":
    main()
