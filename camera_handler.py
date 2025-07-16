import numpy as np
import tensorflow as tf
import datetime
from gpiozero import LED
from time import sleep
from picamera2 import Picamera2
import cv2
import firebase_setup
import constant

# Firebase setup
collection = firebase_setup.db.collection(constant.COLLECTION_NAME)
doc_ref = collection.document(constant.DOCUMENT_MASK)

# LEDs
green_led = LED(17)
red_led = LED(27)

# Load labels
with open('labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='model4.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
floating_model = input_details[0]['dtype'] == np.float32

# Initialize camera
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(main={"size": (width, height)})
picam2.configure(camera_config)
picam2.start()

def preprocess_frame(frame):
    # frame is already (width, height) from picamera2 config, but confirm
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(image_rgb, axis=0)
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5
    else:
        input_data = np.uint8(input_data)
    return input_data

def predict_mask(input_data):
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    if output.dtype in [np.float32, np.float64]:
        probabilities = tf.nn.softmax(output).numpy()
    else:
        probabilities = output / 255.0
    return probabilities

def update_leds(label):
    if "No Mask" in label:
        red_led.on()
        green_led.off()
    else:
        green_led.on()
        red_led.off()

def main():
    try:
        while True:
            frame = picam2.capture_array()
            input_data = preprocess_frame(frame)
            probabilities = predict_mask(input_data)
            top_idx = int(np.argmax(probabilities))
            label = labels[top_idx]
            confidence = float(probabilities[top_idx])

            print(f"Prediction: {label} ({confidence * 100:.2f}%)")

            update_leds(label)

            # Show image with label (optional)
            cv2.putText(frame, f"{label}: {confidence*100:.2f}%", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if "Mask" in label else (0,0,255), 2)
            cv2.imshow("Mask Detector", frame)

            # Update Firebase
            doc_ref.set({
                "mask_status": label,
                "confidence": round(confidence * 100, 2),
                "timestamp": datetime.datetime.now().isoformat()
            })

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            sleep(0.1)

    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        red_led.off()
        green_led.off()

if __name__ == "__main__":
    main()






