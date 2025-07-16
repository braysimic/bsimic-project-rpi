from gpiozero import Button
from signal import pause

# Connect button to GPIO18 (BCM 18, physical pin 12)
button = Button(19, pull_up=False)

def on_press():
    print("Button Pressed")

def on_release():
    print("Button Released")

button.when_pressed = on_press
button.when_released = on_release

pause()