from gpiozero import DistanceSensor
from time import sleep

## Ultrasonic distance sensor HC-SR04
## wiring: echo => GPIO24, trigger => GPIO18

sensor = DistanceSensor(echo=24, trigger=18, max_distance=4)

while True:
    print('Distance: ', sensor.distance * 100, 'cm')
    sleep(1)