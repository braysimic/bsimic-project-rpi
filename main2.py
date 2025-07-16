from signal import pause
from time import sleep

import firebase_setup
import constant
import camera_handler
import led_handler
import button_handler
import ultrasonic_sensor_handler

#######################################
##### Simulattion of Light Control ####
#######################################

led_handler.add_led_snapshot_listener()

#####################################################
##### Simulation of Door monitoring & Camera shot ###
#####################################################

button_handler.add_button_event_handlers()

##########################################
## Check each unit that requires polling #
##########################################

while True:
    ultrasonic_sensor_handler.check_distance() # ultrasonic distance sensor
    sleep(1) # sleep before continuing next round of polling

# never reaches here
pause()
