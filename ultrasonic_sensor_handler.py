from gpiozero import DistanceSensor
import firebase_setup
import constant

## ultrasonic distance sensor
sensor = DistanceSensor(echo=24, trigger=18, max_distance=4)

currentDistance = -100
collection = firebase_setup.db.collection(constant.COLLECTION_NAME)
distance_sensor_ref = collection.document(constant.DOCUMENT_DISTANCE)

def check_distance():
	global sensor
	global currentDistance
	newDistance =  sensor.distance * 100 ## unit = cm
	delta = abs(currentDistance - newDistance)
	if delta > 5: # at least 5cm change
		currentDistance = newDistance
		print('Distance: ', currentDistance, 'cm')
		distance_sensor_ref.update({u'sensor1': currentDistance})