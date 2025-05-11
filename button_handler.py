from gpiozero import Button
import firebase_setup
import constant
import camera_handler

collection = firebase_setup.db.collection(constant.COLLECTION_NAME)
doc_button_ref = collection.document(constant.DOCUMENT_BUTTONS)

button1 = Button(19)

def button_pressed():
	print('Button pressed')
	doc_button_ref.update({u'button1': True})
	# camera still shot
	camera_handler.capture()
	print('Camera capture done')

def button_released():
	print('Button released')
	doc_button_ref.update({u'button1': False})

def add_button_event_handlers():
	button1.when_pressed = button_pressed
	button1.when_released = button_released