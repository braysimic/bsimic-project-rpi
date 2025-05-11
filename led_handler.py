from gpiozero import LED
import constant
import firebase_setup

led1 = LED(21)  # Light
collection = firebase_setup.db.collection(constant.COLLECTION_NAME)
doc_led_ref = collection.document(constant.DOCUMENT_LEDS)

# Create a callback on_snapshot function to capture changes
def on_leddoc_snapshot(doc_snapshot, changes, read_time):
	for doc in doc_snapshot:
		print(f'Received document snapshot: {doc.to_dict()}')
		led1_status = doc.to_dict()["led1"]
		print(f'LED1 {led1_status}')
		if led1_status == True:
			led1.on()
		else:
			led1.off()

def add_led_snapshot_listener():
	# Watch the LED document
	doc_led_ref.on_snapshot(on_leddoc_snapshot)