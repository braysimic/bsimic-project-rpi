from picamera2 import Picamera2
import time
from datetime import timedelta
import constant
import firebase_setup

camera = Picamera2()
camera_config = camera.create_still_configuration(main={"size": (800, 600)})
camera.configure(camera_config)

collection = firebase_setup.db.collection(constant.COLLECTION_NAME)
doc_camera_ref = collection.document(constant.DOCUMENT_CAMERA)
imageUploadPath = constant.STORAGE_FOLDER_CAMERA_IMAGES + '/' + constant.CAMERA_IMAGE_NAME

def capture():
	camera.start()
	time.sleep(1)
	camera.capture_file(constant.CAMERA_IMAGE_NAME)
	camera.stop()
	blob = firebase_setup.bucket.blob(imageUploadPath)
	url = blob.generate_signed_url(
		expiration=timedelta(days=30),
 		method="GET"
	)
	blob.upload_from_filename(constant.CAMERA_IMAGE_NAME)
	print(url)

	doc_camera_ref.update({
		'url': url,
		'timestamp': time.time_ns() # ns since Epoch (Jan 1, 1970)
	})