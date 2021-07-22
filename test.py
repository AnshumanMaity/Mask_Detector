import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_model("mask_recog_ver2.h5")

# To capture video from webcam. 
cap = cv2.VideoCapture(0)
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')

while True:
    # Read the frame
    _, img = cap.read()
    img=cv2.flip(img,1,1)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.07, 4)

    font = cv2.FONT_HERSHEY_SIMPLEX
    preds=[]

    # Check if there is a face or not
    if len(faces)>0:
    	fr=1

    	# For each detected face
    	for (x, y, w, h) in faces:

    		# Drawing a rectangle around the face
	    	cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)



	    	# Passing the face to the model 
	    	face_frame = img[y:y+h,x:x+w]
	    	face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
	    	face_frame = cv2.resize(face_frame, (224, 224))
	    	face_frame = img_to_array(face_frame)
	    	face_frame = np.expand_dims(face_frame, axis=0)
	    	face_frame =  preprocess_input(face_frame)
	    	preds = model.predict(face_frame)
	    	for pred in preds:
	    		(mask, withoutMask) = pred
	    	label = "Mask ON" if mask > withoutMask else "No Mask"
	    	color = (0, 255, 0) if label == "Mask ON" else (0, 0, 255)

	    	# Priinting the Labels and Unique Id to that person
	    	cv2.putText(img,'#0'+str(fr)+"  "+label , (x, y), font, 0.5, color, 1, cv2.LINE_4)
	    	fr+=1


    # If no person detected
    else:
    	cv2.putText(img, 'NO PERSON DETECTED', (50, 50), font, 1, (0, 0, 255), 2, cv2.LINE_4)

    # Display
    cv2.imshow('img', img)

    # Stop if escape key is pressed
    k = cv2.waitKey(1) & 0xff
    if k==27:
        break
        
# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()
