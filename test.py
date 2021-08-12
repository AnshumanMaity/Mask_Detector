from scipy.spatial import distance as dist
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import math
import time
MIN_DISTANCE = 500


# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_model("mask_recog_ver2.h5")

# To capture video from webcam.
cap = cv2.VideoCapture(0)
# To use a video file as input
# cap = cv2.VideoCapture('filename.mp4')q
pTime=time.time()

while True:
    fr=1
    # Read the frame
    _, img = cap.read()
    img = cv2.flip(img, 1, 1)
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,f'FPS:{int(fps)}',(5,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv2.LINE_4)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.07, 4)

    font = cv2.FONT_HERSHEY_SIMPLEX
    preds = []

    # Check if there is a face or not
    if len(faces) > 0:
        violate=set()
        if len(faces) >= 2:
            for i in range(0, len(faces)):
                for j in range(i+1, len(faces)):
                    x1 = (faces[i][0]+(faces[i][2]/2))
                    y1 = (faces[i][1]+(faces[i][3]/2))
                    x2 = (faces[j][0]+(faces[j][2])/2)
                    y2 = (faces[j][1]+(faces[j][3]/2))
                    dis = math.sqrt(math.pow((x1-x2), 2)+math.pow((y1-y2), 2))
                    print(dis)

                    if dis < MIN_DISTANCE:
                        cv2.putText(img,"Maintain Social Distancing", (100, 50),font, 1, (0,215,255), 4, cv2.LINE_4)
                        violate.add(i+1)
                        violate.add(j+1)
        
        # For each detected face
        no=0
        for (x,y,w,h) in faces:
            no+=1            
            color = (255, 255, 255) 
            if no in violate:
                color=(0,255,255)               
            
            # Drawing a rectangle around the face
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

            # Passing the face to the model
            face_frame = img[y:y+h, x:x+w]
            face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
            face_frame = cv2.resize(face_frame, (224, 224))
            face_frame = img_to_array(face_frame)
            face_frame = np.expand_dims(face_frame, axis=0)
            face_frame = preprocess_input(face_frame)
            preds = model.predict(face_frame)
            for pred in preds:
                (mask, withoutMask) = pred
            label = "Mask ON" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask ON" else (0, 0, 255)

            # Priinting the Labels and Unique Id to that person
            cv2.putText(img, 'ID'+str(fr)+"  "+label, (x, y),font, 0.7, color, 2, cv2.LINE_4)
            fr += 1

    # If no person detected
    else:
        cv2.putText(img, 'NO PERSON DETECTED', (50, 50),font, 1, (0, 0, 255), 4, cv2.LINE_4)

    # Display
    cv2.namedWindow('img',cv2.WINDOW_NORMAL)
    cv2.imshow('img', img)

    # Stop if escape key is pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()
