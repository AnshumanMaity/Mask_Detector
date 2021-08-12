# Mask_Detector
Mask and Social Distancing Detector
About:
This a Model that detects Human Face on a live video. It further recognizes that the person has worn a mask or not. It uses Object Detection using Haar feature-based cascade classifiers to detect a person’s face. This model uses  TensorFlow keras library and a pre-trained model to predict whether a person has worn a mask or not. Euclidian distance between centroids of detected face are measured and compared to a fixed value for Displaying the social distancing alert.

Detecting Face:
This model uses OpenCV library to capture frames from the live camera . I have used the haar cascade classifier for detection because it is effective way for face detection. The frame captured is then converted to gray scale for less calculations. Then it loads the haar cascade classifier XML file to detect faces in the frame. Cascade Classifier return a set of coordinates i.e. starting point (x,y) and width(w) and Height(h) of the detected face.

Detecting Mask:
For Mask detection, A pre-trained model is used and TensorFlow keras library is used to load the model. Convolutional Neural Networks (CNN) to classify a person wearing mask or not.. MobileNetV2 model is being used for image pre-processing. Further a cropped image of the face is taken and converted to numpy array . The numpy array is fed to the model for detection of mask. This returns two percentage namely mask percentage and without mask percentage. If the mask percentage is more than the without mask percentage then “Mask ON” is labelled.  

Maintain Social distance Alert:
For every detected face coordinates, the centroid is calculated . Then for every pair of face detected Euclidean distance is calculated between two centroids. If the Euclidean distance calculated is smaller than the minimum required distance then an alert for Maintaining social distance is displayed. I have used the minimum distance a 500 pixels. Further camera calibaration can be done to improve the results.
Video Link: https://drive.google.com/file/d/18Hz3MScHbvsLZ2_0vTGoDyftSlQOpBR1/view?usp=sharing
