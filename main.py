import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_faces(img) : # Defining a fonction that detects faces.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Converting input color to a grey scale, using OpenCV. 
    faces = face_classifier.detectMultiScale(gray, 1.3, 5) # Face identifier using the pre-trained classifier dataset.
    if faces is () :
        return img 
    
    for (x, y, w, h) in faces :
        cv2.rectangle(img,(x,y),(x + w, y + h), (255,0,0),2) # Defining a white square.
    return img

cap = cv2.VideoCapture(0) # Select the first camera.

while True : # Create a loop for capture and display images continuously.
    ret, frame = cap.read() # Reading images through the camera.
    frame = detect_faces(frame)

    cv2.imshow('Video Face Detection', frame) # Display window for the camera. 

    if cv2.waitKey(1) & 0xFF == ord('q') : # Waiting for the user to press "q" keyboard touch to stop and quit the loop.
        break

cap.release() # Release the camera and close the window.
cv2.destroyAllWindows()
