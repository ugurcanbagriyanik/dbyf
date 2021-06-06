import mediapipe as mp
import cv2
import winsound
from math import sqrt
frequency = 3000  # Set Frequency To 2500 Hertz
duration = 100  # Set Duration To 1000 ms == 1 second


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
face_mesh= mp_face_mesh.FaceMesh()

cap = cv2.VideoCapture(0)
with mp_hands.Hands(max_num_hands=2,min_detection_confidence=0.7)  as hands:
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        resultFace=face_mesh.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw face landmarks
        if results.multi_hand_landmarks != None and resultFace.multi_face_landmarks != None :
            #mp_drawing.draw_landmarks(image, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS) #show the skeleton
            fingerTip = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            lips= resultFace.multi_face_landmarks[0].landmark[0]
            if sqrt(abs( (lips.x-fingerTip.x)**2 + (lips.y -fingerTip.y)**2 )) < 0.06:
                 winsound.Beep(frequency, duration)
                        
        cv2.imshow('Dont Bite Your Nail', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()