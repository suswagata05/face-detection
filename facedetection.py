import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Start the cam
print("To start the camera, type 'start' and press enter:")
while True:
  user_input = input()
  if user_input.lower() == 'start':
    break
  else:
    print("Invalid input. Please type 'start' to begin.")

cap = cv2.VideoCapture(0)  

with mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.5) as face_detection, \
     mp_hands.Hands(min_detection_confidence=0.4, max_num_hands=6, min_tracking_confidence=0.4) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    # To improve performance, optionally mark the image as not writeable
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_face = face_detection.process(image)
    results_hands = hands.process(image)

    # Draw the face detection annotations 
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results_face.detections:
      for detection in results_face.detections:
        mp_drawing.draw_detection(image, detection)
    if results_hands.multi_hand_landmarks:
      for hand_landmarks in results_hands.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(153, 51, 255), thickness=1),
            mp_drawing.DrawingSpec(color=(102, 178, 255), thickness=2))

    # Flip the image horizontally for a selfie-view 
    cv2.imshow('image',cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == ord('x'):
      break
    
cap.release()
