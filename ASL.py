import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

class GestureControl:
    def __init__(self):
        self.finger_mode_enabled = False  # Default mode: Hand Gesture Detection
        self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils
    
    def detect_gesture(self, landmarks):
        mp_hands = self.mp_hands  # Fix reference to self.mp_hands
        
        open_hand = all(landmarks[i].y < landmarks[i-2].y for i in range(8, 21, 4))  

        love_you = (landmarks[mp_hands.HandLandmark.THUMB_TIP].y < landmarks[mp_hands.HandLandmark.THUMB_IP].y and  # Thumb up
        landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP].y and  # Index up
        landmarks[mp_hands.HandLandmark.PINKY_TIP].y < landmarks[mp_hands.HandLandmark.PINKY_DIP].y and  # Pinky up
        landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y and  # Middle down
        landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y > landmarks[mp_hands.HandLandmark.RING_FINGER_DIP].y)  # Ring down  
    
        thumb_up = (landmarks[mp_hands.HandLandmark.THUMB_TIP].y < landmarks[mp_hands.HandLandmark.THUMB_IP].y and  
        landmarks[mp_hands.HandLandmark.PINKY_TIP].y > landmarks[mp_hands.HandLandmark.PINKY_DIP].y and 
        landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y > landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP].y and 
        landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y and
        landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y > landmarks[mp_hands.HandLandmark.RING_FINGER_DIP].y)

        yes = (landmarks[mp_hands.HandLandmark.THUMB_TIP].x > landmarks[mp_hands.HandLandmark.THUMB_IP].x and  # Thumb curled
        landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y > landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and  # Index curled
        landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and  # Middle curled
        landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y > landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y and  # Ring curled
        landmarks[mp_hands.HandLandmark.PINKY_TIP].y > landmarks[mp_hands.HandLandmark.PINKY_PIP].y)  # Pinky curled

        no = (landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and  # Index extended
        landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and  # Middle extended
        landmarks[mp_hands.HandLandmark.THUMB_TIP].x > landmarks[mp_hands.HandLandmark.THUMB_IP].x and  # Thumb curled
        landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y > landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y and  # Ring curled
        landmarks[mp_hands.HandLandmark.PINKY_TIP].y > landmarks[mp_hands.HandLandmark.PINKY_PIP].y)  # Pinky curled


        okay = (landmarks[mp_hands.HandLandmark.THUMB_TIP].x > landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x and  # Thumb touching index
        landmarks[mp_hands.HandLandmark.THUMB_TIP].y > landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP].y and  # Circular shape with thumb and index
        landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y and  # Middle finger up
        landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.RING_FINGER_DIP].y and  # Ring finger up
        landmarks[mp_hands.HandLandmark.PINKY_TIP].y < landmarks[mp_hands.HandLandmark.PINKY_DIP].y)  # Pinky up

        peace = (landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and  # Index finger up
        landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and  # Middle finger up
        landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y > landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y and  # Ring finger down
        landmarks[mp_hands.HandLandmark.PINKY_TIP].y > landmarks[mp_hands.HandLandmark.PINKY_PIP].y and  # Pinky down
        landmarks[mp_hands.HandLandmark.THUMB_TIP].x < landmarks[mp_hands.HandLandmark.THUMB_IP].x)  # Thumb curled inward


        if love_you:  
            return "I Love You"  
        elif thumb_up:  
            return "Thumb Up"  
        elif peace:
            return "Peace"
        elif open_hand:  
            return "Open Hand" 
        elif okay:
            return "Okay"
        elif yes:
            return "Yes" 
        elif no:
            return "No"
        else:
            return "Unknown Gesture"
    
    def get_fingers_up():
        cap = cv2.VideoCapture(0)
        cap.set(3, 1280)
        cap.set(4, 720)

        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.75, min_tracking_confidence=0.75)
        mp_draw = mp.solutions.drawing_utils

        screen_width, screen_height = pyautogui.size()
        prev_x, prev_y = 0, 0
        smoothening = 7
        p_time = 0

        def count_fingers(hand_landmarks):
            fingers = []

            # Thumb
            if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
                fingers.append(1)
            else:
                fingers.append(0)

            # Fingers
            for tip in [8, 12, 16, 20]:
                if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
                    fingers.append(1)
                else:
                    fingers.append(0)
            return fingers

        while True:
            img = cap.read()
            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            h, w, _ = img.shape

            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

                    index_tip = handLms.landmark[8]
                    fingers = count_fingers(handLms)

                    x = int(index_tip.x * w)
                    y = int(index_tip.y * h)
                    screen_x = np.interp(x, (0, w), (0, screen_width))
                    screen_y = np.interp(y, (0, h), (0, screen_height))

                    if fingers == [0, 0, 0, 0, 0]:
                        curr_x = prev_x + (screen_x - prev_x) / smoothening
                        curr_y = prev_y + (screen_y - prev_y) / smoothening
                        pyautogui.moveTo(curr_x, curr_y)
                        prev_x, prev_y = curr_x, curr_y
                        print("Move Mode")

                    elif fingers == [0, 1, 0, 0, 0]:
                        pyautogui.click()
                        time.sleep(0.2)
                        print("Single Click")

                    elif fingers == [0, 1, 1, 0, 0]:
                        pyautogui.doubleClick()
                        time.sleep(0.3)
                        print("Double Click")

                    elif sum(fingers) >= 4:
                        print("Idle (Open Hand)")

            # FPS
            c_time = time.time()
            fps = 1 / (c_time - p_time + 1e-5)
            p_time = c_time
            cv2.putText(img, f'FPS: {int(fps)}', (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Show feed
            cv2.imshow("Gesture Mouse Control", img)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    def run(self):
        """ Main loop to process webcam feed and detect gestures """
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            return
    
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)
        
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    landmarks = hand_landmarks.landmark
                    gesture = self.detect_gesture(landmarks)  # Always use detect_gesture
                    cv2.putText(frame, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  
                print("Gesture Mode Active (Media control disabled)")  # Still shows toggle info
            if key == ord('q'):
                break  # Exit program
        
            cv2.imshow('Gesture Control', frame)

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    gesture_control = GestureControl()
    gesture_control.run()
