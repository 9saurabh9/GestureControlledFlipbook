import mediapipe as mp
import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# OpenCV Video Capture
cap = cv2.VideoCapture(0)

prev_centroid = None  # Variable to track the previous position of the hand
hand_detected = False  # Flag to track whether a hand is detected

driver = webdriver.Chrome()  # Ensure chromedriver is in PATH or specify its location
driver.get("https://flip.manager.click")  # Open the desired URL
action = ActionChains(driver)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        # Hand is detected
        hand_detected = True

        for landmarks in results.multi_hand_landmarks:
            # Calculate centroid of the hand
            hand_landmarks = np.array([[lm.x, lm.y] for lm in landmarks.landmark])
            centroid = np.mean(hand_landmarks, axis=0)

            if prev_centroid is not None:
                # Compare previous centroid with current centroid
                x_diff = centroid[0] - prev_centroid[0]

                if x_diff < -0.1:  # Swipe Right Threshold
                    print("Swipe Right Detected")
                    try:
                        next_page_div = driver.find_element(By.XPATH, "//div[@title='Next Page']")
                        action.move_to_element(next_page_div).click().perform()
                        print("Clicked 'Next Page'")
                    except Exception as e:
                        print(f"Error: {e}")
                elif x_diff > 0.1:  # Swipe Left Threshold
                    print("Swipe Left Detected")
                    try:
                        prev_page_div = driver.find_element(By.XPATH, "//div[@title='Previous Page']")
                        action.move_to_element(prev_page_div).click().perform()
                        print("Clicked 'Previous Page'")
                    except Exception as e:
                        print(f"Error: {e}")

            # Update previous centroid
            prev_centroid = centroid

    else:
        # No hand detected; reset variables
        if hand_detected:
            print("Hand left the frame. Resetting variables.")
        prev_centroid = None
        hand_detected = False

    # Display the frame with hand landmarks
    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

driver.quit()
