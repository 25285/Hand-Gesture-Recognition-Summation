import cv2
import mediapipe as mp
import time

# --- MediaPipe Initialization ---
# Initialize MediaPipe Hands model
# model_complexity=0 is faster but less accurate. You can try 1 or 2 for more accuracy if needed.
# min_detection_confidence: Minimum confidence value for hand detection to be considered successful.
# min_tracking_confidence: Minimum confidence value for hand tracking to be considered successful.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils # Utility for drawing landmarks

# --- Finger Tip Landmark IDs ---
# These are the specific landmark IDs for the tips of each finger from MediaPipe's hand model.
TIP_IDS = [mp_hands.HandLandmark.THUMB_TIP,
           mp_hands.HandLandmark.INDEX_FINGER_TIP,
           mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
           mp_hands.HandLandmark.RING_FINGER_TIP,
           mp_hands.HandLandmark.PINKY_TIP]

# --- Summation Variables ---
# Variables to store the numbers for addition.
first_number = None
second_number = None
sum_result = None

# Time tracking for resetting the sum if no hand is detected for a period.
last_hand_present_time = time.time()
reset_delay_sec = 2 # If no hand for this duration, reset the sum calculation.

# --- Webcam Setup ---
cap = cv2.VideoCapture(0) # 0 for the default webcam. Change if you have multiple cameras.

# Check if the webcam opened successfully.
if not cap.isOpened():
    print("Error: Could not open webcam. Please check if it's connected and not in use.")
    exit()

# --- Main Video Processing Loop ---
while cap.isOpened():
    success, image = cap.read() # Read a frame from the webcam.
    if not success:
        print("Ignoring empty camera frame. (Might be end of video stream or camera error)")
        continue

    # Flip the image horizontally for a natural, mirror-like view.
    image = cv2.flip(image, 1)
    # Convert the BGR image (OpenCV default) to RGB (MediaPipe requires RGB).
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image with MediaPipe Hands to find hand landmarks.
    results = hands.process(image_rgb)

    current_finger_count = 0
    hands_detected_in_frame = False # Flag to track if any hand is currently visible.

    # Check if any hands were detected in the current frame.
    if results.multi_hand_landmarks:
        hands_detected_in_frame = True
        last_hand_present_time = time.time() # Update the timestamp when a hand was last seen.

        # Iterate through each detected hand.
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks and connections on the image.
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the list of all 21 landmarks for the current hand.
            landmarks = hand_landmarks.landmark

            # --- Finger Counting Logic ---
            # A list to store 1 if finger is up, 0 if down.
            fingers_up = []

            # Thumb (Special Case):
            # The thumb's "up" position is often determined by its x-coordinate relative to its base or palm.
            # This is a simplified heuristic. For more robustness, consider hand type (left/right).
            # Here, we check if the thumb tip's x-coordinate is further from the MCP joint (base of thumb)
            # than the IP joint (middle joint of thumb). This works for both hands if flipped.
            if landmarks[TIP_IDS[0]].x < landmarks[mp_hands.HandLandmark.THUMB_IP].x and \
               landmarks[TIP_IDS[0]].x < landmarks[mp_hands.HandLandmark.THUMB_MCP].x: # For right hand (thumb points left)
                fingers_up.append(1)
            elif landmarks[TIP_IDS[0]].x > landmarks[mp_hands.HandLandmark.THUMB_IP].x and \
                 landmarks[TIP_IDS[0]].x > landmarks[mp_hands.HandLandmark.THUMB_MCP].x: # For left hand (thumb points right)
                fingers_up.append(1)
            else:
                fingers_up.append(0) # Thumb is likely curled

            # Other 4 Fingers (Index, Middle, Ring, Pinky):
            # For these fingers, we compare the y-coordinate of the fingertip with the y-coordinate
            # of the PIP (Proximal InterPhalangeal) joint (the second knuckle from the palm).
            # If the tip's y-coordinate is *smaller* than the PIP's y-coordinate (higher on the screen),
            # the finger is considered extended.
            for i in range(1, 5): # Iterate from Index Finger (ID 1) to Pinky Finger (ID 4)
                # TIP_IDS[i] is the fingertip landmark ID.
                # TIP_IDS[i]-2 gives the PIP joint landmark ID for that finger.
                if landmarks[TIP_IDS[i]].y < landmarks[TIP_IDS[i]-2].y:
                    fingers_up.append(1) # Finger is up
                else:
                    fingers_up.append(0) # Finger is down

            # Sum the number of extended fingers.
            current_finger_count = sum(fingers_up)

            # Display the current finger count near the hand.
            # Get coordinates for text placement (using index finger tip as a reference).
            x_text = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image.shape[1]) - 50
            y_text = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image.shape[0]) - 50
            cv2.putText(image, f"Count: {current_finger_count}", (x_text, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # --- Summation Logic ---
    # If no hands are detected, check if it's time to reset the sum.
    if not hands_detected_in_frame:
        if time.time() - last_hand_present_time > reset_delay_sec:
            if first_number is not None or second_number is not None:
                print("Resetting sum due to no hand detected for a while.")
            first_number = None
            second_number = None
            sum_result = None
    # If hands are detected AND a valid finger count is obtained (not zero).
    elif hands_detected_in_frame and current_finger_count > 0:
        # Capture the first number.
        if first_number is None:
            first_number = current_finger_count
            sum_result = None # Clear any previous result if starting a new calculation.
            cv2.putText(image, f"First Num: {first_number}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # Capture the second number if the first is set, second is not, and current count is different.
        # This prevents continuously capturing the same number.
        elif first_number is not None and second_number is None and current_finger_count != first_number:
            second_number = current_finger_count
            sum_result = first_number + second_number
            cv2.putText(image, f"First Num: {first_number}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f"Second Num: {second_number}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f"Sum: {sum_result}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
        # If both numbers are already set, just keep displaying the sum.
        elif first_number is not None and second_number is not None:
             cv2.putText(image, f"First Num: {first_number}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
             cv2.putText(image, f"Second Num: {second_number}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
             cv2.putText(image, f"Sum: {sum_result}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)

    # Display the processed frame.
    cv2.imshow('Hand Gesture Recognizer & Sum', image)

    # Break the loop if the 'q' key is pressed.
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# --- Release Resources ---
# Close the MediaPipe Hands model.
hands.close()
# Release the webcam.
cap.release()
# Close all OpenCV windows.
cv2.destroyAllWindows()
