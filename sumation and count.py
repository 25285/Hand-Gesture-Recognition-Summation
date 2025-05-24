import cv2
import mediapipe as mp
import time
import statistics  # For smoothing the finger count

# --- MediaPipe Initialization ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,  # Capable of detecting two hands for 0-10 range
    model_complexity=1, # 0 or 1. 1 is more accurate, 0 is faster. Default is 1.
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# --- Summation Variables & State ---
# Store smoothed finger count and last seen time for each hand (Left, Right)
# Also track stable value and its start time for first number capture
hand_tracking_data = {
    "Left": {"smoothed": 0, "buffer": [], "last_seen_time": time.time()},
    "Right": {"smoothed": 0, "buffer": [], "last_seen_time": time.time()}
}

first_number = None
second_number = None
sum_result = None

# Variables for the first number stabilization
first_number_candidate = None # The potential value for first_number currently being held
first_number_candidate_stable_start_time = None # Time when first_number_candidate became stable

# Time delays
FIRST_NUMBER_STABILIZATION_DELAY = 1.0 # REQUIRED: 1 second for first number capture
SECOND_NUMBER_CAPTURE_DELAY = 2.0      # REQUIRED: 2-second delay between first and second number capture

# Reset timeouts
last_any_hand_time = time.time() # Tracks when any hand was last seen (for global reset)
RESET_TIMEOUT_GLOBAL = 3 # Reset after 3 seconds of no hands in frame
RESET_TIMEOUT_NO_NEW_INPUT = 5 # Reset if first number captured, but no distinct second number input after this time

# Buffer for finger counts smoothing
BUFFER_SIZE = 7 # Increased buffer size for smoother counts (more stable)

# --- Webcam Setup ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# --- Display Configuration ---
FONT_PRIMARY = cv2.FONT_HERSHEY_SIMPLEX
FONT_SECONDARY = cv2.FONT_HERSHEY_PLAIN

# For First, Second, Sum text
COLOR_NUM_INPUT = (0, 200, 0)  # Bright Green
COLOR_SUM_RESULT = (0, 0, 255) # Bright Red
COLOR_WAIT_MSG = (203, 192, 255) # Light Pink/Purple
COLOR_INFO_TEXT = (255, 150, 0) # Orange/Blue for general info
COLOR_RESET_TEXT = (0, 0, 200) # Darker Red for reset messages

SCALE_MAIN_INFO = 1.0
THICKNESS_MAIN_INFO = 2
SCALE_SUM_INFO = 1.2
THICKNESS_SUM_INFO = 3
SCALE_SUB_INFO = 0.8
THICKNESS_SUB_INFO = 1

# For status text like Handedness, Finger count near hand
COLOR_STATUS_PRIMARY = (255, 0, 0) # Blue
COLOR_STATUS_SECONDARY = (0, 255, 255) # Cyan
SCALE_STATUS = 0.7
THICKNESS_STATUS = 2

def get_finger_count(hand_landmarks, handedness_label):
    landmarks = hand_landmarks.landmark
    fingers = []
    
    # --- Thumb Detection ---
    # More robust thumb detection: check if THUMB_TIP is significantly outside the x-range of THUMB_MCP (base of thumb)
    # This accounts for various hand orientations better.
    if handedness_label == "Right":
        # For a right hand, thumb is open if tip is further left than MCP (smaller x-value)
        thumb_is_open = landmarks[mp_hands.HandLandmark.THUMB_TIP].x < landmarks[mp_hands.HandLandmark.THUMB_MCP].x - 0.03 # Add a small threshold
    else:  # Left hand
        # For a left hand, thumb is open if tip is further right than MCP (larger x-value)
        thumb_is_open = landmarks[mp_hands.HandLandmark.THUMB_TIP].x > landmarks[mp_hands.HandLandmark.THUMB_MCP].x + 0.03 # Add a small threshold
    fingers.append(1 if thumb_is_open else 0)
    
    # --- Other four fingers (Index, Middle, Ring, Pinky) ---
    # Check if tip is significantly above the PIP joint (for vertical extension)
    finger_tip_ids = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP
    ]
    finger_pip_ids = [
        mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
        mp_hands.HandLandmark.RING_FINGER_PIP, mp_hands.HandLandmark.PINKY_PIP
    ]
    
    for tip_id, pip_id in zip(finger_tip_ids, finger_pip_ids):
        # Finger is extended if the tip's y-coordinate is significantly above the PIP joint's y-coordinate
        extended = landmarks[tip_id].y < landmarks[pip_id].y - 0.02 # Add a small threshold
        fingers.append(1 if extended else 0)
    
    return sum(fingers)

def reset_calculation_state():
    """Resets all calculation-related variables to their initial state."""
    global first_number, second_number, sum_result, first_number_capture_time, \
           first_number_candidate, first_number_candidate_stable_start_time
    
    first_number = None
    second_number = None
    sum_result = None
    first_number_capture_time = None
    
    first_number_candidate = None
    first_number_candidate_stable_start_time = None
    
    for hand_label in hand_tracking_data:
        hand_tracking_data[hand_label]["smoothed"] = 0
        hand_tracking_data[hand_label]["buffer"].clear() # Clear buffer for fresh start
        hand_tracking_data[hand_label]["last_seen_time"] = time.time() # Reset seen time


print("Starting gesture recognition. Press 'q' to quit.")
print(f"Hold your first number (1-10) for {FIRST_NUMBER_STABILIZATION_DELAY:.1f}s to set the first value.")
print(f"Then, change to the second number (1-10). A {SECOND_NUMBER_CAPTURE_DELAY:.1f}s delay is required between the first and second number captures.")
print("Showing 0 fingers on any active hand or removing all hands will reset the calculation.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Frame not captured. Skipping.")
        continue

    frame = cv2.flip(frame, 1) # Flip horizontally for natural selfie-view
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    current_time = time.time()
    detected_hands_this_frame = {"Left": False, "Right": False}
    
    # Process each detected hand
    if results.multi_hand_landmarks and results.multi_handedness:
        last_any_hand_time = current_time # Update global last detection time

        for hand_landmarks, handedness_entry in zip(results.multi_hand_landmarks, results.multi_handedness):
            handedness_label = handedness_entry.classification[0].label # "Left" or "Right"
            detected_hands_this_frame[handedness_label] = True
            hand_tracking_data[handedness_label]["last_seen_time"] = current_time

            # Draw landmarks
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_draw.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
            )
            
            # Get and buffer finger count (0-5 per hand)
            raw_count = get_finger_count(hand_landmarks, handedness_label)
            hand_tracking_data[handedness_label]["buffer"].append(raw_count)
            if len(hand_tracking_data[handedness_label]["buffer"]) > BUFFER_SIZE:
                hand_tracking_data[handedness_label]["buffer"].pop(0)
            
            # Smooth the count
            if len(hand_tracking_data[handedness_label]["buffer"]) == BUFFER_SIZE:
                smoothed_count = int(statistics.median(hand_tracking_data[handedness_label]["buffer"]))
            else:
                smoothed_count = raw_count # Use raw count if buffer not full yet

            hand_tracking_data[handedness_label]["smoothed"] = smoothed_count

            # Display finger count (0-5) and handedness near the hand
            try:
                # Use a point near the wrist for stable text positioning
                wrist_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                x_coord = int(wrist_landmark.x * frame.shape[1])
                y_coord = int(wrist_landmark.y * frame.shape[0])
                
                # Position text above the hand, adjusting for handedness to avoid overlap
                text_x = x_coord - 70 if handedness_label == "Right" else x_coord + 10
                text_y = max(30, y_coord - 50) # Ensure text is not off screen top
                
                cv2.putText(frame, f"{handedness_label}: {smoothed_count}", (text_x, text_y),
                            FONT_PRIMARY, SCALE_STATUS, COLOR_STATUS_PRIMARY, THICKNESS_STATUS, cv2.LINE_AA)
            except Exception:
                pass # Should not happen if hand is detected
    else: # No hands detected in this frame
        # Clear buffers for all hands as they are no longer detected
        for hand_label in hand_tracking_data:
            hand_tracking_data[hand_label]["smoothed"] = 0
            hand_tracking_data[hand_label]["buffer"].clear()
            # Don't reset last_seen_time here, let it age for timeout detection

    # --- Calculate combined_finger_count (0-10) based on currently detected hands ---
    combined_finger_count = 0
    num_currently_active_hands = 0
    if detected_hands_this_frame["Left"]:
        combined_finger_count += hand_tracking_data["Left"]["smoothed"]
        num_currently_active_hands += 1
    if detected_hands_this_frame["Right"]:
        combined_finger_count += hand_tracking_data["Right"]["smoothed"]
        num_currently_active_hands += 1

    # --- Reset Logic ---
    reset_reason = None

    # 1. Global Reset by No Hand Timeout (no hands detected for too long)
    if not any(detected_hands_this_frame.values()) and (current_time - last_any_hand_time > RESET_TIMEOUT_GLOBAL):
        if first_number is not None or sum_result is not None or first_number_candidate is not None:
            reset_reason = "Timeout: No hands detected."
    
    # 2. Reset by Zero Fingers in combined count (if at least one hand is present and showing zero)
    if not reset_reason and combined_finger_count == 0 and num_currently_active_hands > 0:
        if first_number is not None or sum_result is not None or first_number_candidate is not None:
            reset_reason = "Zero shown (clearing calculation)."
    
    # 3. Reset if first number is set, but no new input (second number) is provided for too long
    if not reset_reason and first_number is not None and second_number is None:
        if (current_time - first_number_capture_time > RESET_TIMEOUT_NO_NEW_INPUT):
            reset_reason = "Timeout: No second number input."

    if reset_reason:
        print(f"Resetting ({reset_reason}).")
        reset_calculation_state()

    # --- Summation Logic (only if not reset) ---
    if not reset_reason:
        
        # Phase 1: Capture First Number (if not already set)
        if first_number is None:
            if combined_finger_count > 0: # A number 1-10 is being shown
                # Check if this combined count is stable for the required duration (FIRST_NUMBER_STABILIZATION_DELAY)
                if first_number_candidate != combined_finger_count:
                    # New candidate number, reset its stable timer
                    first_number_candidate = combined_finger_count
                    first_number_candidate_stable_start_time = current_time
                elif first_number_candidate_stable_start_time is not None and \
                     (current_time - first_number_candidate_stable_start_time >= FIRST_NUMBER_STABILIZATION_DELAY):
                    # Candidate has been stable for long enough, capture it as first_number
                    first_number = combined_finger_count
                    first_number_capture_time = current_time
                    first_number_candidate = None # Clear candidate after capture
                    first_number_candidate_stable_start_time = None
                    print(f"First number captured: {first_number}")

        # Phase 2: Capture Second Number (if first is set and sum not yet calculated)
        elif second_number is None:
            # Only consider capturing second number after the delay from first_number_capture_time
            if (current_time - first_number_capture_time >= SECOND_NUMBER_CAPTURE_DELAY):
                # Ensure a new, non-zero number (1-10) is being shown.
                # The restriction `combined_finger_count != first_number` has been removed.
                if combined_finger_count > 0:
                    second_number = combined_finger_count
                    sum_result = first_number + second_number
                    print(f"Second number captured: {second_number}. Sum: {sum_result}.")


    # --- Display Information on Frame ---
    y_offset_info = 30 # Starting Y for dynamic text

    # Display current combined count (0-10) near the top center for debugging/feedback
    # Only display if hands are actively showing something
    if num_currently_active_hands > 0:
        cv2.putText(frame, f"Total Fingers: {combined_finger_count}", (frame.shape[1] // 2 - 80, 30),
                    FONT_PRIMARY, SCALE_MAIN_INFO, COLOR_STATUS_SECONDARY, THICKNESS_MAIN_INFO, cv2.LINE_AA)


    if sum_result is not None:
        cv2.putText(frame, f"Sum: {first_number} + {second_number} = {sum_result}", (10, y_offset_info),
                    FONT_PRIMARY, SCALE_SUM_INFO, COLOR_SUM_RESULT, THICKNESS_SUM_INFO, cv2.LINE_AA)
        y_offset_info += 50
        cv2.putText(frame, "Show 0 fingers or remove hands to reset.", (10, y_offset_info),
                    FONT_PRIMARY, SCALE_SUB_INFO, COLOR_INFO_TEXT, THICKNESS_SUB_INFO, cv2.LINE_AA)
        y_offset_info += 30
    elif first_number is not None:
        cv2.putText(frame, f"First: {first_number}", (10, y_offset_info),
                    FONT_PRIMARY, SCALE_MAIN_INFO, COLOR_NUM_INPUT, THICKNESS_MAIN_INFO, cv2.LINE_AA)
        y_offset_info += 40

        # Display waiting message for second number
        elapsed_time_since_first_capture = current_time - first_number_capture_time
        
        if elapsed_time_since_first_capture < SECOND_NUMBER_CAPTURE_DELAY:
            time_left_for_delay = SECOND_NUMBER_CAPTURE_DELAY - elapsed_time_since_first_capture
            cv2.putText(frame, f"(Wait {time_left_for_delay:.1f}s for new number)", (10, y_offset_info),
                        FONT_PRIMARY, SCALE_SUB_INFO, COLOR_WAIT_MSG, THICKNESS_SUB_INFO, cv2.LINE_AA)
            y_offset_info += 30
        else:
            cv2.putText(frame, "(Show 2nd num: 1-10)", (10, y_offset_info),
                        FONT_PRIMARY, SCALE_SUB_INFO, COLOR_INFO_TEXT, THICKNESS_SUB_INFO, cv2.LINE_AA)
            y_offset_info += 30
        
        # Warn if no new input for too long
        if elapsed_time_since_first_capture > (RESET_TIMEOUT_NO_NEW_INPUT / 2) and second_number is None:
            cv2.putText(frame, f"Reset if no new input in {RESET_TIMEOUT_NO_NEW_INPUT - elapsed_time_since_first_capture:.1f}s", (10, y_offset_info),
                        FONT_PRIMARY, SCALE_SUB_INFO, COLOR_RESET_TEXT, THICKNESS_SUB_INFO, cv2.LINE_AA)
            y_offset_info += 30

    else: # Initial state, waiting for first number
        cv2.putText(frame, "Show 1-10 fingers to start sum", (10, y_offset_info),
                    FONT_PRIMARY, SCALE_MAIN_INFO, COLOR_INFO_TEXT, THICKNESS_MAIN_INFO, cv2.LINE_AA)
        y_offset_info += 40

        if first_number_candidate is not None and first_number_candidate > 0:
            # Display countdown for first number stabilization
            stability_time_elapsed = current_time - first_number_candidate_stable_start_time
            if stability_time_elapsed < FIRST_NUMBER_STABILIZATION_DELAY:
                time_to_stabilize = FIRST_NUMBER_STABILIZATION_DELAY - stability_time_elapsed
                cv2.putText(frame, f"Holding {first_number_candidate} for {time_to_stabilize:.1f}s...", (10, y_offset_info),
                            FONT_PRIMARY, SCALE_SUB_INFO, COLOR_WAIT_MSG, THICKNESS_SUB_INFO, cv2.LINE_AA)
                y_offset_info += 30

        if not any(detected_hands_this_frame.values()): # No hands detected
            time_since_last_global = current_time - last_any_hand_time
            if time_since_last_global > (RESET_TIMEOUT_GLOBAL / 2): # Warn if half the delay passed
                cv2.putText(frame, f"No hands detected. Reset in {RESET_TIMEOUT_GLOBAL - time_since_last_global:.1f}s", (10, y_offset_info),
                            FONT_PRIMARY, SCALE_SUB_INFO, COLOR_RESET_TEXT, THICKNESS_SUB_INFO, cv2.LINE_AA)
                y_offset_info += 30


    cv2.imshow("Hand Gesture Summation (0-10 Flexible Input)", frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        print("Exiting program.")
        break

if hands:
    hands.close()
if cap:
    cap.release()
cv2.destroyAllWindows()
