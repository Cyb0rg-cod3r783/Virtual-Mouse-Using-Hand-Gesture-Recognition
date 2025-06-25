# OpenCV (Open Source Computer Vision Library) is a free, open-source library that provides a wide range of tools and functions for real-time computer vision and image processing

import cv2
import mediapipe as mp
import time
import pyautogui
from math import hypot # hypot refers to a function used to calculate the Euclidean norm, which is essentially the distance from the origin (0, 0) to a point (x, y) in a 2D plane.

dragging = False
last_click_time = 0
click_count = 0
prev_scroll_y = 0
scroll_mode = False

def fingers_up(lm_list):
                finger_tips = [8, 12, 16, 20]
                finger_status = []

                for tip_id in finger_tips:
                    if lm_list[tip_id].y < lm_list[tip_id -2].y :
                        finger_status.append(1) # Finger is up 

                    else:
                        finger_status.append(0) # Finger is down

                return finger_status 

cap = cv2.VideoCapture(0) # this opens webcam (0 = default camera)

# Initialize MediaPipe Hand Model
mp_hands = mp.solutions.hands # 	Loads the hand tracking model
hands = mp_hands.Hands(max_num_hands = 1) # we will start with one hand
mp_draw = mp.solutions.drawing_utils # for drawing landmarks.

screen_width, screen_height = pyautogui.size() # pyautogui returns a tuple with three values : (height, width, channels)

prev_x, prev_y = 0, 0
smoothing = 5 # you can experiment with 5 - 10


while True:
    success, frame = cap.read() # this is unpacking of the tuple. cap.read() returns a tuple which contains two values, a boolean and a frame. the boolean value is assigned to success and the frame is assigned to the frame variable. Reads one frame (returns success flag + image)

    frame = cv2.flip(frame, 1) # Mirror the image like a real mirror

    frame_height, frame_width, _ = frame.shape

    # convert image to rgb
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb) # Processes the frame and detects hand landmarks

    # check if any hands are detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks: # Contains landmark data for each detected hand

            lm_list = hand_landmarks.landmark # Index fingertip position

            # Get the thumb and finger tip co-ordinates (landmark 8)

            x1 = int(lm_list[4].x * screen_width) # Converts from 0–1 to screen coordinates
            y1 = int(lm_list[4].y * screen_height)

            x2 = int(lm_list[8].x * screen_width)
            y2 = int(lm_list[8].y * screen_height)

            x3 = int(lm_list[12].x * screen_width)
            y3 = int(lm_list[12].y * screen_height)

        # Move the cursor using the index finger as before.
            # Convert to screen coordinates 
            target_x = x2
            target_y = y2
            # target_x, target_y	Where the finger wants to move
            # smoothing 
            curr_x = prev_x + (target_x - prev_x) / smoothing
            curr_y = prev_y + (target_y - prev_y) / smoothing
            # curr_x, curr_y	Where the cursor actually moves (interpolated)

            # Move mouse 
            pyautogui.moveTo(curr_x, curr_y) # Moves the mouse instantly to the position.

            prev_x, prev_y = curr_x, curr_y # this stores the last mouse position 

            # calculate the distance between the thumb and the fingers.

            distance_thumb_index = hypot(x2 - x1, y2 - y1)
            distance_index_middle = hypot(x2 - x3, y2 - y3)
            distance_thumb_middle = hypot(x1 - x3 , y1 - y3)

            # if distance_thumb_index < 60:
            #     now = time.time()

            #     # DOUBLE CLICK detection
            #     if now - last_click_time < 0.4:
            #         click_count += 1
            #         if click_count == 2:
            #             pyautogui.doubleClick()
            #             click_count = 0
            #     else:
            #         click_count = 1
            #         last_click_time = now

            #     # DRAG (if held for long)
            #     if not dragging and (now - last_click_time) > 0.5:
            #         pyautogui.mouseDown()
            #         dragging = True
            # else:
            #     click_count = 0
            #     last_click_time = 0
            #     if dragging:
            #         pyautogui.mouseUp()
            #         dragging = False    

            # If fingers are close enough , trigger click
            if distance_thumb_index < 60 :# you can experiment wiht 35 - 45 
                pyautogui.click() 
                pyautogui.sleep(0.2) # Prevent mulitple clicks instantly.

            if distance_thumb_middle < 60:
                pyautogui.doubleClick()
                pyautogui.sleep(0.2)
 
            if distance_thumb_index < 40 and distance_index_middle < 40:
                pyautogui.rightClick()
                time.sleep(0.2)

            
            
            fingers = fingers_up(lm_list)

            if fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0:
                scroll_mode = True

                # get current Y of index finger 
                current_scroll_y = lm_list[8].y

                if prev_scroll_y != 0:
                    diff = current_scroll_y - prev_scroll_y

                    if abs(diff) > 0.02: # Ignore tiny jitter
                        if diff > 0:
                            pyautogui.scroll(-100) # scroll down 
                        else:
                            pyautogui.scroll(100) # scroll up
                prev_scroll_y = current_scroll_y
            else:
                scroll_mode = False
                prev_scroll_y =0         

            # Draw visual feedback
            cv2.circle(frame, (int(lm_list[8].x * frame_width), int(lm_list[8].y * frame_height)), 10, (255, 0, 255), cv2.FILLED)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS) # Draws dots and lines on hand

    # show the result
    cv2.imshow("Smooth Cursor Control", frame) # Shows the current frame (displays an image in a window.). - frame is the image data you want to show—usually a NumPy array representing the current frame from your webcam. "Webcam Feed" is just the name of the window

    if cv2.waitKey(1) & 0xFF == ord('q'): # This waits 1 millisecond for a key press and allows the window to refresh. It checks “Was the key pressed equal to 'q'?”
    # If yes, you can break the loop and close the window:
        break

cap.release() # closes the connection to your webcam or video file. Releases the hardware resource.
cv2.destroyAllWindows() # this closes all the OpenCV windows

# Smoothing the movements
# We'll apply linear interpolation (LERP) between the old position and the new fingertip position to get gradual movement

# Why Smoothing Helps:
# Real hand motion is a bit shaky
# Mediapipe updates coordinates frame-by-frame — sudden jumps
# We smooth out motion using a “weighted average” of the previous and current positions

# ✅ Smoothing Formula:
# python
# Copy code
# smooth_x = prev_x + (curr_x - prev_x) / smoothing_factor
# Where:

# smoothing_factor (e.g., 5–10) controls the smoothness

# Higher = smoother but more lag

# Lower = quicker but jittery