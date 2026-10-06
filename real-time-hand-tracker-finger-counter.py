import os
import urllib.request
import time
import math
import cv2
import mediapipe as mp

# 1. Path setup & auto-download model
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "hand_landmarker.task")

if not os.path.exists(model_path):
    print("Downloading hand_landmarker.task model file...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("Download complete!")

# 2. Setup MediaPipe HandLandmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

class HandDetectorState:
    result = None

def callback(result, output_image, timestamp_ms):
    HandDetectorState.result = result

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=callback
)

# Skeleton connection pairs
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # Index
    (5, 9), (9, 10), (10, 11), (11, 12),      # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),    # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),   # Pinky
    (0, 17)                                   # Base
]

def get_distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        frame_timestamp_ms = int(time.time() * 1000)
        landmarker.detect_async(mp_image, frame_timestamp_ms)

        h, w, _ = frame.shape
        total_fingers = 0

        if HandDetectorState.result and HandDetectorState.result.hand_landmarks:
            for idx, lm in enumerate(HandDetectorState.result.hand_landmarks):
                # 1. Determine hand label (inverted to account for mirror/flipped view)
                hand_label = "Hand"
                if HandDetectorState.result.handedness and idx < len(HandDetectorState.result.handedness):
                    raw_label = HandDetectorState.result.handedness[idx][0].category_name
                    hand_label = "Left" if raw_label == "Right" else "Right"

                # 2. Draw skeleton
                points = [(int(pt.x * w), int(pt.y * h)) for pt in lm]
                for start_idx, end_idx in HAND_CONNECTIONS:
                    cv2.line(frame, points[start_idx], points[end_idx], (230, 230, 230), 2)
                for pt in points:
                    cv2.circle(frame, pt, 5, (0, 255, 0), -1)

                # 3. Check individual finger status (True = open/up, False = closed)
                # 4 Fingers: check if tip is higher than PIP knuckle
                index_open = lm[8].y < lm[6].y
                middle_open = lm[12].y < lm[10].y
                ring_open = lm[16].y < lm[14].y
                pinky_open = lm[20].y < lm[18].y

                # Thumb Fix: compare distance to pinky knuckle (lm 17) against thumb-base distance
                # When curled across the palm in a fist, thumb tip moves close to the pinky knuckle
                thumb_dist_to_pinky_base = get_distance(lm[4], lm[17])
                thumb_joint_dist = get_distance(lm[2], lm[17])
                thumb_open = thumb_dist_to_pinky_base > thumb_joint_dist * 1.25

                # Tally fingers for this hand
                fingers_status = [thumb_open, index_open, middle_open, ring_open, pinky_open]
                hand_fingers = sum(fingers_status)
                total_fingers += hand_fingers

                # 4. Gesture Detection Logic
                gesture = ""
                # Closed Fist: 0 fingers up
                if hand_fingers == 0:
                    gesture = "Fist"
                # Peace Sign: Index & Middle open, Ring & Pinky closed
                elif index_open and middle_open and not ring_open and not pinky_open:
                    gesture = "Peace Sign"
                # Thumbs Up: Thumb open, all other 4 fingers closed, thumb pointing up
                elif thumb_open and not any([index_open, middle_open, ring_open, pinky_open]) and lm[4].y < lm[3].y:
                    gesture = "Thumbs Up"
                # Thumbs Down: Thumb pointing down relative to its joint, other 4 closed
                elif not any([index_open, middle_open, ring_open, pinky_open]) and lm[4].y > lm[3].y:
                    gesture = "Thumbs Down"

                # 5. Display annotations
                wrist_pt = points[0]
                text = f"{hand_label}: {hand_fingers}"
                if gesture:
                    text += f" | {gesture}"

                cv2.putText(frame, text, (max(10, wrist_pt[0] - 60), max(30, wrist_pt[1] + 35)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Global counter in top-left
        cv2.putText(frame, f"Total Count: {total_fingers}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow("Hand Tracker & Gestures", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()