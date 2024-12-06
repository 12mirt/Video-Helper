import cv2
import mediapipe as mp
import numpy as np
import time
from myapp.static.config import *



def display_text(image, text, position):
    overlay = image.copy()
    cv2.rectangle(overlay, (position[0] - 5, position[1] - 22), (position[0] + 500, position[1] + 7), (0, 0, 0), -1)
    image = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    return image


cap = cv2.VideoCapture(0)
while not cap.read()[0]:
    pass


mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands_detector = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_selfie_segmentation = mp.solutions.selfie_segmentation

hands = mp_hands_detector.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

segmentation = mp_selfie_segmentation.SelfieSegmentation(
    model_selection=0
)
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def detect_faces(frame, landmarks):
    if landmarks.multi_face_landmarks is not None:
        for face_landmarks in landmarks.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1,
                                                                             circle_radius=1),
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=1)
            )
    return frame


def blur_background(frame, landmarks):
    if landmarks is not None:
        mask = landmarks.segmentation_mask
        condition = mask > 0.5
        background = cv2.GaussianBlur(frame, (55, 55), 0)
        frame = np.where(condition[:, :, None], frame, background)
    return frame


def replace_background(frame, landmarks, background_image='static/example.png'):
    if landmarks is not None:
        mask = landmarks.segmentation_mask
        condition = mask > 0.5
        background = cv2.imread(background_image)
        background = cv2.resize(background, (frame.shape[1], frame.shape[0]))
        frame = np.where(condition[:, :, None], frame, background)
    return frame


def track_people(frame, landmarks):
    if landmarks.pose_landmarks is not None:
        h, w, _ = frame.shape
        nose = landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        right_sh = landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_sh = landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]

        nose_y = int(nose.y * h)
        left_sh_x = int(left_sh.x * w)
        left_sh_y = int(left_sh.y * h)
        right_sh_x = int(right_sh.x * w)
        right_sh_y = int(right_sh.y * h)

        margin = 100
        top = max(0, nose_y - 3 * margin)
        bottom = min(h, max(left_sh_y, right_sh_y))
        left = max(0, min(left_sh_x, right_sh_x) - margin)
        right = min(w, max(left_sh_x, right_sh_x) + margin)

        h_len = bottom - top
        x_len2 = int(h_len * 1280 / 1440)
        x_cen = int((left + right) / 2)
        left = max(x_cen - x_len2, 0)
        right = min(x_cen + x_len2, w)

        if left == 0:
            right = x_len2*2
        if right == w:
            left = w - x_len2*2
        if top == 0:
            bottom = h_len
        if bottom == h:
            top = h - h_len

        cropped_frame = frame[top:bottom, left:right]
        resized_frame = cv2.resize(cropped_frame, (480, 270))
        return resized_frame
    return frame



def count_fingers(hand_landmarks, result):
    tips = [8, 12, 16, 20]
    count = 0
    if result.multi_handedness[0].classification[0].label == 'Right':
        palm_x = hand_landmarks.landmark[0].x
        palm_y = hand_landmarks.landmark[0].y


        for tip in tips:
            tip_y = hand_landmarks.landmark[tip].y
            base_y = hand_landmarks.landmark[18].y
            if tip_y < base_y and result.multi_handedness[0].classification[0].label == 'Right':
                count += 1
        thumb_tip = hand_landmarks.landmark[4]
        thumb_ip = hand_landmarks.landmark[3]
        thumb_base = hand_landmarks.landmark[2]

        if palm_x < 0.5:
            if thumb_tip.x > thumb_base.x and abs(thumb_tip.y - thumb_ip.y) < 0.05:
                count += 1
        else:
            if thumb_tip.x < thumb_base.x and abs(thumb_tip.y - thumb_ip.y) < 0.05:
                count += 1
    return count


def face_blur(frame, landmarks):
    if landmarks is not None:
        h, w, _ = frame.shape
        mask = landmarks.segmentation_mask
        condition = mask < 0.5
        background = np.zeros_like(frame)
        if type_face_hide == 'blur':
            background = cv2.GaussianBlur(frame, (55, 55), 0)
        else:
            eyes = face_mesh.process(image_rgb)
            if eyes.multi_face_landmarks is not None:
                for face_landmarks in eyes.multi_face_landmarks:
                    for s, _ in mp_face_mesh.FACEMESH_LEFT_EYE:
                        landmark = face_landmarks.landmark[s]
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        cv2.circle(background, (x, y), 5, (255, 255, 255), -1)
                    for s, _ in mp_face_mesh.FACEMESH_RIGHT_EYE:
                        landmark = face_landmarks.landmark[s]
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        cv2.circle(background, (x, y), 5, (255, 255, 255), -1)

        frame = np.where(condition[:, :, None], frame, background)
    return frame


def enhance_appearance(frame):
    return cv2.bilateralFilter(frame, 9, 75, 75)


background_blur_enabled = False
face_detection_enabled = False
people_tracking_enabled = False
face_blur_enabled = False
enhance_enabled = False
use_hands = False
change_background = False
paused = False

gesture_start_time = None
current_gesture = None
gesture_hold_threshold = 3
fingers_count = 0
ret, frame = cap.read()

while ret:
    frame = cv2.resize(frame, resolution)
    frame = np.fliplr(frame)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    hand_results = hands.process(image_rgb)
    if hand_results.multi_hand_landmarks and use_hands:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            fingers_count = count_fingers(hand_landmarks, hand_results)
            if fingers_count != current_gesture:
                current_gesture = fingers_count
                gesture_start_time = time.time()
            elif fingers_count == 0:
                gesture_start_time = None
            elif time.time() - gesture_start_time > gesture_hold_threshold:
                if fingers_count == 1:
                    face_blur_enabled = not face_blur_enabled
                elif fingers_count == 2:
                    background_blur_enabled = not background_blur_enabled
                elif fingers_count == 3:
                    people_tracking_enabled = not people_tracking_enabled
                elif fingers_count == 4:
                    change_background = not change_background
                elif fingers_count == 5:
                    enhance_enabled = not enhance_enabled
                gesture_start_time = None
                current_gesture = None
    else:
        fingers_count = 0

    key = cv2.waitKey(1)
    if key == ord('1'):
        face_blur_enabled = not face_blur_enabled
    elif key == ord('2'):
        background_blur_enabled = not background_blur_enabled
    elif key == ord('3'):
        people_tracking_enabled = not people_tracking_enabled
    elif key == ord('4'):
        change_background = not change_background
    elif key == ord('5'):
        enhance_enabled = not enhance_enabled
    elif key == ord('6'):
        use_hands = not use_hands
    elif key == ord('9'):
        face_detection_enabled = not face_detection_enabled
    if key == ord('p'):
        paused = not paused
    if paused:
        continue
    if key == 13 or key == ord('q'):
        break

    if face_blur_enabled or background_blur_enabled or change_background:
        segmentation_result = segmentation.process(image_rgb)

    if face_blur_enabled:
        frame = face_blur(frame, segmentation_result)

    if change_background:
        frame = replace_background(frame, segmentation_result)

    if background_blur_enabled:
        frame = blur_background(frame, segmentation_result)

    if people_tracking_enabled:
        frame = track_people(frame, pose.process(image_rgb))

    if face_detection_enabled:
        frame = detect_faces(frame, face_mesh.process(image_rgb))

    if enhance_enabled:
        frame = enhance_appearance(frame)

    frame = cv2.resize(frame, (1280, 720))

    frame = display_text(frame, f"Keys:", (10, 30))
    frame = display_text(frame, f" '1' Face Blur: {face_blur_enabled}", (10, 60))
    frame = display_text(frame, f" '2' BG Blur: {background_blur_enabled}", (10, 90))
    frame = display_text(frame, f" '3' Tracking: {people_tracking_enabled}", (10, 120))
    frame = display_text(frame, f" '4' Change BG: {change_background}", (10, 150))
    frame = display_text(frame, f" '5' Enhance Appearance: {enhance_enabled}", (10, 180))
    frame = display_text(frame, f" '6' Hand control: {use_hands}", (10, 210))

    cv2.imshow('video', frame)
    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()
