import cv2
import mediapipe as mp
import numpy as np


def display_text(image, text, position):
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, 2)


cap = cv2.VideoCapture(0)


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
    model_selection=1
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
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=1)
            )
            pass


def blur_background(frame, landmarks):
    if landmarks is not None:
        mask = landmarks.segmentation_mask
        condition = mask > 0.5
        background = cv2.GaussianBlur(frame, (55, 55), 0)
        frame = np.where(condition[:, :, None], frame, background)
    return frame


def track_people(frame, landmarks):
    answer = frame
    if landmarks.pose_landmarks is not None:
        h, w, _ = frame.shape
        nose = landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        right_sh = landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_sh = landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        nose_x = int(nose.x * w)
        nose_y = int(nose.y * h)

        left_sh_x = int(left_sh.x * w)
        left_sh_y = int(left_sh.y * h)

        right_sh_x = int(right_sh.x * w)
        right_sh_y = int(right_sh.y * h)


        offset = 100
        top = max(0, int(nose_y - 3 * offset))
        bottom = min(h, int(max(left_sh_y, right_sh_y)))
        left = max(0, int(min(left_sh_x, right_sh_x)) - offset)
        right = min(w, int(max(left_sh_x, right_sh_x)) + offset)

        h_len = bottom - top
        x_len2 = int(h_len*1280/1440)
        x_cen = int((left+right)/2)
        left = max(x_cen - x_len2, 0)
        right = min(x_cen + x_len2, w)



        answer = frame.copy()[top:bottom, left:right]
        answer = cv2.resize(answer, (1280, 720))

    return answer

def count_fingers(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    count = 0
    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            count += 1
    return count

def face_blur(frame, landmarks):
    if landmarks is not None:
        h, w, _ = frame.shape
        mask = landmarks.segmentation_mask
        condition = mask < 0.5
        background = np.zeros_like(frame)
        # background = cv2.GaussianBlur(frame, (55, 55), 0)
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



face_tracking_enabled = False
background_blur_enabled = False
face_detection_enabled = False
people_tracking_enabled = False
face_blur_enabled = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = np.fliplr(frame)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(image_rgb)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            fingers_count = count_fingers(hand_landmarks)
            if fingers_count == 1:
                face_blur_enabled = not face_blur_enabled
            elif fingers_count == 2:
                background_blur_enabled = not background_blur_enabled
            elif fingers_count == 3:
                people_tracking_enabled = not people_tracking_enabled
            elif fingers_count == 4:
                face_detection_enabled = not face_detection_enabled

    key = cv2.waitKey(0)
    if key == ord('1'):
        people_tracking_enabled = not face_tracking_enabled
    elif key == ord('2'):
        background_blur_enabled = not background_blur_enabled
    elif key == ord('3'):
        face_blur_enabled = not face_blur_enabled
    elif key == ord('4'):
        face_detection_enabled = not face_detection_enabled
    if key == 13 or key == ord('q') or not ret:
        break

    display_text(frame, f"Face Blur: {face_blur_enabled}", (10, 30))
    display_text(frame, f"BG Blur: {background_blur_enabled}", (10, 60))
    display_text(frame, f"Tracking: {people_tracking_enabled}", (10, 90))
    display_text(frame, f"Face Detect: {face_detection_enabled}", (10, 120))

    if face_blur_enabled:
        frame = face_blur(frame, segmentation.process(image_rgb))

    if background_blur_enabled:
        frame = blur_background(frame, segmentation.process(image_rgb))

    if people_tracking_enabled:
        frame = track_people(frame, pose.process(image_rgb))

    if face_detection_enabled:
        detect_faces(frame, face_mesh.process(image_rgb))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imshow('video', frame)


cap.release()
cv2.destroyAllWindows()