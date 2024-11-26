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


face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


segmentation = mp_selfie_segmentation.SelfieSegmentation(
    model_selection=1
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


def blur_background(frame, pose_landmarks):
    if pose_landmarks is not None:
        results = segmentation.process(image_rgb)
        mask = results.segmentation_mask
        condition = mask > 0.5
        background = cv2.GaussianBlur(frame, (55, 55), 0)
        frame = np.where(condition[:, :, None], frame, background)
    return frame

face_tracking_enabled = False
background_blur_enabled = False
face_detection_enabled = False

while cap.isOpened():
    ret, frame = cap.read()

    if cv2.waitKey(1) & 0xFF == 13 or cv2.waitKey(1) & 0xFF == ord('q') or not ret:
        break

    # if face_tracking_enabled:
    #     frame = track_face(frame, face_mesh.process(image_rgb))

    if background_blur_enabled:
        frame = blur_background(frame, segmentation.process(image_rgb))

    if face_detection_enabled:
        detect_faces(frame, face_mesh.process(image_rgb))

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = np.fliplr(frame)
    cv2.imshow('TEST', frame)

    key = cv2.waitKey(1)
    if key == ord('1'):
        face_tracking_enabled = not face_tracking_enabled
    elif key == ord('2'):
        background_blur_enabled = not background_blur_enabled
    elif key == ord('3'):
        face_detection_enabled = not face_detection_enabled

cap.release()
cv2.destroyAllWindows()