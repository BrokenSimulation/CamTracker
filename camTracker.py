import cv2
import time
import mediapipe as mp
import math

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


hand_base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

hand_options = vision.HandLandmarkerOptions(
    base_options=hand_base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)


face_base_options = python.BaseOptions(
    model_asset_path="face_landmarker.task"
)

face_options = vision.FaceLandmarkerOptions(
    base_options=face_base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=True,
)

face_landmarker = vision.FaceLandmarker.create_from_options(face_options)


camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not access Mac camera.")
    raise SystemExit

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

created_shapes = []
hands_were_together = False

def distance(point1, point2):
    return math.sqrt(
        (point1[0] - point2[0]) ** 2 +
        (point1[1] - point2[1]) ** 2
    )

print("Camera started.")
print("Press ESC to quit.")

created_shapes = []
hands_were_together = False

previous_hand_x = None
swipe_distance = 0
swipe_direction = 0

while True:

    success, frame = camera.read()

    if not success:
        print("Could not read camera frame.")
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    timestamp_ms = time.monotonic_ns() // 1_000_000

    hand_result = hand_landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )

    hand_points = []

    for hand in hand_result.hand_landmarks:

        points = []

        for landmark in hand:

            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])

            points.append((x, y))

            cv2.circle(
                frame,
                (x, y),
                4,
                (0, 255, 0),
                -1
            )

        for start, end in HAND_CONNECTIONS:

            cv2.line(
                frame,
                points[start],
                points[end],
                (0, 255, 0),
                2
            )

        hand_points.append(points[8])

    if len(hand_points) >= 1:

        current_x = hand_points[0][0]

        if previous_hand_x is not None:

            movement = current_x - previous_hand_x

            if abs(movement) > 5:
                swipe_distance += abs(movement)

                if swipe_direction == 0:
                    swipe_direction = 1 if movement > 0 else -1

                elif (movement > 0 and swipe_direction < 0) or \
                     (movement < 0 and swipe_direction > 0):
                    swipe_distance = 0
                    swipe_direction = 1 if movement > 0 else -1

            if swipe_distance > 500:

                created_shapes.clear()

                print("Shapes cleared!")

                swipe_distance = 0
                swipe_direction = 0

        previous_hand_x = current_x

    else:
        previous_hand_x = None
        swipe_distance = 0
        swipe_direction = 0


    hands_are_together = False

    if len(hand_points) == 2:

        distance_between_hands = distance(
            hand_points[0],
            hand_points[1]
        )

        cv2.putText(
            frame,
            f"Distance: {int(distance_between_hands)}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        if distance_between_hands < 60:

            hands_are_together = True

            if not hands_were_together:

                center_x = (
                    hand_points[0][0] +
                    hand_points[1][0]
                ) // 2

                center_y = (
                    hand_points[0][1] +
                    hand_points[1][1]
                ) // 2

                created_shapes.append(
                    (center_x, center_y, 40)
                )

        hands_were_together = hands_are_together

    for x, y, radius in created_shapes:

        cv2.circle(
            frame,
            (x, y),
            radius,
            (255, 0, 255),
            3
        )

    face_result = face_landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )

    for face in face_result.face_landmarks:

        for landmark in face:

            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])

            cv2.circle(
                frame,
                (x, y),
                1,
                (255, 0, 0),
                -1
            )

    if face_result.face_blendshapes:

        blendshapes = face_result.face_blendshapes[0]

        if blendshapes:

            strongest = max(
                blendshapes,
                key=lambda x: x.score
            )

            text = (
                f"{strongest.category_name}: "
                f"{strongest.score:.2f}"
            )

            cv2.putText(
                frame,
                text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

    cv2.imshow("CamTracker", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

camera.release()
cv2.destroyAllWindows()

hand_landmarker.close()
face_landmarker.close()