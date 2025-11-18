from deepface import DeepFace
import cv2
import time

def analyze_face(frame):
    try:
        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False
        )

        print("DeepFace raw result:", result)

        # New DeepFace versions return list
        if isinstance(result, list):
            result = result[0]

        emotion = result.get("dominant_emotion")
        print("Detected emotion:", emotion)
        return emotion

    except Exception as e:
        print("DeepFace error:", e)
        return None


def capture_face_emotion(duration=3):
    """
    Captures webcam for a fixed duration (default 3 sec)
    Returns list of detected emotions.
    No need to press any keyboard key.
    """
    cap = cv2.VideoCapture(0)
    emotions = []

    start_time = time.time()

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break

        emotion = analyze_face(frame)
        if emotion:
            emotions.append(emotion)

        # Show window (optional)
        cv2.imshow("Analyzing your focus...", frame)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()

    return emotions
