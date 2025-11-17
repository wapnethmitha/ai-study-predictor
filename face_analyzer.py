from deepface import DeepFace
import cv2


def analyze_face(frame):
    """
    Analyze a single frame for emotion and attention.
    Returns detected emotion or None if no face.
    """
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result['dominant_emotion']
        return emotion
    except Exception as e:
        return None

def capture_face_emotion():
    """
    Captures from webcam and returns dominant emotion continuously.
    """
    cap = cv2.VideoCapture(0)
    emotions = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            emotion = analyze_face(frame)
            if emotion:
                emotions.append(emotion)
            cv2.imshow("Face Capture - Press 'q' to stop", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return emotions
