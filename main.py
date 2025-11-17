from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import tempfile
import cv2
import mediapipe as mp

app = FastAPI()

POSE_LANDMARKS = 33
HAND_LANDMARKS = 21

mp_holistic = mp.solutions.holistic

def extract_32x162(video_path):
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ok, frame = cap.read()
        if not ok: break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)

        pose = np.zeros((POSE_LANDMARKS, 3), dtype=np.float32)
        right_hand = np.zeros((HAND_LANDMARKS, 3), dtype=np.float32)

        if results.pose_landmarks:
            for i, lm in enumerate(results.pose_landmarks.landmark):
                pose[i] = [lm.x, lm.y, lm.z]

        if results.right_hand_landmarks:
            for i, lm in enumerate(results.right_hand_landmarks.landmark):
                right_hand[i] = [lm.x, lm.y, lm.z]

        feat = np.concatenate([pose.flatten(), right_hand.flatten()])
        frames.append(feat)

    cap.release()

    # Pad/trim to 32
    if len(frames) == 0:
        return None
    
    frames = frames[:32]
    while len(frames) < 32:
        frames.append(frames[-1])

    arr = np.stack(frames, axis=0).astype(float)
    return arr

@app.post("/extract")
async def extract_landmarks(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    features = extract_32x162(tmp_path)

    if features is None:
        return JSONResponse({"error": "no_landmarks"}, status_code=400)

    return {
        "landmarks": features.tolist()
    }
