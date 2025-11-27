import cv2
import mediapipe as mp
import numpy as np
import os

mp_face_mesh = mp.solutions.face_mesh

def analyze_eye_features(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
        result = face_mesh.process(img_rgb)

        if not result.multi_face_landmarks:
            print("No face detected!")
            return None

        h, w, _ = img.shape
        landmarks = result.multi_face_landmarks[0].landmark

        # Left eye landmark indices (MediaPipe)
        left_eye_idx = [33, 133, 159, 145]
        right_eye_idx = [362, 263, 386, 374]

        def get_eye_box(idx_list):
            points = np.array([[int(landmarks[i].x*w), int(landmarks[i].y*h)] for i in idx_list])
            x, y, w_e, h_e = cv2.boundingRect(points)
            return (x, y, w_e, h_e)

        left_eye = get_eye_box(left_eye_idx)
        right_eye = get_eye_box(right_eye_idx)

        # Draw rectangle
        for (x, y, w_e, h_e) in [left_eye, right_eye]:
            cv2.rectangle(img, (x, y), (x+w_e, y+h_e), (0,255,0), 2)

        os.makedirs("results", exist_ok=True)
        output_path = "results/output.jpg"
        cv2.imwrite(output_path, img)
        print(f"Saved → {output_path}")

        return {"left_eye": left_eye, "right_eye": right_eye}

print(analyze_eye_features("test.jpeg"))
