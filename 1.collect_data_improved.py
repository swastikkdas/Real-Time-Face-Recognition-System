import cv2
import numpy as np
import os

# Ensure the output folder exists
os.makedirs('images', exist_ok=True)

# Load both frontal and profile Haar cascades (must be in the same directory)
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
classifier_profile = cv2.CascadeClassifier("haarcascade_profileface.xml")
if classifier.empty() or classifier_profile.empty():
    raise IOError("Could not load one or both Haar cascade XML files.")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot access webcam")

data = []
face_count = 0
skip_frames = 5
frame_counter = 0

while len(data) < 200:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    if frame_counter % skip_frames != 0:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect frontal and profile faces
    faces_f = classifier.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
    faces_p = classifier_profile.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
    all_faces = list(faces_f) + list(faces_p)

    for (x, y, w, h) in all_faces:
        # Add padding
        pad = 20
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)

        face = frame[y1:y2, x1:x2]
        height, width = face.shape[:2]
        if height < 100 or width < 100:
            continue

        # Equalize lighting
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        eq_face = cv2.equalizeHist(gray_face)
        face_eq = cv2.cvtColor(eq_face, cv2.COLOR_GRAY2BGR)

        data.append(face_eq)
        face_count += 1
        print(f"Collected: {face_count}/200")
        break  # one face per frame

    cv2.putText(frame, f"Count: {len(data)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Collection", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save only if enough samples were collected
if len(data) >= 150:
    name = input("Enter name: ").strip()
    person_dir = os.path.join('images', name)
    os.makedirs(person_dir, exist_ok=True)
    for i, face in enumerate(data):
        cv2.imwrite(os.path.join(person_dir, f"{name}_{i}.jpg"), face)
    print("Data collection completed!")
else:
    print("Need more samples!")
