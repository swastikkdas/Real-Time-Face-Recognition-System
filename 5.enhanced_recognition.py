import cv2
import numpy as np
import pickle
from keras.models import load_model
from collections import Counter

# Load your trained model
model = load_model("final_model.h5")

# Load the exact label encoder from training
with open("label_encoder.p", "rb") as f:
    le = pickle.load(f)
classes = le.classes_

# Haar cascade for face detection
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
if classifier.empty():
    raise IOError("Failed to load haarcascade_frontalface_default.xml")

# Inference settings
CONFIDENCE_THRESHOLD = 0.7
FRAME_BUFFER_SIZE = 10

def get_pred_label_with_confidence(pred):
    """Map model output to class name using the trained LabelEncoder."""
    max_prob = np.max(pred)
    if max_prob < CONFIDENCE_THRESHOLD:
        return "Unknown", max_prob
    class_idx = np.argmax(pred)
    return classes[class_idx], max_prob

def preprocess_enhanced(img):
    """Apply the same preprocessing used in training."""
    try:
        # Convert to grayscale if needed
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Resize
        img = cv2.resize(img, (100, 100))
        # CLAHE for lighting
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)
        # Gaussian blur
        img = cv2.GaussianBlur(img, (3, 3), 0)
        # Normalize and reshape for model
        img = img.reshape(1, 100, 100, 1).astype("float32") / 255.0
        return img
    except Exception as e:
        print("Error in preprocessing:", e)
        return None

# Frame buffering for stability
face_tracker = {}

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot access webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
        maxSize=(300, 300)
    )

    for i, (x, y, w, h) in enumerate(faces):
        # Add padding
        pad = 20
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)

        face = frame[y1:y2, x1:x2]
        hgt, wid = face.shape[:2]
        if hgt < 50 or wid < 50:
            continue

        processed = preprocess_enhanced(face)
        if processed is None:
            continue

        pred = model.predict(processed, verbose=0)[0]
        label, confidence = get_pred_label_with_confidence(pred)

        # Track predictions per face to smooth results
        face_id = f"face_{i}"
        buffer = face_tracker.setdefault(face_id, [])
        buffer.append(label)
        if len(buffer) > FRAME_BUFFER_SIZE:
            buffer.pop(0)
        stable_label = Counter(buffer).most_common(1)[0][0]

        # Draw results
        color = (0, 255, 0) if confidence > CONFIDENCE_THRESHOLD else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{stable_label}", #/({confidence:.2f})
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )


    cv2.imshow("Enhanced Face Recognition", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
