### Real-Time Face Recognition System
A robust, modular pipeline for real-time face detection and identification using OpenCV, TensorFlow/Keras, and Haar Cascades.
Designed for live video feeds, automated person recognition, and dataset-centric training workflows.


### 🚀 Introduction
This project showcases a fully operational face recognition system that:

Captures faces from a webcam in real-time.

Detects and preprocesses faces with high accuracy.

Recognizes identities using a trained deep learning model.

Annotates video streams with bounding boxes and predicted names.

Supports both dataset creation and real-world deployment.

### 👇 Pipeline Stages:

Data Collection: Gather, crop, and store face images per individual using your webcam.

Data Consolidation: Organize images, preprocess (resize, grayscale), and label for training.

Recognition: Detect faces on live video and identify each using a trained neural network.


### 🌟 Core Features
Real-time face detection (frontal/profile) via Haar Cascades.

Consistent preprocessing: cropping, padding, grayscale, lighting equalization.

Data augmentation for efficient training.

Deep learning model (CNN) for face recognition (TensorFlow/Keras backend).

Live annotation: bounding boxes, confidence scores, and identities in webcam feeds.

Structured export for dataset reuse and retraining.


### 🛠️ Technologies Used
Python 3.x

OpenCV (cv2)

NumPy

Matplotlib

TensorFlow/Keras

scikit-learn

Pickle (for data serialization)

Haar Cascade Classifiers


### ⚡ System Workflow
1️⃣ Data Collection
Collects 150–200 face samples per user via webcam.

Automatically applies lighting equalization and cropping.

Stores images as <name>_<index>.jpg in images/<name>/.

2️⃣ Data Consolidation
Loads, preprocesses, and labels captured face images.

Converts images to consistent shape and grayscale.

Saves processed data in clean_data/images.p and clean_data/labels.p.

3️⃣ Data Augmentation (Optional)
Uses Keras ImageDataGenerator for rotations, shifts, zoom, and lighting changes.

4️⃣ Model Training
Trains a CNN on the preprocessed dataset.

Saves final model as final_model.h5 and label encoder as label_encoder.p.

5️⃣ Real-Time Face Recognition
Detects faces on live webcam feeds.

Processes images to match training (CLAHE, blur, normalize).

Predicts identities and overlays results (name, confidence).


### 📦 Setup Instructions
Prerequisites
Python 3.x installed.

Clone this repo and navigate to the folder.

Install required libraries:

bash
pip install -r requirements.txt
Place haarcascade_frontalface_default.xml and haarcascade_profileface.xml in your project directory.

Quick Start
1. Data Collection
bash
python 1.collect_data_improved.py
Enter the user's name when prompted.

Collect ≥ 150 samples per individual.

2. Data Consolidation
bash
python 2.consolidated_data.py
3. Data Augmentation (Optional)
python
from augment_data import augment_data
aug_images, aug_labels = augment_data(images, labels, augmentation_factor=3)
4. Model Training
bash
python 4.train_model.py
5. Real-Time Recognition
bash
python 5.enhanced_recognition.py
View annotated webcam frames in real-time.

Press q to quit.


***

> ### 📁 Project Structure
>
> ```
> ├── 1.collect_data_improved.py    # Collects face images using webcam
> ├── 2.consolidated_data.py        # Consolidates collected images into datasets
> ├── 3.augment_data.py             # Data augmentation (rotation, shift, zoom, brightness)
> ├── 4.train_model.py              # CNN training script
> ├── 5.enhanced_recognition.py     # Real-time recognition with enhancements
> ├── images/                       # Stored face images (per user name)
> ├── clean_data/                   # Processed data (pickled images & labels)
> ├── final_model.h5                # Trained CNN model (generated after training)
> ├── label_encoder.p               # Saved LabelEncoder (for consistent label mapping)
> └── README.md                     # Documentation
> ```

***


***

> ### 🧩 How It Works
>
> | **Stage**                | **Action**                                             |
> |--------------------------|--------------------------------------------------------|
> | Data Collection          | Webcam feed → Detect → Preprocess → Save faces         |
> | Data Consolidation       | Resize, grayscale, label → Export pickled data         |
> | Data Augmentation        | Keras generator → Expand with random transforms        |
> | Model Training           | CNN architecture (Conv+Pooling+BN+Dropout+Dense)       |
> | Real-Time Recognition    | Detect faces → Preprocess → Predict → Annotate live feed |

***

***

> ### 📈 Flow Diagram
>
> ```
> [ Webcam Feed ]
>       ↓
> [ Data Collection ]
>       ↓
> [ Data Consolidation ]
>       ↓
> [ Data Augmentation (optional) ]
>       ↓
> [ Model Training ]
>       ↓
> [ Real-Time Recognition ]
>       ↓
> [ Annotated Video Feed ]
> ```

***




### 🌍 Potential Applications
Security & surveillance

Attendance tracking for workplaces/schools

Smart device authentication

Personalized services


### 🎯 Future Enhancements
Integrate emotion recognition or age/gender estimation.

Multi-user recognition in crowded frames.

Expand dataset for improved robustness.

Deploy as a web service or mobile app.

Use more advanced models (FaceNet, ArcFace).


### 📬 Contact
For questions or contributions, contact:
[swastikdasoff@gmail.com]
