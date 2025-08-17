# import os
# import pickle
# import numpy as np
#
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten,
#                                      Dense, Dropout, BatchNormalization)
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import to_categorical
#
# # 1. Load processed data
# with open(os.path.join('clean_data', 'images.p'), 'rb') as f:
#     images = pickle.load(f)
# with open(os.path.join('clean_data', 'labels.p'), 'rb') as f:
#     labels = pickle.load(f)
#
# # 2. Encode labels
# le = LabelEncoder()
# labels_encoded = le.fit_transform(labels)
# labels_categorical = to_categorical(labels_encoded)
#
# # Save the LabelEncoder for inference
# with open('label_encoder.p', 'wb') as f:
#     pickle.dump(le, f)
#
# # 3. Preprocess images
# images = images.reshape(-1, 100, 100, 1).astype('float32') / 255.0
#
# # 4. Train-test split
# X_train, X_test, y_train, y_test = train_test_split(
#     images, labels_categorical, test_size=0.2, random_state=42
# )
#
# # 5. Build CNN model
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
#     BatchNormalization(),
#     MaxPooling2D(),
#     Dropout(0.25),
#
#     Conv2D(64, (3, 3), activation='relu'),
#     BatchNormalization(),
#     MaxPooling2D(),
#     Dropout(0.25),
#
#     Conv2D(128, (3, 3), activation='relu'),
#     BatchNormalization(),
#     MaxPooling2D(),
#     Dropout(0.25),
#
#     Flatten(),
#     Dense(512, activation='relu'),
#     BatchNormalization(),
#     Dropout(0.5),
#
#     Dense(len(le.classes_), activation='softmax')
# ])
#
# # 6. Compile and train
# model.compile(optimizer=Adam(0.001),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
#
# history = model.fit(
#     X_train, y_train,
#     validation_data=(X_test, y_test),
#     epochs=50,
#     batch_size=32
# )
# import os
#
# main_folder = 'images'
#
# for person_name in os.listdir(main_folder):
#     person_folder = os.path.join(main_folder, person_name)
#     if os.path.isdir(person_folder):
#         print(f"Processing images for {person_name}")
#         for filename in os.listdir(person_folder):
#             if filename.endswith('.jpg') or filename.endswith('.png'):
#                 file_path = os.path.join(person_folder, filename)
#                 print(file_path)  # use cv2.imread(file_path) to read the image
#
#
# # 7. Save the trained model
# model.save('final_model.h5')
# print("Model and label encoder saved successfully.")
import os
import cv2
import pickle
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten,
                                     Dense, Dropout, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# 1. Load all images and labels from folder structure
main_folder = 'images'
image_data = []
labels = []

for person_name in os.listdir(main_folder):
    person_folder = os.path.join(main_folder, person_name)
    if os.path.isdir(person_folder):
        print(f"Processing images for {person_name}")
        for filename in os.listdir(person_folder):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                file_path = os.path.join(person_folder, filename)
                image = cv2.imread(file_path)
                if image is not None:
                    # Preprocess: grayscale & resize
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    image = cv2.resize(image, (100, 100))
                    image_data.append(image)
                    labels.append(person_name)  # label per folder
                else:
                    print(f"Warning: Could not read {file_path}")

# Convert to numpy arrays
image_data = np.array(image_data)
labels = np.array(labels)

if len(labels) == 0:
    raise ValueError("No face data found! Please check your image folders.")

# 2. Encode labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)

# Save the LabelEncoder for inference
with open('label_encoder.p', 'wb') as f:
    pickle.dump(le, f)

# 3. Preprocess images (reshape and scale)
images = image_data.reshape(-1, 100, 100, 1).astype('float32') / 255.0

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    images, labels_categorical, test_size=0.2, random_state=42
)

# 5. Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.25),

    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),

    Dense(len(le.classes_), activation='softmax')
])

# 6. Compile and train
model.compile(optimizer=Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32
)

# 7. Save the trained model
model.save('final_model.h5')
print("Model and label encoder saved successfully.")
