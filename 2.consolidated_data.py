import cv2
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

data_dir = os.path.join(os.getcwd(), 'clean_data')
img_dir = os.path.join(os.getcwd(), 'images')

# Create directories if they don't exist
os.makedirs(data_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)

image_data = []
labels = []

for filename in os.listdir(img_dir):
    file_path = os.path.join(img_dir, filename)
    # Filter for image files
    if not (filename.endswith(".jpg") or filename.endswith(".png")):
        continue

    image = cv2.imread(file_path)
    if image is None:
        print(f"Warning: Could not read image file {filename}")
        continue

    # Preprocess: resize and convert to grayscale
    image = cv2.resize(image, (100, 100))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_data.append(image)

    # Extract label (everything before first underscore)
    label = filename.split("_")[0]
    labels.append(label)

# Convert lists to numpy arrays
image_data = np.array(image_data)
labels = np.array(labels)

print(f"Loaded {len(image_data)} images")
print(f"Unique labels: {np.unique(labels)}")

# Show the 5th image for label 'UserX', if enough images exist
userx_indices = np.where(labels == 'UserX')[0]
if len(userx_indices) >= 5:
    idx = userx_indices[4]
    plt.imshow(image_data[idx], cmap="gray")
    plt.title("5th image of UserX")
    plt.axis('off')
    plt.show()
else:
    print("Not enough images labelled 'UserX' to display the 5th one.")

# Save processed data
with open(os.path.join(data_dir, "images.p"), 'wb') as f:
    pickle.dump(image_data, f)

with open(os.path.join(data_dir, "labels.p"), 'wb') as f:
    pickle.dump(labels, f)

print("Data processing completed!")
print(f"Images saved to: {os.path.join(data_dir, 'images.p')}")
print(f"Labels saved to: {os.path.join(data_dir, 'labels.p')}")