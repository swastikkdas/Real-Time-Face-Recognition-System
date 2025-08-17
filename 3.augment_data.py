import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def augment_data(images, labels, augmentation_factor=3):
    """Apply data augmentation to increase dataset size"""
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.8, 1.2],
        zoom_range=0.1,
        horizontal_flip=False  # Don't flip faces
    )

    augmented_images = []
    augmented_labels = []

    for i in range(len(images)):
        img = images[i].reshape(1, 100, 100, 1)
        label = labels[i]

        # Add original
        augmented_images.append(images[i])
        augmented_labels.append(label)

        # Generate augmented versions
        count = 0
        for batch in datagen.flow(img, batch_size=1):
            augmented_images.append(batch[0].reshape(100, 100))
            augmented_labels.append(label)
            count += 1
            if count >= augmentation_factor:
                break

    return np.array(augmented_images), np.array(augmented_labels)