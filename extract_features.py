import os
import cv2
import numpy as np
import pickle
from skimage.feature import hog
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

DATASET_PATH = "dataset"

SETTINGS = {
    "A": {
        "orientations": 9,
        "pixels_per_cell": (8, 8),
        "cells_per_block": (2, 2)
    },
    "B": {
        "orientations": 9,
        "pixels_per_cell": (16, 16),
        "cells_per_block": (2, 2)
    }
}


def extract_hog_features(image, params, visualize=False):
    image = cv2.resize(image, (128, 128))

    if visualize:
        features, hog_image = hog(
            image,
            orientations=params["orientations"],
            pixels_per_cell=params["pixels_per_cell"],
            cells_per_block=params["cells_per_block"],
            visualize=True
        )
        return features, hog_image

    features = hog(
        image,
        orientations=params["orientations"],
        pixels_per_cell=params["pixels_per_cell"],
        cells_per_block=params["cells_per_block"]
    )

    return features


for setting_name, params in SETTINGS.items():

    X = []
    y = []

    classes = os.listdir(DATASET_PATH)

    for class_name in classes:

        class_path = os.path.join(DATASET_PATH, class_name)

        for img_name in os.listdir(class_path):

            img_path = os.path.join(class_path, img_name)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            features = extract_hog_features(img, params)

            X.append(features)
            y.append(class_name)

    X = np.array(X)
    y = np.array(y)

    print(f"\nSetting {setting_name}")
    print("Feature Vector Length:", X.shape[1])

    with open(f"features_{setting_name}.pkl", "wb") as f:
        pickle.dump((X, y), f)

print("Features saved successfully!")

# HOG Visualization for Setting A
params = SETTINGS["A"]

for class_name in os.listdir(DATASET_PATH):

    img_file = os.listdir(os.path.join(DATASET_PATH, class_name))[0]

    img_path = os.path.join(DATASET_PATH, class_name, img_file)

    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    image = cv2.resize(image, (128,128))

    features, hog_img = extract_hog_features(
        image,
        params,
        visualize=True
    )

    plt.figure(figsize=(8,4))

    plt.subplot(1,2,1)
    plt.imshow(image, cmap='gray')
    plt.title("Original")

    plt.subplot(1,2,2)
    plt.imshow(hog_img, cmap='gray')
    plt.title("HOG")

    plt.savefig(f"{class_name}_hog.png")
    plt.close()

print("HOG visualizations saved.")
