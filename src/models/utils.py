import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split

def load_image(image_path, target_size=(256, 256)):
    """Load and resize an image from disk."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return img_resized

def load_dataset(dataset_dir, test_size=0.2):
    """Load dataset from directory and split into train and test sets."""
    image_files = os.listdir(dataset_dir)
    images = []
    for image_file in image_files:
        image_path = os.path.join(dataset_dir, image_file)
        if os.path.isfile(image_path):
            image = load_image(image_path)
            images.append(image)
    
    # Split into train and test sets
    train_images, test_images = train_test_split(images, test_size=test_size, random_state=42)
    return np.array(train_images), np.array(test_images)

def save_image(image, save_path):
    """Save an image to disk."""
    cv2.imwrite(save_path, image)

def visualize_images(images, cols=4):
    """Visualize a list of images."""
    import matplotlib.pyplot as plt
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.flatten()
    for i, img in enumerate(images):
        axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()
