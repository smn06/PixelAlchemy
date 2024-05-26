import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from pix2pix_generator import build_generator

# Constants
CHECKPOINT_DIR = './checkpoints/'
OUTPUT_DIR = './outputs/'

def load_image(image_path):
    """
    Load and preprocess a single image.
    
    Parameters:
    - image_path: Path to the image file.
    
    Returns:
    - Preprocessed image as a NumPy array.
    """
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (256, 256))
    img = tf.cast(img, tf.float32) / 127.5 - 1.0
    return img.numpy()

def visualize_results(generator, input_image_path, output_image_path, target_image_path):
    """
    Visualize the results of the Pix2Pix model.
    
    Parameters:
    - generator: Generator model.
    - input_image_path: Path to the input (satellite) image.
    - output_image_path: Path to save the generated (map) image.
    - target_image_path: Path to the target (ground truth map) image.
    """
    # Load images
    input_image = load_image(input_image_path)
    target_image = load_image(target_image_path)
    
    # Generate image
    input_tensor = tf.expand_dims(input_image, 0)
    generated_image = generator(input_tensor, training=False)[0].numpy()
    
    # Rescale images from [-1, 1] to [0, 1]
    input_image = (input_image + 1) / 2.0
    generated_image = (generated_image + 1) / 2.0
    target_image = (target_image + 1) / 2.0
    
    # Plot images
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 3, 1)
    plt.title('Input Satellite Image')
    plt.imshow(input_image)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title('Generated Map Image')
    plt.imshow(generated_image)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title('Target Map Image')
    plt.imshow(target_image)
    plt.axis('off')
    
    # Save or show plot
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(output_image_path)
    plt.show()

def main():
    # Load trained generator model
    generator = build_generator()
    checkpoint = tf.train.Checkpoint(generator=generator)
    checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_DIR))
    
    # Example visualization
    input_image_path = '../data/processed/satellite_to_map/test/satellite_images/example_satellite.jpg'
    output_image_path = os.path.join(OUTPUT_DIR, 'generated_map.jpg')
    target_image_path = '../data/processed/satellite_to_map/test/map_images/example_map.jpg'
    
    visualize_results(generator, input_image_path, output_image_path, target_image_path)

if __name__ == "__main__":
    main()
