import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from cyclegan_generator import build_generator
from cyclegan_discriminator import build_discriminator

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

def visualize_results(generator_AtoB, generator_BtoA, input_image_path, output_image_path_AtoB, output_image_path_BtoA):
    """
    Visualize the results of the CycleGAN model.
    
    Parameters:
    - generator_AtoB: Generator model from domain A to domain B.
    - generator_BtoA: Generator model from domain B to domain A.
    - input_image_path: Path to the input image (from domain A).
    - output_image_path_AtoB: Path to save the generated image (from domain A to B).
    - output_image_path_BtoA: Path to save the generated image (from domain B to A).
    """
    # Load input image
    input_image = load_image(input_image_path)
    
    # Generate images
    input_tensor = tf.expand_dims(input_image, 0)
    generated_image_AtoB = generator_AtoB(input_tensor, training=False)[0].numpy()
    generated_image_BtoA = generator_BtoA(input_tensor, training=False)[0].numpy()
    
    # Rescale images from [-1, 1] to [0, 1]
    input_image = (input_image + 1) / 2.0
    generated_image_AtoB = (generated_image_AtoB + 1) / 2.0
    generated_image_BtoA = (generated_image_BtoA + 1) / 2.0
    
    # Plot images
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.title('Input Image (Domain A)')
    plt.imshow(input_image)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title('Generated Image (Domain A to B)')
    plt.imshow(generated_image_AtoB)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title('Generated Image (Domain B to A)')
    plt.imshow(generated_image_BtoA)
    plt.axis('off')
    
    # Save or show plot
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(output_image_path_AtoB)
    plt.show()
    
    # Save or show plot for Domain B to A
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.title('Input Image (Domain B)')
    plt.imshow(input_image)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title('Generated Image (Domain B to A)')
    plt.imshow(generated_image_BtoA)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title('Generated Image (Domain A to B)')
    plt.imshow(generated_image_AtoB)
    plt.axis('off')
    
    # Save or show plot
    plt.savefig(output_image_path_BtoA)
    plt.show()

def main():
    # Build generators
    generator_AtoB = build_generator()
    generator_BtoA = build_generator()
    
    # Restore checkpoints
    checkpoint_AtoB = tf.train.Checkpoint(generator=generator_AtoB)
    checkpoint_BtoA = tf.train.Checkpoint(generator=generator_BtoA)
    checkpoint_AtoB.restore(tf.train.latest_checkpoint(os.path.join(CHECKPOINT_DIR, 'AtoB')))
    checkpoint_BtoA.restore(tf.train.latest_checkpoint(os.path.join(CHECKPOINT_DIR, 'BtoA')))
    
    # Example visualization
    input_image_path = '../data/trainA/example_image.jpg'
    output_image_path_AtoB = os.path.join(OUTPUT_DIR, 'generated_image_AtoB.jpg')
    output_image_path_BtoA = os.path.join(OUTPUT_DIR, 'generated_image_BtoA.jpg')
    
    visualize_results(generator_AtoB, generator_BtoA, input_image_path, output_image_path_AtoB, output_image_path_BtoA)

if __name__ == "__main__":
    main()
