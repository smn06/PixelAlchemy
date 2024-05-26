import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Mean
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from pix2pix_generator import build_generator
from pix2pix_discriminator import build_discriminator
import os

# Constants
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 2e-4
LAMBDA = 100  # Regularization parameter for cycle-consistency loss

# Directories
TRAIN_DIR = '../data/processed/satellite_to_map/train/'
TEST_DIR = '../data/processed/satellite_to_map/test/'
CHECKPOINT_DIR = './checkpoints/'

def load_data(train_dir, test_dir):
    """
    Load train and test datasets.
    
    Parameters:
    - train_dir: Path to the training data directory.
    - test_dir: Path to the testing data directory.
    
    Returns:
    - train_dataset: TensorFlow Dataset object for training.
    - test_dataset: TensorFlow Dataset object for testing.
    """
    train_satellite_paths = [os.path.join(train_dir, 'satellite_images', filename) for filename in os.listdir(os.path.join(train_dir, 'satellite_images'))]
    train_map_paths = [os.path.join(train_dir, 'map_images', filename) for filename in os.listdir(os.path.join(train_dir, 'map_images'))]
    test_satellite_paths = [os.path.join(test_dir, 'satellite_images', filename) for filename in os.listdir(os.path.join(test_dir, 'satellite_images'))]
    test_map_paths = [os.path.join(test_dir, 'map_images', filename) for filename in os.listdir(os.path.join(test_dir, 'map_images'))]
    
    train_dataset = tf.data.Dataset.from_tensor_slices((train_satellite_paths, train_map_paths))
    train_dataset = train_dataset.shuffle(len(train_satellite_paths)).map(load_image_pair, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((test_satellite_paths, test_map_paths))
    test_dataset = test_dataset.map(load_image_pair, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE)
    
    return train_dataset, test_dataset

def load_image_pair(satellite_path, map_path):
    """
    Load and preprocess a pair of satellite and map images.
    
    Parameters:
    - satellite_path: Path to the satellite image file.
    - map_path: Path to the corresponding map image file.
    
    Returns:
    - Tuple of preprocessed satellite and map images.
    """
    satellite = tf.io.read_file(satellite_path)
    satellite = tf.image.decode_jpeg(satellite, channels=3)
    satellite = tf.image.resize(satellite, (256, 256))
    satellite = tf.cast(satellite, tf.float32) / 127.5 - 1.0
    
    map_image = tf.io.read_file(map_path)
    map_image = tf.image.decode_jpeg(map_image, channels=3)
    map_image = tf.image.resize(map_image, (256, 256))
    map_image = tf.cast(map_image, tf.float32) / 127.5 - 1.0
    
    return satellite, map_image

def generator_loss(discriminator_real_outputs, generated_outputs, target_images):
    """
    Calculate the generator loss.
    
    Parameters:
    - discriminator_real_outputs: Outputs from the discriminator for real images.
    - generated_outputs: Outputs from the discriminator for generated images.
    - target_images: Target images (ground truth).
    
    Returns:
    - Total generator loss.
    """
    adversarial_loss = BinaryCrossentropy(from_logits=True)(tf.ones_like(discriminator_real_outputs), generated_outputs)
    # Mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target_images - generated_outputs))
    total_gen_loss = adversarial_loss + (LAMBDA * l1_loss)
    return total_gen_loss

def discriminator_loss(discriminator_real_outputs, discriminator_generated_outputs):
    """
    Calculate the discriminator loss.
    
    Parameters:
    - discriminator_real_outputs: Outputs from the discriminator for real images.
    - discriminator_generated_outputs: Outputs from the discriminator for generated images.
    
    Returns:
    - Total discriminator loss.
    """
    real_loss = BinaryCrossentropy(from_logits=True)(tf.ones_like(discriminator_real_outputs), discriminator_real_outputs)
    generated_loss = BinaryCrossentropy(from_logits=True)(tf.zeros_like(discriminator_generated_outputs), discriminator_generated_outputs)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss

def train_step(generator, discriminator, input_images, target_images, generator_optimizer, discriminator_optimizer, training=True):
    """
    Perform one training step.
    
    Parameters:
    - generator: Generator model.
    - discriminator: Discriminator model.
    - input_images: Batch of input images (satellite images).
    - target_images: Batch of target images (map images).
    - generator_optimizer: Optimizer for the generator.
    - discriminator_optimizer: Optimizer for the discriminator.
    - training: Boolean, whether in training mode.
    
    Returns:
    - Generator loss, discriminator loss.
    """
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(input_images, training=training)
        
        discriminator_real_outputs = discriminator([input_images, target_images], training=training)
        discriminator_generated_outputs = discriminator([input_images, generated_images], training=training)
        
        gen_loss = generator_loss(discriminator_real_outputs, discriminator_generated_outputs, target_images)
        disc_loss = discriminator_loss(discriminator_real_outputs, discriminator_generated_outputs)
        
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return gen_loss, disc_loss

def main():
    # Build generator and discriminator
    generator = build_generator()
    discriminator = build_discriminator()
    
    # Optimizers
    generator_optimizer = Adam(learning_rate=LEARNING_RATE, beta_1=0.5)
    discriminator_optimizer = Adam(learning_rate=LEARNING_RATE, beta_1=0.5)
    
    # Load data
    train_dataset, test_dataset = load_data(TRAIN_DIR, TEST_DIR)
    
    # Checkpoint
    checkpoint_prefix = os.path.join(CHECKPOINT_DIR, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)
    
    # Metrics
    gen_loss_metric = Mean(name='gen_loss')
    disc_loss_metric = Mean(name='disc_loss')
    
    # Training loop
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        
        # Reset metrics
        gen_loss_metric.reset_states()
        disc_loss_metric.reset_states()
        
        # Training
        for batch_num, (input_images, target_images) in enumerate(train_dataset):
            gen_loss, disc_loss = train_step(generator, discriminator, input_images, target_images,
                                             generator_optimizer, discriminator_optimizer)
            
            # Update metrics
            gen_loss_metric.update_state(gen_loss)
            disc_loss_metric.update_state(disc_loss)
            
            # Print training metrics
            if (batch_num + 1) % 100 == 0:
                print(f"Batch {batch_num + 1}, Generator Loss: {gen_loss_metric.result()}, Discriminator Loss: {disc_loss_metric.result()}")
        
        # Save checkpoint (every epoch)
        checkpoint.save(file_prefix=checkpoint_prefix)
        
        # Validation
        for batch_num, (input_images, target_images) in enumerate(test_dataset):
            _ = train_step(generator, discriminator, input_images, target_images,
                           generator_optimizer, discriminator_optimizer, training=False)
        
        # Print validation metrics
        print(f"Validation - Generator Loss: {gen_loss_metric.result()}, Discriminator Loss: {disc_loss_metric.result()}")

if __name__ == "__main__":
    main()
