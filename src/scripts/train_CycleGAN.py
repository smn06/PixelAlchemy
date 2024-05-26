import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import Mean
from tensorflow.keras.callbacks import Callback
from cycle_gan import build_generator, build_discriminator, cycle_gan_loss

# Parameters
input_shape = (256, 256, 3)  # Input image shape
output_channels = 3          # RGB channels for output images
learning_rate = 0.0002       # Learning rate
lambda_cycle = 10.0          # Cycle consistency loss weight
epochs = 100                 # Number of training epochs
batch_size = 4               # Batch size

# Build generators and discriminators
G_AB = build_generator(input_shape, output_channels)  # Generator A to B
G_BA = build_generator(input_shape, output_channels)  # Generator B to A
D_A = build_discriminator(input_shape)               # Discriminator for domain A
D_B = build_discriminator(input_shape)               # Discriminator for domain B

# Loss function
generator_loss, discriminator_loss = cycle_gan_loss()

# Optimizers
optimizer_G = Adam(learning_rate, beta_1=0.5)
optimizer_D = Adam(learning_rate, beta_1=0.5)

# Compile models
G_AB.compile(optimizer=optimizer_G, loss=generator_loss)
G_BA.compile(optimizer=optimizer_G, loss=generator_loss)
D_A.compile(optimizer=optimizer_D, loss=discriminator_loss)
D_B.compile(optimizer=optimizer_D, loss=discriminator_loss)

# Print model summaries
G_AB.summary()
G_BA.summary()
D_A.summary()
D_B.summary()

# Data loading (dummy example, replace with your dataset loading logic)
def load_data(dataset_dir, batch_size):
    # Example: Load satellite and map images
    dataset_A = np.random.rand(batch_size, 256, 256, 3)  # Replace with your actual dataset loading
    dataset_B = np.random.rand(batch_size, 256, 256, 3)  # Replace with your actual dataset loading
    return dataset_A, dataset_B

# Training loop
def train_cycle_gan(dataset_A, dataset_B, epochs, batch_size):
    for epoch in range(epochs):
        # Iterate over batches
        for step in range(len(dataset_A) // batch_size):
            # Sample a batch of images
            batch_A = dataset_A[step * batch_size:(step + 1) * batch_size]
            batch_B = dataset_B[step * batch_size:(step + 1) * batch_size]

            # Train discriminators (each domain)
            with tf.GradientTape() as tape_D:
                fake_B = G_AB(batch_A, training=True)
                fake_A = G_BA(batch_B, training=True)

                real_A_output = D_A(batch_A, training=True)
                real_B_output = D_B(batch_B, training=True)
                fake_A_output = D_A(fake_A, training=True)
                fake_B_output = D_B(fake_B, training=True)

                D_A_loss = discriminator_loss(real_A_output, fake_A_output)
                D_B_loss = discriminator_loss(real_B_output, fake_B_output)

            gradients_D_A = tape_D.gradient(D_A_loss, D_A.trainable_variables)
            gradients_D_B = tape_D.gradient(D_B_loss, D_B.trainable_variables)

            optimizer_D.apply_gradients(zip(gradients_D_A, D_A.trainable_variables))
            optimizer_D.apply_gradients(zip(gradients_D_B, D_B.trainable_variables))

            # Train generators (cycle consistency and adversarial loss)
            with tf.GradientTape() as tape_G:
                fake_B = G_AB(batch_A, training=True)
                fake_A = G_BA(batch_B, training=True)

                reconstructed_A = G_BA(fake_B, training=True)
                reconstructed_B = G_AB(fake_A, training=True)

                real_A_output = D_A(batch_A, training=True)
                real_B_output = D_B(batch_B, training=True)
                fake_A_output = D_A(fake_A, training=True)
                fake_B_output = D_B(fake_B, training=True)

                G_AB_loss = generator_loss(fake_B_output)
                G_BA_loss = generator_loss(fake_A_output)
                cycle_loss_A = tf.reduce_mean(tf.abs(batch_A - reconstructed_A))
                cycle_loss_B = tf.reduce_mean(tf.abs(batch_B - reconstructed_B))
                total_cycle_loss = lambda_cycle * (cycle_loss_A + cycle_loss_B)

                G_loss = G_AB_loss + G_BA_loss + total_cycle_loss

            gradients_G = tape_G.gradient(G_loss, G_AB.trainable_variables + G_BA.trainable_variables)
            optimizer_G.apply_gradients(zip(gradients_G, G_AB.trainable_variables + G_BA.trainable_variables))

        # Print training progress
        print(f'Epoch {epoch+1}/{epochs}, Discriminator Loss: D_A={D_A_loss:.4f}, D_B={D_B_loss:.4f}, Generator Loss: G={G_loss:.4f}')

# Example usage
if __name__ == "__main__":
    # Load datasets (replace with your actual dataset loading)
    dataset_A, dataset_B = load_data('data/processed/satellite_to_map/train/', batch_size)

    # Train CycleGAN
    train_cycle_gan(dataset_A, dataset_B, epochs, batch_size)
