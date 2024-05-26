import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

# Generator and Discriminator architectures for CycleGAN

def build_generator(input_shape, output_channels):
    """Builds the generator network for CycleGAN."""
    def conv_block(x, filters, kernel_size, strides=1, padding='same', activation='relu'):
        x = Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        return x
    
    def residual_block(x, filters, kernel_size=3):
        y = conv_block(x, filters, kernel_size)
        y = conv_block(y, filters, kernel_size, activation=None)
        return Add()([x, y])
    
    # Input
    input = Input(shape=input_shape)
    
    # Encoder
    x = conv_block(input, 64, 7)
    x = conv_block(x, 128, 3, strides=2)
    x = conv_block(x, 256, 3, strides=2)
    
    # Residual blocks
    for _ in range(6):
        x = residual_block(x, 256)
    
    # Decoder
    x = Conv2DTranspose(128, 3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2DTranspose(64, 3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Output
    output = Conv2D(output_channels, 7, activation='tanh', padding='same')(x)
    
    return Model(inputs=input, outputs=output)

def build_discriminator(input_shape):
    """Builds the discriminator network for CycleGAN."""
    def conv_block(x, filters, kernel_size, strides=2, padding='same', activation='relu'):
        x = Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        return x
    
    # Input
    input = Input(shape=input_shape)
    
    # Convolutional layers
    x = conv_block(input, 64, 4, strides=2)
    x = conv_block(x, 128, 4, strides=2)
    x = conv_block(x, 256, 4, strides=2)
    
    # Output
    output = Conv2D(1, 4, activation='sigmoid', padding='same')(x)
    
    return Model(inputs=input, outputs=output)

def cycle_gan_loss():
    """Defines CycleGAN loss function."""
    mse = MeanSquaredError()
    
    def generator_loss(fake_output):
        return mse(tf.ones_like(fake_output), fake_output)
    
    def discriminator_loss(real_output, fake_output):
        real_loss = mse(tf.ones_like(real_output), real_output)
        fake_loss = mse(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss
    
    return generator_loss, discriminator_loss
