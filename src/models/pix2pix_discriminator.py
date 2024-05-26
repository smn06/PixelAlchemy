import tensorflow as tf
from tensorflow.keras.layers import Input, Concatenate, Conv2D, LeakyReLU, Flatten, Dense
from tensorflow.keras.models import Model

def build_discriminator(input_shape=(256, 256, 3), output_channels=3):
    """
    Build the discriminator model for Pix2Pix.
    
    Parameters:
    - input_shape: Tuple, shape of the input image (height, width, channels).
    - output_channels: Integer, number of output channels (e.g., 3 for RGB).
    
    Returns:
    - Discriminator model.
    """
    # Input images
    input_image = Input(shape=input_shape, name='input_image')
    target_image = Input(shape=input_shape, name='target_image')
    
    # Concatenate input and target images
    combined_input = Concatenate()([input_image, target_image])
    
    # Convolutional layers
    conv1 = Conv2D(64, (4, 4), strides=(2, 2), padding='same')(combined_input)
    conv1 = LeakyReLU(alpha=0.2)(conv1)
    
    conv2 = Conv2D(128, (4, 4), strides=(2, 2), padding='same')(conv1)
    conv2 = LeakyReLU(alpha=0.2)(conv2)
    
    conv3 = Conv2D(256, (4, 4), strides=(2, 2), padding='same')(conv2)
    conv3 = LeakyReLU(alpha=0.2)(conv3)
    
    conv4 = Conv2D(512, (4, 4), padding='same')(conv3)
    conv4 = LeakyReLU(alpha=0.2)(conv4)
    
    # Output layer
    output = Conv2D(1, (4, 4), padding='same')(conv4)
    
    # Build model
    discriminator = Model(inputs=[input_image, target_image], outputs=output, name='discriminator')
    
    return discriminator
