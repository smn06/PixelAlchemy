import os
import cv2  # OpenCV for image processing
import numpy as np
from sklearn.model_selection import train_test_split

# Directories
RAW_DATA_DIR = '../data/raw/'
PROCESSED_DATA_DIR = '../data/processed/satellite_to_map/'

# Function to resize images
def resize_image(img_path, output_size=(256, 256)):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img_resized = cv2.resize(img, output_size, interpolation=cv2.INTER_AREA)
    return img_resized

# Function to preprocess data
def preprocess_data(raw_satellite_dir, raw_map_dir, processed_dir, test_size=0.2):
    # Create directories if they don't exist
    train_dir = os.path.join(processed_dir, 'train')
    test_dir = os.path.join(processed_dir, 'test')
    satellite_train_dir = os.path.join(train_dir, 'satellite_images')
    map_train_dir = os.path.join(train_dir, 'map_images')
    satellite_test_dir = os.path.join(test_dir, 'satellite_images')
    map_test_dir = os.path.join(test_dir, 'map_images')
    
    os.makedirs(satellite_train_dir, exist_ok=True)
    os.makedirs(map_train_dir, exist_ok=True)
    os.makedirs(satellite_test_dir, exist_ok=True)
    os.makedirs(map_test_dir, exist_ok=True)
    
    # List all files in raw directories
    satellite_files = os.listdir(raw_satellite_dir)
    map_files = os.listdir(raw_map_dir)
    
    # Split into train and test sets
    satellite_train, satellite_test, map_train, map_test = train_test_split(
        satellite_files, map_files, test_size=test_size, random_state=42)
    
    # Process train images
    for sat_file, map_file in zip(satellite_train, map_train):
        sat_path = os.path.join(raw_satellite_dir, sat_file)
        map_path = os.path.join(raw_map_dir, map_file)
        
        sat_img_resized = resize_image(sat_path)
        map_img_resized = resize_image(map_path)
        
        # Save resized images to train directories
        cv2.imwrite(os.path.join(satellite_train_dir, sat_file), sat_img_resized)
        cv2.imwrite(os.path.join(map_train_dir, map_file), map_img_resized)
    
    # Process test images
    for sat_file, map_file in zip(satellite_test, map_test):
        sat_path = os.path.join(raw_satellite_dir, sat_file)
        map_path = os.path.join(raw_map_dir, map_file)
        
        sat_img_resized = resize_image(sat_path)
        map_img_resized = resize_image(map_path)
        
        # Save resized images to test directories
        cv2.imwrite(os.path.join(satellite_test_dir, sat_file), sat_img_resized)
        cv2.imwrite(os.path.join(map_test_dir, map_file), map_img_resized)
    
    print("Data preprocessing completed.")

if __name__ == "__main__":
    preprocess_data(RAW_DATA_DIR + 'satellite_images/', RAW_DATA_DIR + 'map_images/',
                    PROCESSED_DATA_DIR + 'train/', test_size=0.2)
