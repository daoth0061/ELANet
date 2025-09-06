"""
Dataset utilities for FaceForensics++ and custom face datasets
"""

import os
import random
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from .frequency_processing import extract_all_frequency_features, prepare_grayscale_for_haft


def read_ff_dataset(base_url: str, deepfake_type: str, train_ratio: float = 0.8) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Read FaceForensics++ dataset by deepfake type
    
    Args:
        base_url: Base directory path to FF+ dataset
        deepfake_type: Type of deepfake {deepfakes, face2face, faceshifter, faceswap, neuraltexture}
        train_ratio: Ratio for train/test split (default: 0.8)
        
    Returns:
        Tuple containing:
        - train_real_url: List of training real image paths
        - test_real_url: List of testing real image paths  
        - train_fake_url: List of training fake image paths
        - test_fake_url: List of testing fake image paths
    """
    train_real_url = []
    test_real_url = []
    train_fake_url = []
    test_fake_url = []

    # List all video IDs
    vid_ids = os.listdir(base_url)
    vid_ids = [vid_id for vid_id in vid_ids if '.' not in vid_id]

    # Number of training videos (Train : Test = train_ratio : (1-train_ratio))
    num_vid_train = int(train_ratio * len(vid_ids))

    # Read dataset
    for i, vid_id in enumerate(vid_ids):
        # Define deepfake and original path based on folder structure
        deepfake_folder_path = f"{base_url}/{vid_id}/{deepfake_type}"
        original_folder_path = f"{base_url}/{vid_id}/original"
        
        # Check if directories exist
        if not os.path.exists(deepfake_folder_path) or not os.path.exists(original_folder_path):
            continue
        
        if i < num_vid_train:
            for image_name in os.listdir(deepfake_folder_path):
                train_fake_url.append(os.path.join(deepfake_folder_path, image_name))
                train_real_url.append(os.path.join(original_folder_path, image_name))
        else:
            for image_name in os.listdir(deepfake_folder_path):
                test_fake_url.append(os.path.join(deepfake_folder_path, image_name))
                test_real_url.append(os.path.join(original_folder_path, image_name))

    return train_real_url, test_real_url, train_fake_url, test_fake_url


class FaceDataset(Dataset):
    """
    Custom Dataset for face manipulation detection
    
    Supports both RGB images and frequency domain features with HAFT processing
    """
    
    def __init__(self, 
                 list_img_url: List[str], 
                 config,
                 transform=None, 
                 real_transform=None,
                 is_training: bool = True):
        """
        Initialize FaceDataset
        
        Args:
            list_img_url: List of image file paths
            config: Configuration object
            transform: Transform for fake images
            real_transform: Transform for real images
            is_training: Whether this is training dataset
        """
        self.list_img_url = list_img_url
        self.config = config
        self.transform = transform
        self.real_transform = real_transform
        self.is_training = is_training
        
        # Get image sizes from config
        self.rgb_size = tuple(config.get('data_processing.image_size', [256, 256]))
        self.freq_size = tuple(config.get('data_processing.frequency_size', [160, 160]))
        
    def __len__(self):
        return len(self.list_img_url)
        
    def __getitem__(self, idx):
        img_url = self.list_img_url[idx]
        
        try:
            img = Image.open(img_url)
        except Exception as e:
            print(f"Error loading image {img_url}: {e}")
            # Return a default/fallback item or raise exception
            raise
        
        # Determine if image is real or fake
        if '/original/' in img_url:
            # Real image
            mask = np.zeros(self.rgb_size, dtype='float32')
            mask = torch.from_numpy(mask).unsqueeze(dim=0)
            label = 0
            rgb_img = self.real_transform(img) if self.real_transform else self.transform(img)
        else:
            # Fake image - load corresponding mask
            mask_url = img_url.replace('FF+', 'FF+Mask')
            try:
                mask = Image.open(mask_url)
                mask = self.transform(mask) if self.transform else transforms.ToTensor()(mask)
            except:
                # If mask not found, create zero mask
                mask = np.zeros(self.rgb_size, dtype='float32')
                mask = torch.from_numpy(mask).unsqueeze(dim=0)
            
            label = 1
            rgb_img = self.transform(img) if self.transform else transforms.ToTensor()(img)

        # Extract frequency features
        try:
            freq_img = extract_all_frequency_features(img_url, self.config)
            freq_img = self.transform(freq_img)
        except Exception as e:
            print(f"Error extracting frequency features from {img_url}: {e}")
            # Create fallback frequency features
            freq_img = torch.zeros((8, self.freq_size[1], self.freq_size[0]))
        
        # Prepare grayscale image for HAFT processing
        try:
            gray_img = prepare_grayscale_for_haft(img_url, target_size=self.freq_size)
        except Exception as e:
            print(f"Error preparing grayscale image from {img_url}: {e}")
            # Create fallback grayscale image
            gray_img = torch.zeros((1, self.freq_size[1], self.freq_size[0]))
        
        return rgb_img, freq_img, gray_img, mask, label


def create_transforms(config):
    """
    Create data transforms based on configuration
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (transform, real_transform)
    """
    image_size = tuple(config.get('data_processing.image_size', [256, 256]))
    
    # Standard transform for fake images and testing
    transform = transforms.Compose([
        transforms.ToTensor(),  
        transforms.Resize(image_size)
    ])

    # Enhanced transform for real images during training
    real_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(),
        transforms.Resize(image_size)
    ])
    
    return transform, real_transform


def create_datasets(config):
    """
    Create training and testing datasets
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    # Set random seed for reproducibility
    random.seed(config.get('dataset.random_seed', 42))
    
    # Read dataset
    base_url = config.get('dataset.base_url')
    deepfake_type = config.get('dataset.deepfake_type', 'deepfakes')
    train_ratio = config.get('dataset.train_test_split', 0.8)
    
    train_real_url, test_real_url, train_fake_url, test_fake_url = read_ff_dataset(
        base_url, deepfake_type, train_ratio
    )
    
    # Combine real and fake URLs
    train_list = train_real_url + train_fake_url 
    test_list = test_real_url + test_fake_url
    
    print(f"Train real images: {len(train_real_url)}")
    print(f"Test real images: {len(test_real_url)}")
    print(f"Train fake images: {len(train_fake_url)}")
    print(f"Test fake images: {len(test_fake_url)}")
    
    # Create transforms
    transform, real_transform = create_transforms(config)
    
    # Create datasets
    train_dataset = FaceDataset(
        train_list, 
        config,
        transform=transform, 
        real_transform=real_transform,
        is_training=True
    )
    
    test_dataset = FaceDataset(
        test_list, 
        config,
        transform=transform, 
        real_transform=transform,  # Use same transform for test
        is_training=False
    )
    
    return train_dataset, test_dataset
