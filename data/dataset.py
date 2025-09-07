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
import torch.nn.functional as F


from .frequency_processing import extract_all_frequency_features, prepare_grayscale_for_haft


def read_ff_dataset(base_url: str, deepfake_types: List[str] = None, train_ratio: float = 0.8) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Read FaceForensics++ dataset by deepfake types
    
    Args:
        base_url: Base directory path to FF+ dataset
        deepfake_types: List of deepfake types {deepfakes, face2face, faceshifter, faceswap, neuraltexture}
                       If None, use all available types
        train_ratio: Ratio for train/test split (default: 0.8)
        
    Returns:
        Tuple containing:
        - train_real_url: List of training real image paths
        - test_real_url: List of testing real image paths  
        - train_fake_url: List of training fake image paths
        - test_fake_url: List of testing fake image paths
    """
    if deepfake_types is None:
        deepfake_types = ['deepfakes', 'face2face', 'faceshifter', 'faceswap', 'neuraltexture']
    
    # Ensure deepfake_types is a list
    if isinstance(deepfake_types, str):
        deepfake_types = [deepfake_types]
    
    train_real_url = []
    test_real_url = []
    train_fake_url = []
    test_fake_url = []

    # List all video IDs
    vid_ids = os.listdir(base_url)
    vid_ids = [vid_id for vid_id in vid_ids if '.' not in vid_id]

    # Number of training videos (Train : Test = train_ratio : (1-train_ratio))
    num_vid_train = int(train_ratio * len(vid_ids))

    # First, collect all real images (only once, not per deepfake type)
    for i, vid_id in enumerate(vid_ids):
        original_folder_path = f"{base_url}/{vid_id}/original"
        
        # Check if original folder exists
        if not os.path.exists(original_folder_path):
            continue
        
        # Get corresponding real images (only once per video)
        real_images = []
        for image_name in os.listdir(original_folder_path):
            real_images.append(os.path.join(original_folder_path, image_name))
        
        # Split into train/test
        if i < num_vid_train:
            train_real_url.extend(real_images)
        else:
            test_real_url.extend(real_images)

    # Then, collect fake images for each deepfake type
    for deepfake_type in deepfake_types:
        print(f"Loading deepfake type: {deepfake_type}")
        
        for i, vid_id in enumerate(vid_ids):
            # Define deepfake path
            deepfake_folder_path = f"{base_url}/{vid_id}/{deepfake_type}"
            
            # Check if deepfake folder exists
            if not os.path.exists(deepfake_folder_path):
                print(f"Warning: {deepfake_folder_path} does not exist, skipping.")
                continue
            
            # Process fake images - check if mask exists
            fake_images = []
            for image_name in os.listdir(deepfake_folder_path):
                fake_img_path = os.path.join(deepfake_folder_path, image_name)
                mask_path = fake_img_path.replace('FF+', 'FF+Mask')
                
                # Only include fake image if its mask exists
                if os.path.exists(mask_path):
                    fake_images.append(fake_img_path)
                else:
                    print(f"Skipping {fake_img_path} - mask not found")
            
            # Split into train/test
            if i < num_vid_train:
                train_fake_url.extend(fake_images)
            else:
                test_fake_url.extend(fake_images)

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
                 is_training: bool = True,
                 augment_real_factor: int = 1):
        """
        Initialize FaceDataset
        
        Args:
            list_img_url: List of image file paths
            config: Configuration object
            transform: Transform for fake images
            real_transform: Transform for real images
            is_training: Whether this is training dataset
            augment_real_factor: Factor to augment real images (default: 1)
        """
        self.list_img_url = list_img_url
        self.config = config
        self.transform = transform
        self.real_transform = real_transform
        self.is_training = is_training
        self.augment_real_factor = augment_real_factor
        
        # Get image sizes from config
        self.rgb_size = tuple(config.get('data_processing.image_size', [256, 256]))
        self.freq_size = tuple(config.get('data_processing.frequency_size', [256, 256]))
        
        # If augmenting real images, expand the dataset
        if self.augment_real_factor > 1 and self.is_training:
            expanded_urls = []
            for img_url in self.list_img_url:
                if '/original/' in img_url:
                    # Add multiple copies of real images for augmentation
                    expanded_urls.extend([img_url] * self.augment_real_factor)
                else:
                    # Keep fake images as is
                    expanded_urls.append(img_url)
            self.list_img_url = expanded_urls
        
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
        
        # Apply augmentation first (keeping PIL format) - matching main file style
        if '/original/' in img_url:
            # Real image - apply augmentation (returns PIL image)
            augmented_img = self.real_transform(img) if self.real_transform else img
            label = 0
            # Create zero mask
            mask = np.zeros(self.rgb_size, dtype='float32')
            mask = torch.from_numpy(mask).unsqueeze(dim=0)
        else:
            # Fake image - no augmentation (returns PIL image)
            augmented_img = self.transform(img) if self.transform else img
            label = 1
            # Load actual mask
            mask_url = img_url.replace('FF+', 'FF+Mask')
            mask = Image.open(mask_url)
            mask = transforms.ToTensor()(mask)
            # Resize mask to target size
            mask = F.interpolate(mask.unsqueeze(0), size=self.rgb_size, mode='bilinear', align_corners=False).squeeze(0)
            # Normalize mask per channel  
            mask = (mask - mask.mean(dim=(1,2), keepdim=True)) / (mask.std(dim=(1,2), keepdim=True) + 1e-6)
                

        # Convert augmented PIL image to grayscale numpy array once for efficiency
        gray_img_array = np.array(augmented_img.convert('L'))

        # Extract frequency features using grayscale array
        try:
            freq_img = extract_all_frequency_features(gray_img_array, self.config)
        except Exception as e:
            print(f"Error extracting frequency features from {img_url}: {e}")
            # Create fallback frequency features
            freq_img = torch.zeros((8, self.freq_size[1], self.freq_size[0]))
        
        # Prepare grayscale image for HAFT processing using grayscale array
        try:
            gray_img = prepare_grayscale_for_haft(gray_img_array, self.config)
        except Exception as e:
            print(f"Error preparing grayscale image from {img_url}: {e}")
            # Create fallback grayscale image
            gray_img = torch.zeros((1, self.freq_size[1], self.freq_size[0]))
        
        # Convert augmented PIL image to tensor and process - matching main file style
        rgb_img = transforms.ToTensor()(augmented_img)
        rgb_img = F.interpolate(rgb_img.unsqueeze(0), size=self.rgb_size, mode='bilinear', align_corners=False).squeeze(0)
        
        return rgb_img, freq_img, gray_img, mask, label


def create_transforms(config):
    """
    Create data transforms based on configuration
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (transform, real_transform)
    """
    # Transform without resizing and without ToTensor - keep PIL format for frequency analysis
    transform = transforms.Compose([
        # No ToTensor here - we'll convert manually after frequency processing
    ])

    # Real transform with augmentation but without resizing and without ToTensor
    real_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter()
        # No ToTensor here - we'll convert manually after frequency processing
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
    deepfake_types = config.get('dataset.deepfake_types', ['deepfakes'])
    train_ratio = config.get('dataset.train_test_split', 0.8)
    
    # Set augment_real_factor to number of deepfake types if not specified
    augment_real_factor = config.get('dataset.augment_real_factor')
    if augment_real_factor is None:
        augment_real_factor = len(deepfake_types)
        # Update config with the computed value
        config['dataset']['augment_real_factor'] = augment_real_factor
    
    train_real_url, test_real_url, train_fake_url, test_fake_url = read_ff_dataset(
        base_url, deepfake_types, train_ratio
    )
    
    # Combine real and fake URLs
    train_list = train_real_url + train_fake_url 
    test_list = test_real_url + test_fake_url
    
    print(f"Train real images (before {augment_real_factor}x augmentation): {len(train_real_url)}")
    print(f"Test real images: {len(test_real_url)}")
    print(f"Train fake images: {len(train_fake_url)}")
    print(f"Test fake images: {len(test_fake_url)}")
    print(f"Using deepfake types: {deepfake_types}")
    print(f"Real image augmentation factor: {augment_real_factor}x")
    
    # Create transforms
    transform, real_transform = create_transforms(config)
    
    # Create datasets
    train_dataset = FaceDataset(
        train_list, 
        config,
        transform=transform, 
        real_transform=real_transform,
        is_training=True,
        augment_real_factor=augment_real_factor
    )
    
    test_dataset = FaceDataset(
        test_list, 
        config,
        transform=transform, 
        real_transform=transform,  # Use same transform for test
        is_training=False,
        augment_real_factor=1  # No augmentation for test
    )
    
    print(f"After {augment_real_factor}x augmentation - Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    return train_dataset, test_dataset
