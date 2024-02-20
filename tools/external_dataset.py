# data_loader.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(root_path, batch_size=32, num_workers=2):
    """
    Creates and returns dataloaders for the melanoma cancer dataset with data augmentations for the training set.

    Parameters:
    - root_path (str): The root directory containing the 'train' and 'test' folders.
    - batch_size (int): The size of the batches.
    - num_workers (int): The number of subprocesses to use for data loading.

    Returns:
    - train_loader, test_loader: DataLoaders for the training and testing datasets.
    """
    # Augmentations for the training data
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=15),  # Rotate images by up to 15 degrees
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Scale and crop images
        transforms.RandomHorizontalFlip(),  # Flip images horizontally
        transforms.RandomVerticalFlip(),  # Flip images vertically (optional)
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust color settings
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Standard transformations for the test data
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(root=f'{root_path}/train', transform=train_transforms)
    test_dataset = datasets.ImageFolder(root=f'{root_path}/test', transform=test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
