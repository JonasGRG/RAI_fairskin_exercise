# data_loader.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import albumentations

def get_data_loaders(root_path, image_size=224, batch_size=32, num_workers=2):
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
    transforms_train = albumentations.Compose([
        albumentations.Transpose(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.RandomBrightness(limit=0.2, p=0.75),
        albumentations.RandomContrast(limit=0.2, p=0.75),
        albumentations.OneOf([
            albumentations.MotionBlur(blur_limit=5),
            albumentations.MedianBlur(blur_limit=5),
            albumentations.GaussianBlur(blur_limit=5),
            albumentations.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=0.7),

        albumentations.OneOf([
            albumentations.OpticalDistortion(distort_limit=1.0),
            albumentations.GridDistortion(num_steps=5, distort_limit=1.),
            albumentations.ElasticTransform(alpha=3),
        ], p=0.7),

        albumentations.CLAHE(clip_limit=4.0, p=0.7),
        albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
        albumentations.Resize(image_size, image_size),
        albumentations.Cutout(max_h_size=int(image_size * 0.375), max_w_size=int(image_size * 0.375), num_holes=1, p=0.7),
        albumentations.Normalize()
    ])

    transforms_val = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize()
    ])

    train_dataset = datasets.ImageFolder(root=f'{root_path}/train', transform=transforms_train)
    test_dataset = datasets.ImageFolder(root=f'{root_path}/test', transform=transforms_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
