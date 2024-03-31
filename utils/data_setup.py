'''
Creates dataloaders for PyTorch 
'''

import os

from torchvision import datasets, transforms 
from torch.utils.data import DataLoader 

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
        train_dir: str, 
        test_dir: str, 
        transform: transforms.Compose,  
        batch_size: int, 
        num_workers: int=NUM_WORKERS
        ): 
        '''
        Creates training and testing DataLoaders 

        Takes a train and test directory path and creates PyTorch Datasets 
        and then turns them into PyTorch dataloaders to use in train and test loops

        Args: 
                train_dir: Path to training data
                test_dir: Path to test data 
                transform: Transform to apply to data
                batch_size: Number of samples in each batch of the data
                num_workers: Number of workers per DataLoader 

        Returns: 
                A tuple of (train_dataloader, test_dataloader, class_names)
                class_names: list of class names of images

        '''
        
        train_data = datasets.ImageFolder(train_dir, transform=transform)
        test_data = datasets.ImageFolder(test_dir, transform=transform) 

        class_names = train_data.classes

        # Turn images into data loaders
        train_dataloader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        test_dataloader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        return train_dataloader, test_dataloader, class_names





