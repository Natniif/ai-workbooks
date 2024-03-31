'''
Utilities for model saving 
'''

import torch 
from pathlib import Path 

import torchvision
import matplotlib.pyplot as plt

def save_model(model: torch.nn.Module, 
               target_dir: str, 
               model_name: str): 
    '''
    Saves a PyTorch model to a target directory 

    Args: 
        model: PyTorch model to save 
        target_dir: Directory to save model to 
        model_name: Name of model for file to be named. 
            Must have either ".pth" or ".pt" as the file extension
    '''

    # Create target directory 
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, 
                          exist_ok=True)

    # create model save path 
    assert model_name.endswith('.pth') or model_name.endswith('.pt'), "model needs to end with .pth or .pt"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
             f=model_save_path)

def image_look_torchvision(dataset: torchvision.datasets,
        index: int): 
    '''
    function to look at a particular image in the dataset. 
    Mainly for curiously looking in dataset. 
    Must be a torchvision dataset
    
    Args: 
        dataset: dataset with images you want to inspect
        index: index in dataset of image you want to look at 

    example usage: 
        image_look_torchvision(train, 200)
    '''
    image, label = dataset[index] 
    print(f'Image:{label},/n size: {image.shape}')
    
    image = image.cpu().numpy().transpose((1, 2, 0))
    image = (image / 2.0) + 0.5 

    plt.imshow(image) 
    plt.show()




