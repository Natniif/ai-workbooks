'''
Script for training machine learning model 
'''

import os 
import torch 
import data_setup, engine, model_builder, utils

import argparse
from dataclasses import dataclass

from torchvision import transforms

@dataclass
class ModelConfig: 
    pass



if __name__ == '__main__': 

    parser = argparse.ArgumentParser(description='Model tester')

