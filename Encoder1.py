#imports for layers

#import torch.optim as optim
#from torch.utils.data import Dataset, DataLoader
#import torchvision.datasets as Datasets
#import torchvision.transforms as transforms
#import torch.nn.functional as F
#import torchvision.models as models
#import torchvision.utils as vutils

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import os
import random
import numpy as np
import math
#from IPython.display import clear_output
#import matplotlib.pyplot as plt
#from PIL import Image
#from tqdm.notebook import trange, tqdm


##imports for model
import tqdm
import pathlib
from datetime import timedelta
from sklearn.model_selection import train_test_split
import gc

from typing import List, Optional
from numpy.typing import NDArray

from .model_core import StellarPerceptronCore
from .layers import NonLinearEmbedding, StellarPerceptronTorchModel
from .nn_utils import TrainingGenerator, robust_mean_squared_error, mean_squared_error
from .torch_utils_collect_env import get_torch_env_info

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist


print('check 1 imports working')


#tokeniser start wiht numpy take a spectra from low res and high res and convert to f lamba (lamba f lamba instead of f nu) and make the tokenizer
#look at tokenisation step in omni spec paper, should only need numpy






#this section is for the layers
#this is taken from spectra fm model as i could not find the activation details of omni spec
def get_activation(activation):
    if activation is None:
        return F.relu
    elif isinstance(activation, str):
        return getattr(F, activation)
    else:
        return activation
##start with positional encoding? this may need to be updated to be more so similar to the omni spec model

def default_initialization(torch_layer: nn.Module):
     #need to fill this in specifically for omni spec

def positional_encoding_spectral(self, input_pixel_wavelength, d_model):
        batch_size = input_pixel_wavelength.shape[0]
        seq_length = input_pixel_wavelength.shape[1]
        pe = torch.zeros(batch_size, seq_length, d_model, **self.factory_kwargs)
        is_zero = input_pixel_wavelength == 0
        min_wavelength =  # put min wavelength from jwst
        max_wavelength =  # put max wavelength from jwst
        #do we want to proceed with this normalisation step?
        normalized_wavelengths = (input_pixel_wavelength - min_wavelength) / (max_wavelength - min_wavelength)

        #this isnt finished 


class TransformerBlock(nn.Module):
        

class StellarPerceptronEncoder(nn.Module):
    def __init__



#this section is for the model

class StellarPerceptron(StellarPerceptronCore): #from pytorch
     