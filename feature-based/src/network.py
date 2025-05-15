import random
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import vgg16, VGG16_Weights

class VGGModel(object):
    def __init__(self,
                 num_classes:int=10, 
                 device:str="cpu"):
        self.num_classes = int(num_classes)
        model = vgg16(weights=VGG16_Weights.DEFAULT)
        for param in model.features.parameters():
            param.requires_grad = False  # Freeze feature extractor

        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, self.num_classes)
        model = model.to(device)
        self.model = model
        
    def __call__(self, x):
        return self.model(x)
    
    def forward(self, x):
        return self.model(x)
