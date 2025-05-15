import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist_transform_with_padding():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Pad(padding=2),  # Pads 2 pixels on each side (28+2*2 = 32)
        transforms.Lambda(lambda x: torch.cat([x, torch.zeros(1, 32, 32), torch.zeros(1, 32, 32)], dim=0))
    ])

class MNISTDataset:
    def __init__(self):
        transform = get_mnist_transform_with_padding()
        # Load datasets
        self.train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        self.test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        # Create data loaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=64, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=1000, shuffle=False)
        
class FashionMNISTDataset:
    def __init__(self):
        transform = get_mnist_transform_with_padding()
        # Load datasets
        self.train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        self.test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        # Create data loaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=64, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=1000, shuffle=False)

class CIFAR10Dataset:
    def __init__(self, root: str = './data', 
                 train: bool = True, 
                 download: bool = True):
        self.root = root
        self.train = train
        self.download = download
        self.inp_shape = (32, 32, 3)
        
        # Transformation for the model
        self.transform_model = transforms.Compose([
            transforms.Resize((224, 224)),     # VGG16 expects 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean/std
                                std=[0.229, 0.224, 0.225]),
        ])
        # Transformation for the explainer
        self.transform_explainer = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        
        self.train_model = datasets.CIFAR10(root=self.root, 
                                        train=True, 
                                        download=True, 
                                        transform=self.transform_model)
        
        self.test_model = datasets.CIFAR10(root=self.root, 
                                        train=False, 
                                        download=True, 
                                        transform=self.transform_model)
        
        self.train_explainer = datasets.CIFAR10(root=self.root, 
                                        train=True, 
                                        download=True, 
                                        transform=self.transform_explainer)

        self.test_explainer = datasets.CIFAR10(root=self.root, 
                                        train=False, 
                                        download=True, 
                                        transform=self.transform_explainer)
        
        self.train_loader_model = DataLoader(self.train_model, batch_size=64, shuffle=True)
        self.test_loader_model = DataLoader(self.test_model, batch_size=1000, shuffle=False)
        
        self.train_loader_explainer = DataLoader(self.train_explainer, batch_size=64, shuffle=True)
        self.test_loader_explainer = DataLoader(self.test_explainer, batch_size=1000, shuffle=False)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
    
