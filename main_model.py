import logging
import os
import sys
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants and configuration
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
MODEL_DIR = PROJECT_ROOT / 'models'
CONFIG_FILE = PROJECT_ROOT / 'config.json'

# Data structures and models
@dataclass
class StrawberryData:
    image: np.ndarray
    mask: np.ndarray
    label: float

class StrawberryDataset(Dataset):
    def __init__(self, data_dir: Path, transform: Optional[transforms.Compose] = None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = []
        for file in data_dir.glob('*.jpg'):
            image = cv2.imread(str(file))
            mask = cv2.imread(str(file.parent / 'masks' / file.name), cv2.IMREAD_GRAYSCALE)
            label = float(file.name.split('_')[0])
            self.data.append(StrawberryData(image, mask, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data = self.data[index]
        if self.transform:
            data.image = self.transform(data.image)
        return data

class StrawberryModel(nn.Module):
    def __init__(self):
        super(StrawberryModel, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.fc = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor):
        x = self.resnet(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x

# Key functions
def load_data(data_dir: Path) -> StrawberryDataset:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return StrawberryDataset(data_dir, transform)

def train_model(model: StrawberryModel, data_loader: DataLoader, device: torch.device, epochs: int):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        for batch in data_loader:
            images = batch.image
            labels = batch.label
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = nn.MSELoss()(outputs, labels.view(-1, 1))
            loss.backward()
            optimizer.step()
        logger.info(f'Epoch {epoch+1}, Loss: {loss.item()}')

def evaluate_model(model: StrawberryModel, data_loader: DataLoader, device: torch.device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            images = batch.image
            labels = batch.label
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = nn.MSELoss()(outputs, labels.view(-1, 1))
            total_loss += loss.item()
    return total_loss / len(data_loader)

def main():
    # Load configuration
    config = {}
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)

    # Set up data and model
    data_dir = DATA_DIR
    model = StrawberryModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Load data
    data_loader = DataLoader(load_data(data_dir), batch_size=32, shuffle=True)

    # Train model
    train_model(model, data_loader, device, epochs=10)

    # Evaluate model
    loss = evaluate_model(model, data_loader, device)
    logger.info(f'Test Loss: {loss}')

if __name__ == '__main__':
    main()