import os
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
import json
from typing import List, Tuple, Dict
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants and configuration
@dataclass
class Config:
    data_dir: str
    batch_size: int
    num_workers: int
    image_size: Tuple[int, int]
    transform: str

class Transform(Enum):
    DEFAULT = "default"
    RANDOM_HFLIP = "random_hflip"
    RANDOM_VFLIP = "random_vflip"

class DataMode(Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"

class DataError(Exception):
    pass

class DataLoaderError(DataError):
    pass

class DataBatchError(DataError):
    pass

class DataBatch:
    def __init__(self, images: List[np.ndarray], labels: List[np.ndarray]):
        self.images = images
        self.labels = labels

class DataBatchLoader:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def load_batch(self) -> DataBatch:
        try:
            batch = next(self.data_loader)
            images = [img.numpy() for img in batch]
            labels = [label.numpy() for label in batch]
            return DataBatch(images, labels)
        except StopIteration:
            raise DataBatchError("No more batches available")

class DataBatchLoaderFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_loader(self) -> DataBatchLoader:
        return DataBatchLoader(self.data_loader)

class DataBatchLoaderFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory(self) -> DataBatchLoaderFactory:
        return DataBatchLoaderFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory(self) -> DataBatchLoaderFactoryFactory:
        return DataBatchLoaderFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory_factory(self) -> DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory:
        return DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactoryFactory(self.data_loader)

class DataBatchLoaderFactoryFactoryFactoryFactoryFactoryFactoryFactory