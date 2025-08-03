import logging
import numpy as np
import cv2
import torch
from typing import Tuple, List
from PIL import Image
from torchvision import transforms
from config import Config
from utils import load_config, get_logger

class Preprocessor:
    """
    Image preprocessing utilities
    """

    def __init__(self, config: Config):
        """
        Initialize the Preprocessor with the given configuration

        Args:
            config (Config): The configuration object
        """
        self.config = config
        self.logger = get_logger(__name__)

    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load an image from the given path

        Args:
            image_path (str): The path to the image

        Returns:
            np.ndarray: The loaded image
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"Failed to load image from {image_path}")
                raise FileNotFoundError
            return image
        except Exception as e:
            self.logger.error(f"Failed to load image from {image_path}: {str(e)}")
            raise

    def resize_image(self, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """
        Resize the image to the given size

        Args:
            image (np.ndarray): The image to resize
            size (Tuple[int, int]): The desired size

        Returns:
            np.ndarray: The resized image
        """
        try:
            resized_image = cv2.resize(image, size)
            return resized_image
        except Exception as e:
            self.logger.error(f"Failed to resize image: {str(e)}")
            raise

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize the image to the range [0, 1]

        Args:
            image (np.ndarray): The image to normalize

        Returns:
            np.ndarray: The normalized image
        """
        try:
            normalized_image = image / 255.0
            return normalized_image
        except Exception as e:
            self.logger.error(f"Failed to normalize image: {str(e)}")
            raise

    def apply_transforms(self, image: np.ndarray) -> torch.Tensor:
        """
        Apply the specified transforms to the image

        Args:
            image (np.ndarray): The image to transform

        Returns:
            torch.Tensor: The transformed image
        """
        try:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            transformed_image = transform(image)
            return transformed_image
        except Exception as e:
            self.logger.error(f"Failed to apply transforms: {str(e)}")
            raise

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess the image at the given path

        Args:
            image_path (str): The path to the image

        Returns:
            torch.Tensor: The preprocessed image
        """
        try:
            image = self.load_image(image_path)
            resized_image = self.resize_image(image, self.config.image_size)
            normalized_image = self.normalize_image(resized_image)
            transformed_image = self.apply_transforms(normalized_image)
            return transformed_image
        except Exception as e:
            self.logger.error(f"Failed to preprocess image: {str(e)}")
            raise


class StrawberryPreprocessor(Preprocessor):
    """
    Preprocessor for strawberry images
    """

    def __init__(self, config: Config):
        """
        Initialize the StrawberryPreprocessor with the given configuration

        Args:
            config (Config): The configuration object
        """
        super().__init__(config)

    def detect_strawberries(self, image: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect strawberries in the given image

        Args:
            image (np.ndarray): The image to detect strawberries in

        Returns:
            List[Tuple[int, int]]: The detected strawberry locations
        """
        try:
            # Implement strawberry detection algorithm here
            # For now, just return a dummy list of locations
            return [(10, 10), (20, 20), (30, 30)]
        except Exception as e:
            self.logger.error(f"Failed to detect strawberries: {str(e)}")
            raise


def main():
    config = load_config()
    preprocessor = Preprocessor(config)
    image_path = "path/to/image.jpg"
    preprocessed_image = preprocessor.preprocess_image(image_path)
    print(preprocessed_image.shape)


if __name__ == "__main__":
    main()