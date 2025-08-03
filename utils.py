import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image

from config import Config

logger = logging.getLogger(__name__)

# Configure temporary file storage
TEMP_DIR = tempfile.gettempdir()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger.setLevel(Config.LOG_LEVEL)


class Utils:
    @classmethod
    def validate_input(cls, data: np.ndarray, input_name: str) -> None:
        """
        Validate the input data array.

        Parameters:
        data (np.ndarray): The input data array to validate.
        input_name (str): The name of the input variable.

        Raises:
        ValueError: If the data array is not 3-dimensional or has invalid data type.
        """
        if data.ndim != 3:
            raise ValueError(f"{input_name} must be a 3-dimensional array, but received shape {data.shape}.")
        if data.dtype != np.uint8:
            raise ValueError(f"{input_name} must have data type np.uint8, but received {data.dtype}.")

    @classmethod
    def load_image(cls, image_path: str) -> np.ndarray:
        """
        Load an image from the file path and return it as a numpy array.

        Parameters:
        image_path (str): The file path of the image to load.

        Returns:
        np.ndarray: The loaded image as a numpy array.
        """
        try:
            image = Image.open(image_path)
            return np.array(image)
        except IOError as e:
            logger.error(f"Error loading image from path '{image_path}': {e}")
            raise ValueError(f"Failed to load image from path '{image_path}'.") from e

    @classmethod
    def save_image(cls, image: np.ndarray, output_path: str) -> None:
        """
        Save the numpy array as an image to the specified output path.

        Parameters:
        image (np.ndarray): The image data to save.
        output_path (str): The output file path to save the image.

        Raises:
        ValueError: If the image data has invalid dimensions or data type.
        """
        if image.ndim not in (2, 3):
            raise ValueError("Invalid image dimensions. Image must be 2D or 3D array.")
        if image.dtype != np.uint8:
            raise ValueError("Invalid image data type. Image data must be np.uint8.")

        try:
            image_pil = Image.fromarray(image)
            image_pil.save(output_path)
        except IOError as e:
            logger.error(f"Error saving image to path '{output_path}': {e}")
            raise ValueError(f"Failed to save image to path '{output_path}'.") from e

    @classmethod
    def create_directory(cls, directory: str) -> None:
        """
        Create a new directory if it doesn't exist.

        Parameters:
        directory (str): The path of the directory to create.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

    @classmethod
    def get_temp_filename(cls, extension: str = "") -> str:
        """
        Generate a unique temporary file name with an optional extension.

        Parameters:
        extension (str): The optional file extension (including the dot).

        Returns:
        str: The unique temporary file name.
        """
        return os.path.join(TEMP_DIR, next(tempfile._get_candidate_names()) + extension)

    @classmethod
    def initialize_device(cls) -> torch.device:
        """
        Initialize the device (CPU or GPU) for PyTorch.

        Returns:
        torch.device: The device to use for PyTorch computations.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        return device

    @classmethod
    def move_to_device(cls, data: Any, device: torch.device) -> Any:
        """
        Move the data to the specified device.

        Parameters:
        data (Any): The data to move.
        device (torch.device): The device to move the data to.

        Returns:
        Any: The data on the specified device.
        """
        return data.to(device) if torch.is_tensor(data) else data

    @classmethod
    def setup_seed(cls, seed: int = 42) -> None:
        """
        Set the random seed for reproducibility.

        Parameters:
        seed (int): The random seed to use. Defaults to 42.
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        logger.info(f"Random seed set to {seed}.")

    @classmethod
    def compute_iou(cls, box1: List[int], box2: List[int]) -> float:
        """
        Compute the Intersection over Union (IoU) between two bounding boxes.

        Parameters:
        box1 (List[int]): The first bounding box in format [x1, y1, x2, y2].
        box2 (List[int]): The second bounding box in format [x1, y1, x2, y2].

        Returns:
        float: The IoU value between the two bounding boxes.
        """
        # Calculate the coordinates of the intersection box
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # Calculate the area of the intersection box
        intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

        # Calculate the area of each box
        box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

        # Compute the IoU
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        return iou

    @classmethod
    def compute_center(cls, box: List[int]) -> Tuple[float, float]:
        """
        Compute the center coordinates of a bounding box.

        Parameters:
        box (List[int]): The bounding box in format [x1, y1, x2, y2].

        Returns:
        Tuple[float, float]: The center coordinates (x, y) of the bounding box.
        """
        x = (box[0] + box[2]) / 2
        y = (box[1] + box[3]) / 2
        return x, y

    # ... More utility functions ...


class Config:
    """
    Configuration class to store project settings.
    """

    LOG_LEVEL = logging.INFO
    SEED = 42
    DEVICE = Utils.initialize_device()


class ImageProcessor:
    """
    Class to process and manipulate images.
    """

    def __init__(self, image_path: str):
        self.image_path = image_path
        self.image = Utils.load_image(image_path)

    def save_processed_image(self, output_path: str) -> None:
        """
        Save the processed image to the specified output path.

        Parameters:
        output_path (str): The output file path to save the processed image.
        """
        Utils.save_image(self.image, output_path)


# Example usage
if __name__ == "__main__":
    # Load and process image
    image_path = "input.jpg"
    output_path = "output.jpg"
    image_processor = ImageProcessor(image_path)
    image_processor.save_processed_image(output_path)

    # Validate input data
    data = np.random.rand(10, 20, 3)
    Utils.validate_input(data, "input_data")

    # Get temporary file name
    temp_file = Utils.get_temp_filename(".jpg")
    logger.info(f"Temporary file name: {temp_file}")