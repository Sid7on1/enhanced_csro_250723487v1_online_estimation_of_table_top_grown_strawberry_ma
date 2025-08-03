# config.py
"""
Model configuration file for the computer_vision project.

This file contains the configuration settings for the model, including
hyperparameters, data paths, and other relevant settings.
"""

import logging
import os
from typing import Dict, List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# Define configuration classes
class StrawberryConfig:
    """
    Configuration class for the Strawberry model.

    Attributes:
        batch_size (int): Batch size for training.
        num_epochs (int): Number of epochs for training.
        learning_rate (float): Learning rate for the optimizer.
        momentum (float): Momentum for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
    """

    def __init__(self):
        self.batch_size = 32
        self.num_epochs = 10
        self.learning_rate = 0.001
        self.momentum = 0.9
        self.weight_decay = 0.0005

    def __str__(self):
        return f"StrawberryConfig(batch_size={self.batch_size}, num_epochs={self.num_epochs}, learning_rate={self.learning_rate}, momentum={self.momentum}, weight_decay={self.weight_decay})"

class GanConfig:
    """
    Configuration class for the GAN model.

    Attributes:
        batch_size (int): Batch size for training.
        num_epochs (int): Number of epochs for training.
        learning_rate (float): Learning rate for the generator and discriminator.
        momentum (float): Momentum for the generator and discriminator.
        weight_decay (float): Weight decay for the generator and discriminator.
    """

    def __init__(self):
        self.batch_size = 32
        self.num_epochs = 10
        self.learning_rate = 0.001
        self.momentum = 0.9
        self.weight_decay = 0.0005

    def __str__(self):
        return f"GanConfig(batch_size={self.batch_size}, num_epochs={self.num_epochs}, learning_rate={self.learning_rate}, momentum={self.momentum}, weight_decay={self.weight_decay})"

class CycleGANConfig:
    """
    Configuration class for the CycleGAN model.

    Attributes:
        batch_size (int): Batch size for training.
        num_epochs (int): Number of epochs for training.
        learning_rate (float): Learning rate for the generator and discriminator.
        momentum (float): Momentum for the generator and discriminator.
        weight_decay (float): Weight decay for the generator and discriminator.
    """

    def __init__(self):
        self.batch_size = 32
        self.num_epochs = 10
        self.learning_rate = 0.001
        self.momentum = 0.9
        self.weight_decay = 0.0005

    def __str__(self):
        return f"CycleGANConfig(batch_size={self.batch_size}, num_epochs={self.num_epochs}, learning_rate={self.learning_rate}, momentum={self.momentum}, weight_decay={self.weight_decay})"

class PolynomialRegressionConfig:
    """
    Configuration class for the polynomial regression model.

    Attributes:
        degree (int): Degree of the polynomial.
        learning_rate (float): Learning rate for the optimizer.
        momentum (float): Momentum for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
    """

    def __init__(self):
        self.degree = 3
        self.learning_rate = 0.001
        self.momentum = 0.9
        self.weight_decay = 0.0005

    def __str__(self):
        return f"PolynomialRegressionConfig(degree={self.degree}, learning_rate={self.learning_rate}, momentum={self.momentum}, weight_decay={self.weight_decay})"

class Config:
    """
    Main configuration class.

    Attributes:
        strawberry_config (StrawberryConfig): Configuration for the Strawberry model.
        gan_config (GanConfig): Configuration for the GAN model.
        cycle_gan_config (CycleGANConfig): Configuration for the CycleGAN model.
        polynomial_regression_config (PolynomialRegressionConfig): Configuration for the polynomial regression model.
    """

    def __init__(self):
        self.strawberry_config = StrawberryConfig()
        self.gan_config = GanConfig()
        self.cycle_gan_config = CycleGANConfig()
        self.polynomial_regression_config = PolynomialRegressionConfig()

    def __str__(self):
        return f"Config(strawberry_config={self.strawberry_config}, gan_config={self.gan_config}, cycle_gan_config={self.cycle_gan_config}, polynomial_regression_config={self.polynomial_regression_config})"

# Create a singleton instance of the Config class
config = Config()

# Define a function to load the configuration from a file
def load_config(file_path: str) -> Config:
    """
    Load the configuration from a file.

    Args:
        file_path (str): Path to the configuration file.

    Returns:
        Config: Loaded configuration instance.
    """
    try:
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
            config = Config()
            config.strawberry_config.batch_size = config_dict['strawberry_config']['batch_size']
            config.strawberry_config.num_epochs = config_dict['strawberry_config']['num_epochs']
            config.strawberry_config.learning_rate = config_dict['strawberry_config']['learning_rate']
            config.strawberry_config.momentum = config_dict['strawberry_config']['momentum']
            config.strawberry_config.weight_decay = config_dict['strawberry_config']['weight_decay']
            config.gan_config.batch_size = config_dict['gan_config']['batch_size']
            config.gan_config.num_epochs = config_dict['gan_config']['num_epochs']
            config.gan_config.learning_rate = config_dict['gan_config']['learning_rate']
            config.gan_config.momentum = config_dict['gan_config']['momentum']
            config.gan_config.weight_decay = config_dict['gan_config']['weight_decay']
            config.cycle_gan_config.batch_size = config_dict['cycle_gan_config']['batch_size']
            config.cycle_gan_config.num_epochs = config_dict['cycle_gan_config']['num_epochs']
            config.cycle_gan_config.learning_rate = config_dict['cycle_gan_config']['learning_rate']
            config.cycle_gan_config.momentum = config_dict['cycle_gan_config']['momentum']
            config.cycle_gan_config.weight_decay = config_dict['cycle_gan_config']['weight_decay']
            config.polynomial_regression_config.degree = config_dict['polynomial_regression_config']['degree']
            config.polynomial_regression_config.learning_rate = config_dict['polynomial_regression_config']['learning_rate']
            config.polynomial_regression_config.momentum = config_dict['polynomial_regression_config']['momentum']
            config.polynomial_regression_config.weight_decay = config_dict['polynomial_regression_config']['weight_decay']
            return config
    except Exception as e:
        logger.error(f"Failed to load configuration from file: {e}")
        return None

# Define a function to save the configuration to a file
def save_config(file_path: str) -> None:
    """
    Save the configuration to a file.

    Args:
        file_path (str): Path to the configuration file.
    """
    try:
        config_dict = {
            'strawberry_config': {
                'batch_size': config.strawberry_config.batch_size,
                'num_epochs': config.strawberry_config.num_epochs,
                'learning_rate': config.strawberry_config.learning_rate,
                'momentum': config.strawberry_config.momentum,
                'weight_decay': config.strawberry_config.weight_decay
            },
            'gan_config': {
                'batch_size': config.gan_config.batch_size,
                'num_epochs': config.gan_config.num_epochs,
                'learning_rate': config.gan_config.learning_rate,
                'momentum': config.gan_config.momentum,
                'weight_decay': config.gan_config.weight_decay
            },
            'cycle_gan_config': {
                'batch_size': config.cycle_gan_config.batch_size,
                'num_epochs': config.cycle_gan_config.num_epochs,
                'learning_rate': config.cycle_gan_config.learning_rate,
                'momentum': config.cycle_gan_config.momentum,
                'weight_decay': config.cycle_gan_config.weight_decay
            },
            'polynomial_regression_config': {
                'degree': config.polynomial_regression_config.degree,
                'learning_rate': config.polynomial_regression_config.learning_rate,
                'momentum': config.polynomial_regression_config.momentum,
                'weight_decay': config.polynomial_regression_config.weight_decay
            }
        }
        with open(file_path, 'w') as f:
            json.dump(config_dict, f)
    except Exception as e:
        logger.error(f"Failed to save configuration to file: {e}")

# Define a function to print the configuration
def print_config() -> None:
    """
    Print the configuration.
    """
    logger.info(f"StrawberryConfig: {config.strawberry_config}")
    logger.info(f"GanConfig: {config.gan_config}")
    logger.info(f"CycleGANConfig: {config.cycle_gan_config}")
    logger.info(f"PolynomialRegressionConfig: {config.polynomial_regression_config}")

# Define a function to validate the configuration
def validate_config() -> bool:
    """
    Validate the configuration.

    Returns:
        bool: True if the configuration is valid, False otherwise.
    """
    try:
        # Validate the Strawberry configuration
        if config.strawberry_config.batch_size <= 0:
            logger.error("Invalid batch size for Strawberry configuration")
            return False
        if config.strawberry_config.num_epochs <= 0:
            logger.error("Invalid number of epochs for Strawberry configuration")
            return False
        if config.strawberry_config.learning_rate <= 0:
            logger.error("Invalid learning rate for Strawberry configuration")
            return False
        if config.strawberry_config.momentum <= 0:
            logger.error("Invalid momentum for Strawberry configuration")
            return False
        if config.strawberry_config.weight_decay <= 0:
            logger.error("Invalid weight decay for Strawberry configuration")
            return False

        # Validate the GAN configuration
        if config.gan_config.batch_size <= 0:
            logger.error("Invalid batch size for GAN configuration")
            return False
        if config.gan_config.num_epochs <= 0:
            logger.error("Invalid number of epochs for GAN configuration")
            return False
        if config.gan_config.learning_rate <= 0:
            logger.error("Invalid learning rate for GAN configuration")
            return False
        if config.gan_config.momentum <= 0:
            logger.error("Invalid momentum for GAN configuration")
            return False
        if config.gan_config.weight_decay <= 0:
            logger.error("Invalid weight decay for GAN configuration")
            return False

        # Validate the CycleGAN configuration
        if config.cycle_gan_config.batch_size <= 0:
            logger.error("Invalid batch size for CycleGAN configuration")
            return False
        if config.cycle_gan_config.num_epochs <= 0:
            logger.error("Invalid number of epochs for CycleGAN configuration")
            return False
        if config.cycle_gan_config.learning_rate <= 0:
            logger.error("Invalid learning rate for CycleGAN configuration")
            return False
        if config.cycle_gan_config.momentum <= 0:
            logger.error("Invalid momentum for CycleGAN configuration")
            return False
        if config.cycle_gan_config.weight_decay <= 0:
            logger.error("Invalid weight decay for CycleGAN configuration")
            return False

        # Validate the polynomial regression configuration
        if config.polynomial_regression_config.degree <= 0:
            logger.error("Invalid degree for polynomial regression configuration")
            return False
        if config.polynomial_regression_config.learning_rate <= 0:
            logger.error("Invalid learning rate for polynomial regression configuration")
            return False
        if config.polynomial_regression_config.momentum <= 0:
            logger.error("Invalid momentum for polynomial regression configuration")
            return False
        if config.polynomial_regression_config.weight_decay <= 0:
            logger.error("Invalid weight decay for polynomial regression configuration")
            return False

        # If all configurations are valid, return True
        return True
    except Exception as e:
        logger.error(f"Failed to validate configuration: {e}")
        return False

# Define a function to get the configuration
def get_config() -> Config:
    """
    Get the configuration.

    Returns:
        Config: Configuration instance.
    """
    return config

# Define a function to set the configuration
def set_config(config: Config) -> None:
    """
    Set the configuration.

    Args:
        config (Config): Configuration instance.
    """
    global config
    config = config

# Define a function to load the default configuration
def load_default_config() -> Config:
    """
    Load the default configuration.

    Returns:
        Config: Default configuration instance.
    """
    return Config()

# Define a function to save the default configuration
def save_default_config(file_path: str) -> None:
    """
    Save the default configuration to a file.

    Args:
        file_path (str): Path to the configuration file.
    """
    save_config(file_path)

# Define a function to print the default configuration
def print_default_config() -> None:
    """
    Print the default configuration.
    """
    print_config()

# Define a function to validate the default configuration
def validate_default_config() -> bool:
    """
    Validate the default configuration.

    Returns:
        bool: True if the default configuration is valid, False otherwise.
    """
    return validate_config()

# Define a function to get the default configuration
def get_default_config() -> Config:
    """
    Get the default configuration.

    Returns:
        Config: Default configuration instance.
    """
    return load_default_config()

# Define a function to set the default configuration
def set_default_config(config: Config) -> None:
    """
    Set the default configuration.

    Args:
        config (Config): Configuration instance.
    """
    set_config(config)

# Define a function to load the configuration from a JSON file
def load_config_from_json(file_path: str) -> Config:
    """
    Load the configuration from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        Config: Loaded configuration instance.
    """
    try:
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
            config = Config()
            config.strawberry_config.batch_size = config_dict['strawberry_config']['batch_size']
            config.strawberry_config.num_epochs = config_dict['strawberry_config']['num_epochs']
            config.strawberry_config.learning_rate = config_dict['strawberry_config']['learning_rate']
            config.strawberry_config.momentum = config_dict['strawberry_config']['momentum']
            config.strawberry_config.weight_decay = config_dict['strawberry_config']['weight_decay']
            config.gan_config.batch_size = config_dict['gan_config']['batch_size']
            config.gan_config.num_epochs = config_dict['gan_config']['num_epochs']
            config.gan_config.learning_rate = config_dict['gan_config']['learning_rate']
            config.gan_config.momentum = config_dict['gan_config']['momentum']
            config.gan_config.weight_decay = config_dict['gan_config']['weight_decay']
            config.cycle_gan_config.batch_size = config_dict['cycle_gan_config']['batch_size']
            config.cycle_gan_config.num_epochs = config_dict['cycle_gan_config']['num_epochs']
            config.cycle_gan_config.learning_rate = config_dict['cycle_gan_config']['learning_rate']
            config.cycle_gan_config.momentum = config_dict['cycle_gan_config']['momentum']
            config.cycle_gan_config.weight_decay = config_dict['cycle_gan_config']['weight_decay']
            config.polynomial_regression_config.degree = config_dict['polynomial_regression_config']['degree']
            config.polynomial_regression_config.learning_rate = config_dict['polynomial_regression_config']['learning_rate']
            config.polynomial_regression_config.momentum = config_dict['polynomial_regression_config']['momentum']
            config.polynomial_regression_config.weight_decay = config_dict['polynomial_regression_config']['weight_decay']
            return config
    except Exception as e:
        logger.error(f"Failed to load configuration from JSON file: {e}")
        return None

# Define a function to save the configuration to a JSON file
def save_config_to_json(file_path: str) -> None:
    """
    Save the configuration to a JSON file.

    Args:
        file_path (str): Path to the JSON file.
    """
    try:
        config_dict = {
            'strawberry_config': {
                'batch_size': config.strawberry_config.batch_size,
                'num_epochs': config.strawberry_config.num_epochs,
                'learning_rate': config.strawberry_config.learning_rate,
                'momentum': config.strawberry_config.momentum,
                'weight_decay': config.strawberry_config.weight_decay
            },
            'gan_config': {
                'batch_size': config.gan_config.batch_size,
                'num_epochs': config.gan_config.num_epochs,
                'learning_rate': config.gan_config.learning_rate,
                'momentum': config.gan_config.momentum,
                'weight_decay': config.gan_config.weight_decay
            },
            'cycle_gan_config': {
                'batch_size': config.cycle_gan_config.batch_size,
                'num_epochs': config.cycle_gan_config.num_epochs,
                'learning_rate': config.cycle_gan_config.learning_rate,
                'momentum': config.cycle_gan_config.momentum,
                'weight_decay': config.cycle_gan_config.weight_decay
            },
            'polynomial_regression_config': {
                'degree': config.polynomial_regression_config.degree,
                'learning_rate': config.polynomial_regression_config.learning_rate,
                'momentum': config.polynomial_regression_config.momentum,
                'weight_decay': config.polynomial_regression_config.weight_decay
            }
        }
        with open(file_path, 'w') as f:
            json.dump(config_dict, f)
    except Exception as e:
        logger.error(f"Failed to save configuration to JSON file: {e}")

# Define a function to print the configuration to a JSON file
def print_config_to_json(file_path: str) -> None:
    """
    Print the configuration to a JSON file.

    Args:
        file_path (str): Path to the JSON file.
    """
    try:
        config_dict = {
            'strawberry_config': {
                'batch_size': config.strawberry_config.batch_size,
                'num_epochs': config.strawberry_config.num_epochs,
                'learning_rate': config.strawberry_config.learning_rate,
                'momentum': config.strawberry_config.momentum,
                'weight_decay': config.strawberry_config.weight_decay
            },
            'gan_config': {
                'batch_size': config.gan_config.batch_size,
                'num_epochs': config.gan_config.num_epochs,
                'learning_rate': config.gan_config.learning_rate,
                'momentum': config.gan_config.momentum,
                'weight_decay': config.gan_config.weight_decay
            },
            'cycle_gan_config': {
                'batch_size': config.cycle_gan_config.batch_size,
                'num_epochs': config.cycle_gan_config.num_epochs,
                'learning_rate': config.cycle_gan_config.learning_rate,
                'momentum': config.cycle_gan_config.momentum,
                'weight_decay': config.cycle_gan_config.weight_decay
            },
            'polynomial_regression_config': {
                'degree': config.polynomial_regression_config.degree,
                'learning_rate': config.polynomial_regression_config.learning_rate,
                'momentum': config.polynomial_regression_config.momentum,
                'weight_decay': config.polynomial_regression_config.weight_decay
            }
        }
        with open(file_path, 'w') as f:
            json.dump(config_dict, f)
    except Exception as e:
        logger.error(f"Failed to print configuration to JSON file: {e}")

# Define a function to validate the configuration from a JSON file
def validate_config_from_json(file_path: str) -> bool:
    """
    Validate the configuration from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        bool: True if the configuration is valid, False otherwise.
    """
    try:
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
            config = Config()
            config.strawberry_config.batch_size = config_dict['strawberry_config']['batch_size']
            config.strawberry_config.num_epochs = config_dict['strawberry_config']['num_epochs']
            config.strawberry_config.learning_rate = config_dict['strawberry_config']['learning_rate']
            config.strawberry_config.momentum = config_dict['strawberry_config']['momentum']
            config.strawberry_config.weight_decay = config_dict['strawberry_config']['weight_decay']
            config.gan_config.batch_size = config_dict['gan_config']['batch_size']
            config.gan_config.num_epochs = config_dict['gan_config']['num_epochs']
            config.gan_config.learning_rate = config_dict['gan_config']['learning_rate']
            config.gan_config.momentum = config_dict['gan_config']['momentum']
            config.gan_config.weight_decay = config_dict['gan_config']['weight_decay']
            config.cycle_gan_config.batch_size = config_dict['cycle_gan_config']['batch_size']
            config.cycle_gan_config.num_epochs = config_dict['cycle_gan_config']['num_epochs']
            config.cycle_gan_config.learning_rate = config_dict['cycle_gan_config']['learning_rate']
            config.cycle_gan_config.momentum = config_dict['cycle_gan_config']['momentum']
            config.cycle_gan_config.weight_decay = config_dict['cycle_gan_config']['weight_decay']
            config.polynomial_regression_config.degree = config_dict['polynomial_regression_config']['degree']
            config.polynomial_regression_config.learning_rate = config_dict['polynomial_regression_config']['learning_rate']
            config.polynomial_regression_config.momentum = config_dict['polynomial_regression_config']['momentum']
            config.polynomial_regression_config.weight_decay = config_dict['polynomial_regression_config']['weight_decay']
            return validate_config()
    except Exception as e:
        logger.error(f"Failed to validate configuration from JSON file: {e}")
        return False

# Define a function to get the configuration from a JSON file
def get_config_from_json(file_path: str) -> Config:
    """
    Get the configuration from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        Config: Loaded configuration instance.
    """
    try:
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
            config = Config()
            config.strawberry_config.batch_size = config_dict['strawberry_config']['batch_size']
            config.strawberry_config.num_epochs = config_dict['strawberry_config']['num_epochs']
            config.strawberry_config.learning_rate = config_dict['strawberry_config']['learning_rate']
            config.strawberry_config.momentum = config_dict['strawberry_config']['momentum']
            config.strawberry_config.weight_decay = config_dict['strawberry_config']['weight_decay']
            config.gan_config.batch_size = config_dict['gan_config']['batch_size']
            config.gan_config.num_epochs = config_dict['gan_config']['num_epochs']
            config.gan_config.learning_rate = config_dict['gan_config']['learning_rate']
            config.gan_config.momentum = config_dict['gan_config']['momentum']
            config.gan_config.weight_decay = config_dict['gan_config']['weight_decay']
            config.cycle_gan_config.batch_size = config_dict['cycle_gan_config']['batch_size']
            config.cycle_gan_config.num_epochs = config_dict['cycle_gan_config']['num_epochs']
            config.cycle_gan_config.learning_rate = config_dict['cycle_gan_config']['learning_rate']
            config.cycle_gan_config.momentum = config_dict['cycle_gan_config']['momentum']
            config.cycle_gan_config.weight_decay = config_dict['cycle_gan_config']['weight_decay']
            config.polynomial_regression_config.degree = config_dict['polynomial_regression_config']['degree']
            config.polynomial_regression_config.learning_rate = config_dict['polynomial_regression_config']['learning_rate']
            config.polynomial_regression_config.momentum = config_dict['polynomial_regression_config']['momentum']
            config.polynomial_regression_config.weight_decay = config_dict['polynomial_regression_config']['weight_decay']
            return config
    except Exception as e:
        logger.error(f"Failed to get configuration from JSON file: {e}")
        return None

# Define a function to set the configuration from a JSON file
def set_config_from_json(file_path: str) -> None:
    """
    Set the configuration from a JSON file.

    Args:
        file_path (str): Path to the JSON file.
    """
    global config
    config = get_config_from_json(file_path)

# Define a function to load the default configuration from a JSON file
def load_default_config_from_json(file_path: str) -> Config:
    """
    Load the default configuration from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        Config: Loaded default configuration instance.
    """
    return get_config_from_json(file_path)

# Define a function to save the default configuration to a JSON file
def save_default_config_to_json(file_path: str) -> None:
    """
    Save the default configuration to a JSON file.

    Args:
        file_path (str): Path to the JSON file.
    """
    save_config_to_json(file_path)

# Define a function to print the default configuration to a JSON file
def print_default_config_to_json(file_path: str) -> None:
    """
    Print the default configuration to a JSON file.

    Args:
        file_path (str): Path to the JSON file.
    """
    print_config_to_json(file_path)

# Define a function to validate the default configuration from a JSON file
def validate_default_config_from_json(file_path: str) -> bool:
    """
    Validate the default configuration from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        bool: True if the default configuration is valid, False otherwise.
    """
    return validate_config_from_json(file_path)

# Define a function to get the default configuration from a JSON file
def get_default_config_from_json(file_path: str) -> Config:
    """
    Get the default configuration from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        Config: Loaded default configuration instance.
    """
    return get_config_from_json(file_path)

# Define a function to set the default configuration from a JSON file
def set_default_config_from_json(file_path: str) -> None:
    """
    Set the default configuration from a JSON file.

    Args:
        file_path (str): Path to the JSON file.
    """
    set_config_from_json(file_path)

# Define a function to load the configuration from a YAML file
def load_config_from_yaml(file_path: str) -> Config:
    """
    Load the configuration from a YAML file.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        Config: Loaded configuration instance.
    """
    try:
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            config = Config()
            config.strawberry_config.batch_size = config_dict['strawberry_config']['batch_size']
            config.strawberry_config.num_epochs = config_dict['strawberry_config']['num_epochs']
            config.strawberry_config.learning_rate = config_dict['strawberry_config']['learning_rate']
            config.strawberry_config.momentum = config_dict['strawberry_config']['momentum']
            config.strawberry_config.weight_decay = config_dict['strawberry_config']['weight_decay']
            config.gan_config.batch_size = config_dict['gan_config']['batch_size']
            config.gan_config.num_epochs = config_dict['gan_config']['num_epochs']
            config.gan_config.learning_rate = config_dict['gan_config']['learning_rate']
            config.gan_config.momentum = config_dict['gan_config']['momentum']
            config.gan_config.weight_decay = config_dict['gan_config']['weight_decay']
            config.cycle_gan_config.batch_size = config_dict['cycle_gan_config']['batch_size']
            config.cycle_gan_config.num_epochs = config_dict['cycle_gan_config']['num_epochs']
            config.cycle_gan_config.learning_rate = config_dict['cycle_gan_config']['learning_rate']
            config.cycle_gan_config.momentum = config_dict['cycle_gan_config']['momentum']
            config.cycle_gan_config.weight_decay = config_dict['cycle_gan_config']['weight_decay']
            config.polynomial_regression_config.degree = config_dict['polynomial_regression_config']['degree']
            config.polynomial_regression_config.learning_rate = config_dict['polynomial_regression_config']['learning_rate']
            config.polynomial_regression_config.momentum = config_dict['polynomial_regression_config']['momentum']
            config.polynomial_regression_config.weight_decay = config_dict['polynomial_regression_config']['weight_decay']
            return config
    except Exception as e:
        logger.error(f"Failed to load configuration from YAML file: {e}")
        return None

# Define a function to save the configuration to a YAML file
def save_config_to_yaml(file_path: str) -> None:
    """
    Save the configuration to a YAML file.

    Args:
        file_path (str): Path to the YAML file.
    """
    try:
        config_dict = {
            'strawberry_config': {
                'batch_size': config.strawberry_config.batch_size,
                'num_epochs': config.strawberry_config.num_epochs,
                'learning_rate': config.strawberry_config.learning_rate,
                'momentum': config.strawberry_config.momentum,
                'weight_decay': config.strawberry_config.weight_decay
            },
            'gan_config': {
                'batch_size': config.gan_config.batch_size,
                'num_epochs': config.gan_config.num_epochs,
                'learning_rate': config.gan_config.learning_rate,
                'momentum': config.gan_config.momentum,
                'weight_decay': config.gan_config.weight_decay
            },
            'cycle_gan_config': {
                'batch_size': config.cycle_gan_config.batch_size,
                'num_epochs': config.cycle_gan_config.num_epochs,
                'learning_rate': config.cycle_gan_config.learning_rate,
                'momentum': config.cycle_gan_config.momentum,
                'weight_decay': config.cycle_gan_config.weight_decay
            },
            'polynomial_regression_config': {
                'degree': config.polynomial_regression_config.degree,
                'learning_rate': config.polynomial_regression_config.learning_rate,
                'momentum': config.polynomial_regression_config.momentum,
                'weight_decay': config.polynomial_regression_config.weight_decay
            }
        }
        with open(file_path, 'w') as f:
            yaml.dump(config_dict, f)
    except Exception as e:
        logger.error(f"Failed to save configuration to YAML file: {e}")

# Define a function to print the configuration to a YAML file
def print_config_to_yaml(file_path: str) -> None:
    """
    Print the configuration to a YAML file.

    Args:
        file_path (str): Path to the YAML file.
    """
    try:
        config_dict = {
            'strawberry_config': {
                'batch_size': config.strawberry_config.batch_size,
                'num_epochs': config.strawberry_config.num_epochs,
                'learning_rate': config.strawberry_config.learning_rate,
                'momentum': config.strawberry_config.momentum,
                'weight_decay': config.strawberry_config.weight_decay
            },
            'gan_config': {
                'batch_size': config.gan_config.batch_size,
                'num_epochs': config.gan_config.num_epochs,
                'learning_rate': config.gan_config.learning_rate,
                'momentum': config.gan_config.momentum,
                'weight_decay': config.gan_config.weight_decay
            },
            'cycle_gan_config': {
                'batch_size': config.cycle_gan_config.batch_size,
                'num_epochs': config.cycle_gan_config.num_epochs,
                'learning_rate': config.cycle_gan_config.learning_rate,
                'momentum': config.cycle_gan_config.momentum,
                'weight_decay': config.cycle_gan_config.weight_decay
            },
            'polynomial_regression_config': {
                'degree': config.polynomial_regression_config.degree,
                'learning_rate': config.polynomial_regression_config.learning_rate,
                'momentum': config.polynomial_regression_config.momentum,
                'weight_decay': config.polynomial_regression_config.weight_decay
            }
        }
        with open(file_path, 'w') as f:
            yaml.dump(config_dict, f)
    except Exception as e:
        logger.error(f"Failed to print configuration to YAML file: {e}")

# Define a function to validate the configuration from a YAML file
def validate_config_from_yaml(file_path: str) -> bool:
    """
    Validate the configuration from a YAML file.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        bool: True if the configuration is valid, False otherwise.
    """
    try:
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            config = Config()
            config.strawberry_config.batch_size = config_dict['strawberry_config']['batch_size']
            config.strawberry_config.num_epochs = config_dict['strawberry_config']['num_epochs']
            config.strawberry_config.learning_rate = config_dict['strawberry_config']['learning_rate']
            config.strawberry_config.momentum = config_dict['strawberry_config']['momentum']
            config.strawberry_config.weight_decay = config_dict['strawberry_config']['weight_decay']
            config.gan_config.batch_size = config_dict['gan_config']['batch_size']
            config.gan_config.num_epochs = config_dict['gan_config']['num_epochs']
            config.gan_config.learning_rate = config_dict['gan_config']['learning_rate']
            config.gan_config.momentum = config_dict['gan_config']['momentum']
            config.gan_config.weight_decay = config_dict['gan_config']['weight_decay']
            config.cycle_gan_config.batch_size = config_dict['cycle_gan_config']['batch_size']
            config.cycle_gan_config.num_epochs = config_dict['cycle_gan_config']['num_epochs']
            config.cycle_gan_config.learning_rate = config_dict['cycle_gan_config']['learning_rate']
            config.cycle_gan_config.momentum = config_dict['cycle_gan_config']['momentum']
            config.cycle_gan_config.weight_decay = config_dict['cycle_gan_config']['weight_decay']
            config.polynomial_regression_config.degree = config_dict['polynomial_regression_config']['degree']
            config.polynomial_regression_config.learning_rate = config_dict['polynomial_regression_config']['learning_rate']
            config.polynomial_regression_config.momentum = config_dict['polynomial_regression_config']['momentum']
            config.polynomial_regression_config.weight_decay = config_dict['polynomial_regression_config']['weight_decay']
            return validate_config()
    except Exception as e:
        logger.error(f"Failed to validate configuration from YAML file: {e}")
        return False

# Define a function to get the configuration from a YAML file
def get_config_from_yaml(file_path: str) -> Config:
    """
    Get the configuration from a YAML file.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        Config: Loaded configuration instance.
    """
    try:
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            config = Config()
            config.strawberry_config.batch_size = config_dict['strawberry_config']['batch_size']
            config.strawberry_config.num_epochs = config_dict['strawberry_config']['num_epochs']
            config.strawberry_config.learning_rate = config_dict['strawberry_config']['learning_rate']
            config.strawberry_config.momentum = config_dict['strawberry_config']['momentum']
            config.strawberry_config.weight_decay = config_dict['strawberry_config']['weight_decay']
            config.gan_config.batch_size = config_dict['gan_config']['batch_size']
            config.gan_config.num_epochs = config_dict['gan_config']['num_epochs']
            config.gan_config.learning_rate = config_dict['gan_config']['learning_rate']
            config.gan_config.momentum = config_dict['gan_config']['momentum']
            config.gan_config.weight_decay = config_dict['gan_config']['weight_decay']
            config.cycle_gan_config.batch_size = config_dict['cycle_gan_config']['batch_size']
            config.cycle_gan_config.num_epochs = config_dict['cycle_gan_config']['num_epochs']
            config.cycle_gan_config.learning_rate = config_dict['cycle_gan_config']['learning_rate']
            config.cycle_gan_config.momentum = config_dict['cycle_gan_config']['momentum']
            config.cycle_gan_config.weight_decay = config_dict['cycle_gan_config']['weight_decay']
            config.polynomial_regression_config.degree = config_dict['polynomial_regression_config']['degree']
            config.polynomial_regression_config.learning_rate = config_dict['polynomial_regression_config']['learning_rate']
            config.polynomial_regression_config.momentum = config_dict['polynomial_regression_config']['momentum']
            config.polynomial_regression_config.weight_decay = config_dict['polynomial_regression_config']['weight_decay']
            return config
    except Exception as e:
        logger.error(f"Failed to get configuration from YAML file: {e}")
        return None

# Define a function to set the configuration from a YAML file
def set_config_from_yaml(file_path: str) -> None:
    """
    Set the configuration from a YAML file.

    Args:
        file_path (str):