# evaluation.py

import logging
import numpy as np
import torch
from torch import nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple
from config import Config
from models import StrawberryModel
from utils import load_data, evaluate_model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Evaluation:
    def __init__(self, config: Config):
        self.config = config
        self.model = StrawberryModel(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(self, data: Dict[str, List[Tuple[float, float]]]) -> Dict[str, float]:
        """
        Evaluate the model on the given data.

        Args:
            data (Dict[str, List[Tuple[float, float]]]): Data to evaluate the model on.

        Returns:
            Dict[str, float]: Evaluation metrics.
        """
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            predictions = []
            targets = []
            for batch in data["inputs"]:
                inputs = torch.tensor(batch, dtype=torch.float32).to(self.device)
                outputs = self.model(inputs)
                predictions.extend(outputs.cpu().numpy())
                targets.extend(data["targets"])
            predictions = np.array(predictions)
            targets = np.array(targets)
            metrics = {
                "mse": mean_squared_error(targets, predictions),
                "mae": mean_absolute_error(targets, predictions),
                "r2": r2_score(targets, predictions),
            }
            logger.info(f"Metrics: {metrics}")
            return metrics

    def evaluate_model(self, data: Dict[str, List[Tuple[float, float]]]) -> Dict[str, float]:
        """
        Evaluate the model on the given data and return the evaluation metrics.

        Args:
            data (Dict[str, List[Tuple[float, float]]]): Data to evaluate the model on.

        Returns:
            Dict[str, float]: Evaluation metrics.
        """
        metrics = self.evaluate(data)
        return metrics

def load_evaluation_data(config: Config) -> Dict[str, List[Tuple[float, float]]]:
    """
    Load evaluation data from the given configuration.

    Args:
        config (Config): Configuration to load data from.

    Returns:
        Dict[str, List[Tuple[float, float]]]: Evaluation data.
    """
    data = load_data(config)
    return data

def main():
    config = Config()
    evaluation = Evaluation(config)
    data = load_evaluation_data(config)
    metrics = evaluation.evaluate_model(data)
    logger.info(f"Final Metrics: {metrics}")

if __name__ == "__main__":
    main()