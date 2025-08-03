# loss_functions.py
"""
Custom loss functions for the computer vision project.
"""

import logging
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
from torch import nn
from torch.nn import functional as F

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomLoss(nn.Module):
    """
    Base class for custom loss functions.
    """

    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

class MeanAbsoluteErrorLoss(CustomLoss):
    """
    Mean Absolute Error (MAE) loss function.
    """

    def __init__(self):
        super(MeanAbsoluteErrorLoss, self).__init__()

    def forward(self, predictions: nn.Tensor, targets: nn.Tensor) -> nn.Tensor:
        """
        Compute the Mean Absolute Error (MAE) loss.

        Args:
            predictions (nn.Tensor): Model predictions.
            targets (nn.Tensor): Ground truth targets.

        Returns:
            nn.Tensor: MAE loss value.
        """
        return F.l1_loss(predictions, targets)

class MeanSquaredErrorLoss(CustomLoss):
    """
    Mean Squared Error (MSE) loss function.
    """

    def __init__(self):
        super(MeanSquaredErrorLoss, self).__init__()

    def forward(self, predictions: nn.Tensor, targets: nn.Tensor) -> nn.Tensor:
        """
        Compute the Mean Squared Error (MSE) loss.

        Args:
            predictions (nn.Tensor): Model predictions.
            targets (nn.Tensor): Ground truth targets.

        Returns:
            nn.Tensor: MSE loss value.
        """
        return F.mse_loss(predictions, targets)

class BinaryCrossEntropyLoss(CustomLoss):
    """
    Binary Cross-Entropy (BCE) loss function.
    """

    def __init__(self):
        super(BinaryCrossEntropyLoss, self).__init__()

    def forward(self, predictions: nn.Tensor, targets: nn.Tensor) -> nn.Tensor:
        """
        Compute the Binary Cross-Entropy (BCE) loss.

        Args:
            predictions (nn.Tensor): Model predictions.
            targets (nn.Tensor): Ground truth targets.

        Returns:
            nn.Tensor: BCE loss value.
        """
        return F.binary_cross_entropy(predictions, targets)

class DiceLoss(CustomLoss):
    """
    Dice loss function.
    """

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, predictions: nn.Tensor, targets: nn.Tensor) -> nn.Tensor:
        """
        Compute the Dice loss.

        Args:
            predictions (nn.Tensor): Model predictions.
            targets (nn.Tensor): Ground truth targets.

        Returns:
            nn.Tensor: Dice loss value.
        """
        smooth = 1e-6
        intersection = (predictions * targets).sum(dim=[1, 2, 3])
        union = predictions.sum(dim=[1, 2, 3]) + targets.sum(dim=[1, 2, 3])
        loss = 1 - (2 * intersection + smooth) / (union + smooth)
        return loss.mean()

class VelocityThresholdLoss(CustomLoss):
    """
    Velocity Threshold loss function.
    """

    def __init__(self, velocity_threshold: float = 0.5):
        super(VelocityThresholdLoss, self).__init__()
        self.velocity_threshold = velocity_threshold

    def forward(self, predictions: nn.Tensor, targets: nn.Tensor) -> nn.Tensor:
        """
        Compute the Velocity Threshold loss.

        Args:
            predictions (nn.Tensor): Model predictions.
            targets (nn.Tensor): Ground truth targets.

        Returns:
            nn.Tensor: Velocity Threshold loss value.
        """
        velocity = np.abs(predictions - targets)
        loss = np.where(velocity > self.velocity_threshold, velocity, 0)
        return loss.mean()

class FlowTheoryLoss(CustomLoss):
    """
    Flow Theory loss function.
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5):
        super(FlowTheoryLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, predictions: nn.Tensor, targets: nn.Tensor) -> nn.Tensor:
        """
        Compute the Flow Theory loss.

        Args:
            predictions (nn.Tensor): Model predictions.
            targets (nn.Tensor): Ground truth targets.

        Returns:
            nn.Tensor: Flow Theory loss value.
        """
        loss = self.alpha * F.mse_loss(predictions, targets) + self.beta * F.l1_loss(predictions, targets)
        return loss

class HybridViewLoss(CustomLoss):
    """
    Hybrid View loss function.
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5):
        super(HybridViewLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, predictions: nn.Tensor, targets: nn.Tensor) -> nn.Tensor:
        """
        Compute the Hybrid View loss.

        Args:
            predictions (nn.Tensor): Model predictions.
            targets (nn.Tensor): Ground truth targets.

        Returns:
            nn.Tensor: Hybrid View loss value.
        """
        loss = self.alpha * F.mse_loss(predictions, targets) + self.beta * F.binary_cross_entropy(predictions, targets)
        return loss

class PolynomialRegressionLoss(CustomLoss):
    """
    Polynomial Regression loss function.
    """

    def __init__(self, degree: int = 2):
        super(PolynomialRegressionLoss, self).__init__()
        self.degree = degree

    def forward(self, predictions: nn.Tensor, targets: nn.Tensor) -> nn.Tensor:
        """
        Compute the Polynomial Regression loss.

        Args:
            predictions (nn.Tensor): Model predictions.
            targets (nn.Tensor): Ground truth targets.

        Returns:
            nn.Tensor: Polynomial Regression loss value.
        """
        loss = 0
        for i in range(self.degree + 1):
            loss += F.mse_loss(predictions ** i, targets ** i)
        return loss

# Define a dictionary to store the custom loss functions
custom_losses = {
    'mean_absolute_error': MeanAbsoluteErrorLoss(),
    'mean_squared_error': MeanSquaredErrorLoss(),
    'binary_cross_entropy': BinaryCrossEntropyLoss(),
    'dice': DiceLoss(),
    'velocity_threshold': VelocityThresholdLoss(),
    'flow_theory': FlowTheoryLoss(),
    'hybrid_view': HybridViewLoss(),
    'polynomial_regression': PolynomialRegressionLoss()
}

def get_loss_function(loss_name: str) -> CustomLoss:
    """
    Get a custom loss function by name.

    Args:
        loss_name (str): Name of the loss function.

    Returns:
        CustomLoss: Custom loss function instance.
    """
    if loss_name in custom_losses:
        return custom_losses[loss_name]
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")

if __name__ == "__main__":
    # Test the custom loss functions
    predictions = nn.Tensor([1, 2, 3])
    targets = nn.Tensor([4, 5, 6])

    loss_function = get_loss_function('mean_absolute_error')
    loss = loss_function(predictions, targets)
    print(f"Mean Absolute Error Loss: {loss.item()}")

    loss_function = get_loss_function('mean_squared_error')
    loss = loss_function(predictions, targets)
    print(f"Mean Squared Error Loss: {loss.item()}")

    loss_function = get_loss_function('binary_cross_entropy')
    loss = loss_function(predictions, targets)
    print(f"Binary Cross-Entropy Loss: {loss.item()}")

    loss_function = get_loss_function('dice')
    loss = loss_function(predictions, targets)
    print(f"Dice Loss: {loss.item()}")

    loss_function = get_loss_function('velocity_threshold')
    loss = loss_function(predictions, targets)
    print(f"Velocity Threshold Loss: {loss.item()}")

    loss_function = get_loss_function('flow_theory')
    loss = loss_function(predictions, targets)
    print(f"Flow Theory Loss: {loss.item()}")

    loss_function = get_loss_function('hybrid_view')
    loss = loss_function(predictions, targets)
    print(f"Hybrid View Loss: {loss.item()}")

    loss_function = get_loss_function('polynomial_regression')
    loss = loss_function(predictions, targets)
    print(f"Polynomial Regression Loss: {loss.item()}")