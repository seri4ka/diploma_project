import torch
import torch.nn as nn
import torch.optim as optim

class MeanReturnLoss(nn.Module):
    """
    Custom loss function that incorporates the Sharpe ratio and a penalty for turnover.

    This loss function is designed to optimize financial predictions by balancing
    the return on investment with the costs associated with trading (turnover).

    Attributes:
        alpha (float): Weighting factor for the penalty terms.
        epsilon (float): Small value to avoid division by zero.
        weight_penalty_factor (float): Coefficient for weight penalty.
        target_turnover (float): Target turnover level to aim for.
        k (float): Coefficient for the exponential part of the turnover penalty.
        turnover_penalty_weight (float): Weight for the polynomial part of the turnover penalty.
    """

    def __init__(self, alpha=0.5, epsilon=1e-8, weight_penalty_factor=0.0001, target_turnover=0.2, k=50.0, turnover_penalty_weight=0.1):
        """
        Initializes the SharpeLossPenalty.

        Args:
            alpha (float): Weighting factor for the penalty terms.
            epsilon (float): Small value to avoid division by zero.
            weight_penalty_factor (float): Coefficient for weight penalty.
            target_turnover (float): Target turnover level to aim for.
            k (float): Coefficient for the exponential part of the turnover penalty.
            turnover_penalty_weight (float): Weight for the polynomial part of the turnover penalty.
        """
        super(MeanReturnLoss, self).__init__()
        self.alpha = alpha
        self.epsilon = epsilon  # To avoid division by zero
        self.weight_penalty_factor = weight_penalty_factor  # Penalty factor for weight size
        self.target_turnover = target_turnover  # Target turnover value
        self.k = k  # Coefficient for the exponential part
        self.turnover_penalty_weight = turnover_penalty_weight  # Weight for polynomial penalty

    def turnover_penalty(self, turnover_value):
        """
        Calculates the penalty for deviation from the target turnover.

        Args:
            turnover_value (float): The calculated turnover value.

        Returns:
            float: The computed penalty for the given turnover value.
        """
        # Penalize deviation from the target_turnover value
        penalty = (turnover_value - self.target_turnover) ** 2  # Quadratic penalty
        return penalty


    def forward(self, predictions, targets):
        """
        Computes the loss based on predictions and targets.

        This method normalizes the weights derived from predictions, calculates the mean
        weighted return, and computes the turnover penalty. Finally, it returns the loss
        as the negative mean return adjusted by the turnover penalty.

        Args:
            predictions (torch.Tensor): Predicted portfolio weights of shape (batch_size, num_assets).
            targets (torch.Tensor): Actual returns of assets of shape (batch_size, num_assets).

        Returns:
            torch.Tensor: The computed loss value.
        """
        # Normalize weights
        weights = predictions - predictions.mean()
        weights = weights.abs().sum() + self.epsilon
        weights = torch.where(weights != 0, predictions / weights, torch.zeros_like(predictions))

        # Calculate returns
        weighted_returns = weights * targets
        mean_weighted_return = weighted_returns.sum(axis=1).mean()

        # Calculate volatility (differentiable alternative to std())
        portfolio_returns = weighted_returns.sum(axis=1)
        volatility = torch.sqrt(((portfolio_returns - portfolio_returns.mean()) ** 2).mean() + self.epsilon)

        # Check for division by zero before calculating Sharpe Ratio
        mean_sharpe_ratio = torch.where(volatility != 0, mean_weighted_return / volatility, torch.zeros_like(mean_weighted_return))

        # Calculate turnover
        predictions_shifted = torch.cat((predictions[1:], predictions.new_zeros((1, predictions.size(1)))), dim=0)
        turnover_value = (predictions - predictions_shifted).abs().sum(dim=1).mean()

        # Calculate turnover penalty
        turnover_penalty = self.turnover_penalty(turnover_value)
        
        # Return loss with penalty, as well as sharpe and turnover for separate logging
        loss = -mean_weighted_return + turnover_penalty
        return loss, mean_sharpe_ratio.item(), turnover_value.item()
    
class SharpeLoss(nn.Module):
    """
    Custom loss function that incorporates the Sharpe ratio and a penalty for turnover.

    This loss function is designed to optimize financial predictions by balancing
    the return on investment with the costs associated with trading (turnover).

    Attributes:
        alpha (float): Weighting factor for the penalty terms.
        epsilon (float): Small value to avoid division by zero.
        weight_penalty_factor (float): Coefficient for weight penalty.
        target_turnover (float): Target turnover level to aim for.
        k (float): Coefficient for the exponential part of the turnover penalty.
        turnover_penalty_weight (float): Weight for the polynomial part of the turnover penalty.
    """

    def __init__(self, alpha=0.5, epsilon=1e-8, weight_penalty_factor=0.0001, target_turnover=0.2, k=50.0, turnover_penalty_weight=1):
        """
        Initializes the SharpeLossPenalty.

        Args:
            alpha (float): Weighting factor for the penalty terms.
            epsilon (float): Small value to avoid division by zero.
            weight_penalty_factor (float): Coefficient for weight penalty.
            target_turnover (float): Target turnover level to aim for.
            k (float): Coefficient for the exponential part of the turnover penalty.
            turnover_penalty_weight (float): Weight for the polynomial part of the turnover penalty.
        """
        super(SharpeLoss, self).__init__()
        self.alpha = alpha
        self.epsilon = epsilon  # To avoid division by zero
        self.weight_penalty_factor = weight_penalty_factor  # Penalty factor for weight size
        self.target_turnover = target_turnover  # Target turnover value
        self.k = k  # Coefficient for the exponential part
        self.turnover_penalty_weight = turnover_penalty_weight  # Weight for polynomial penalty

    def turnover_penalty(self, turnover_value):
        """
        Calculates the penalty for deviation from the target turnover.

        This method uses both exponential and polynomial components to create a smooth
        penalty function.

        Args:
            turnover_value (float): The calculated turnover value.

        Returns:
            float: The computed penalty for the given turnover value.
        """
        penalty = (1 / (turnover_value) ** 10)
        return penalty

    def forward(self, predictions, targets):
        """
        Computes the loss based on predictions and targets.

        This method normalizes the weights derived from predictions, calculates the mean
        weighted return, and computes the turnover penalty. Finally, it returns the loss
        as the negative mean return adjusted by the turnover penalty.

        Args:
            predictions (torch.Tensor): Predicted portfolio weights of shape (batch_size, num_assets).
            targets (torch.Tensor): Actual returns of assets of shape (batch_size, num_assets).

        Returns:
            torch.Tensor: The computed loss value.
        """
        # Normalize weights
        weights = predictions - predictions.mean()
        weights = weights.abs().sum() + self.epsilon
        weights = torch.where(weights != 0, predictions / weights, torch.zeros_like(predictions))

        # Calculate returns
        weighted_returns = weights * targets
        mean_weighted_return = weighted_returns.sum(axis=1).mean()

        # Calculate volatility (differentiable alternative to std())
        portfolio_returns = weighted_returns.sum(axis=1)
        volatility = torch.sqrt(((portfolio_returns - portfolio_returns.mean()) ** 2).mean() + self.epsilon)

        # Check for division by zero before calculating Sharpe Ratio
        mean_sharpe_ratio = torch.where(volatility != 0, mean_weighted_return / volatility, torch.zeros_like(mean_weighted_return))

        # Calculate turnover
        predictions_shifted = torch.cat((predictions[1:], predictions.new_zeros((1, predictions.size(1)))), dim=0)
        turnover_value = (predictions - predictions_shifted).abs().sum(dim=1).mean()
        
        # Calculate turnover penalty
        turnover_penalty = self.turnover_penalty(turnover_value)
        
        # Return loss with penalty
        return -(mean_weighted_return / volatility) + turnover_penalty
