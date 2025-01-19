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
        Initializes the MeanReturnLoss.

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
        self.epsilon = epsilon
        self.weight_penalty_factor = weight_penalty_factor
        self.target_turnover = target_turnover
        self.k = k
        self.turnover_penalty_weight = turnover_penalty_weight

    def turnover_penalty(self, turnover_value):
        """
        Calculates the penalty for deviation from the target turnover.

        Args:
            turnover_value (float): The calculated turnover value.

        Returns:
            float: The computed penalty for the given turnover value.
        """
        return 1 / (self.target_turnover - turnover_value) ** 4 * self.turnover_penalty_weight

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
            float: Sharpe ratio value for logging purposes.
            float: Turnover penalty for logging purposes.
        """
        # Normalize weights row-wise for better balance
        weights = predictions - predictions.mean(dim=1, keepdim=True)
        weights_sum = weights.abs().sum(dim=1, keepdim=True) + self.epsilon
        weights = torch.where(weights_sum != 0, weights / weights_sum, torch.zeros_like(weights))

        # Calculate mean weighted return
        weighted_returns = weights * targets
        mean_weighted_return = weighted_returns.sum(axis=1).mean()

        # Calculate turnover
        predictions_shifted = torch.cat((predictions[1:], predictions.new_zeros((1, predictions.size(1)))), dim=0)
        turnover_value = (predictions - predictions_shifted).abs().sum(dim=1).mean()

        # Calculate turnover penalty
        turnover_penalty = self.turnover_penalty(turnover_value)

        # Calculate volatility (differentiable alternative to std())
        # portfolio_returns = weighted_returns.sum(axis=1)
        # volatility = torch.sqrt(((portfolio_returns - portfolio_returns.mean()) ** 2).mean() + self.epsilon)

        # New Sharpe ratio with log to stabilize turnover impact
        sharpe_ratio = mean_weighted_return / turnover_value

        # Return loss with penalty, as well as sharpe and turnover for separate logging
        loss = -sharpe_ratio + turnover_penalty
        return loss, sharpe_ratio.item(), turnover_penalty.item()
    
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
        penalty = (1 / (turnover_value) ** 10) / 100
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


class AltMeanReturnLoss(nn.Module):
    """
    Custom loss function that balances return, turnover, and volatility.
    Designed to encourage active trading strategies by penalizing static portfolios.
    """

    def __init__(self, alpha=0.5, epsilon=1e-8, weight_penalty_factor=0.0001, 
                 min_turnover=0.05, turnover_weight=2.0, volatility_weight=0.1):
        """
        Initializes the DynamicPortfolioLoss with additional parameters for stronger turnover penalty.

        Args:
            alpha (float): Weighting factor for the penalty terms.
            epsilon (float): Small value to avoid division by zero.
            weight_penalty_factor (float): Coefficient for weight penalty.
            min_turnover (float): Minimum desired turnover level to avoid static portfolios.
            turnover_weight (float): Coefficient to strengthen the turnover penalty.
            volatility_weight (float): Coefficient for volatility penalty.
        """
        super(AltMeanReturnLoss, self).__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.weight_penalty_factor = weight_penalty_factor
        self.min_turnover = min_turnover
        self.turnover_weight = turnover_weight
        self.volatility_weight = volatility_weight

    def turnover_penalty(self, turnover_value):
        """
        Calculates a penalty that encourages turnover to exceed a minimum threshold.

        Args:
            turnover_value (float): The calculated turnover value.

        Returns:
            float: The computed penalty encouraging higher turnover.
        """
        penalty = torch.exp(-self.turnover_weight * (turnover_value - self.min_turnover))
        return penalty

    def forward(self, predictions, targets):
        """
        Computes the loss based on predictions and targets, including penalties for low turnover and high volatility.

        Args:
            predictions (torch.Tensor): Predicted portfolio weights of shape (batch_size, num_assets).
            targets (torch.Tensor): Actual returns of assets of shape (batch_size, num_assets).

        Returns:
            torch.Tensor: The computed loss value.
            float: Mean weighted return for logging.
            float: Turnover penalty for logging.
            float: Volatility penalty for logging.
        """
        # Normalize weights to sum to 1
        weights_sum = predictions.abs().sum(dim=1, keepdim=True) + self.epsilon
        weights = torch.where(weights_sum != 0, predictions / weights_sum, torch.zeros_like(predictions))

        # Calculate mean weighted return
        weighted_returns = weights * targets
        mean_weighted_return = weighted_returns.sum(axis=1).mean()

        # Calculate turnover by summing absolute changes in positions over time
        predictions_shifted = torch.cat((predictions[1:], predictions.new_zeros((1, predictions.size(1)))), dim=0)
        turnover_value = (predictions - predictions_shifted).abs().sum(dim=1).mean()

        # Turnover penalty increases if turnover is below the desired minimum threshold
        turnover_penalty = self.turnover_penalty(turnover_value)

        # Calculate volatility as a smooth alternative to standard deviation
        portfolio_returns = weighted_returns.sum(axis=1)
        volatility = torch.sqrt(((portfolio_returns - portfolio_returns.mean()) ** 2).mean() + self.epsilon)
        volatility_penalty = volatility * self.volatility_weight

        # Calculate final loss, encouraging higher turnover and penalizing high volatility
        loss = -mean_weighted_return + turnover_penalty + volatility_penalty

        return loss, mean_weighted_return.item(), turnover_penalty.item(), volatility_penalty.item()


class DynamicPortfolioLoss(nn.Module):
    def __init__(self, alpha=0.5, epsilon=1e-8, min_turnover=0.001, turnover_weight=1000, volatility_weight=0.1, l1_weight=0.01):
        super(DynamicPortfolioLoss, self).__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.min_turnover = min_turnover
        self.turnover_weight = turnover_weight
        self.volatility_weight = volatility_weight
        self.l1_weight = l1_weight

    def turnover_penalty(self, turnover_value):
        return 1 / (turnover_value + self.epsilon + 0.8) ** 2

    def forward(self, predictions, targets):
        # Нормализация весов
        weights_sum = predictions.abs().sum(dim=1, keepdim=True) + self.epsilon
        weights = torch.where(weights_sum != 0, predictions / weights_sum, torch.zeros_like(predictions))

        # Средняя взвешенная доходность
        weighted_returns = weights * targets
        mean_weighted_return = weighted_returns.sum(axis=1).mean()

        # Оборот
        predictions_shifted = torch.cat((predictions[1:], predictions.new_zeros((1, predictions.size(1)))), dim=0)
        turnover_value = (predictions - predictions_shifted).abs().sum(dim=1).mean()

        # Штраф за низкий оборот
        turnover_penalty = self.turnover_penalty(turnover_value)

        # Волатильность
        # portfolio_returns = weighted_returns.sum(axis=1)
        # volatility = torch.sqrt(((portfolio_returns - portfolio_returns.mean()) ** 2).mean() + self.epsilon)
        # volatility_penalty = volatility * self.volatility_weight

        # L1-регуляризация на изменения позиций
        # l1_penalty = self.l1_weight * (predictions - predictions_shifted).abs().sum()

        # Потери
        loss = -mean_weighted_return / turnover_penalty / turnover_value
        
        return loss, mean_weighted_return.item(), turnover_penalty.item()
