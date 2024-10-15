import torch
import torch.nn as nn
import torch.optim as optim


import torch.nn as nn

class LSTMModel(nn.Module):
    """
    LSTM model consisting of two LSTM layers followed by a fully connected layer.

    Attributes:
        lstm1: First LSTM layer.
        lstm2: Second LSTM layer.
        fc: Fully connected layer for output.
        dropout: Dropout layer to prevent overfitting.
        activation: ReLU activation function.
    """
    
    def __init__(self, input_size, hidden_size, hidden_size2, output_size, dropout=0, dropout2=0):
        """
        Initializes the LSTMModel.

        Args:
            input_size (int): Size of the input features.
            hidden_size (int): Size of the first LSTM hidden state.
            hidden_size2 (int): Size of the second LSTM hidden state.
            output_size (int): Size of the output layer.
            dropout (float): Dropout probability for the LSTM layers.
            dropout2 (float): Dropout probability for the output layer.
        """
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=dropout)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size2, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size2, output_size)
        self.dropout = nn.Dropout(p=dropout2)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor: Output tensor after LSTM processing and fully connected layer.
        """
        lstm_out1, _ = self.lstm1(x)
        lstm_out2, _ = self.lstm2(lstm_out1)
        
        # Apply Dropout
        lstm_out2 = self.dropout(lstm_out2)
        
        # Use the last time step
        lstm_out2 = lstm_out2[:, -1, :]
        
        # Apply fully connected layer and activation
        out = self.fc(lstm_out2)
        out = self.activation(out)
        
        return out


class LSTMWithAttention(nn.Module):
    """
    LSTM model with attention mechanism.

    Attributes:
        lstm1: First LSTM layer.
        lstm2: Second LSTM layer.
        attention: Multihead attention layer.
        fc: Fully connected layer for output.
        dropout: Dropout layer to prevent overfitting.
        activation: ReLU activation function.
    """
    
    def __init__(self, input_size, hidden_size, hidden_size2, output_size, dropout=0, dropout2=0):
        """
        Initializes the LSTMWithAttention model.

        Args:
            input_size (int): Size of the input features.
            hidden_size (int): Size of the first LSTM hidden state.
            hidden_size2 (int): Size of the second LSTM hidden state.
            output_size (int): Size of the output layer.
            dropout (float): Dropout probability for the LSTM layers.
            dropout2 (float): Dropout probability for the output layer.
        """
        super(LSTMWithAttention, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=dropout)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size2, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size2, output_size)
        self.attention = nn.MultiheadAttention(hidden_size2, num_heads=7)
        self.dropout = nn.Dropout(p=dropout2)
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the model with attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor: Output tensor after LSTM processing, attention, and fully connected layer.
        """
        lstm_out1, _ = self.lstm1(x)
        lstm_out2, _ = self.lstm2(lstm_out1)
        
        # Apply Attention
        lstm_out2, _ = self.attention(lstm_out2, lstm_out2, lstm_out2)
        
        lstm_out2 = self.dropout(lstm_out2)
        lstm_out2 = lstm_out2[:, -1, :]
        
        out = self.fc(lstm_out2)
        out = self.activation(out)
        
        return out


class LSTMWithAttentionAlt(nn.Module):
    """
    Alternative LSTM model with attention mechanism and layer normalization.

    Attributes:
        lstm1: First LSTM layer.
        lstm2: Second LSTM layer.
        attention: Multihead attention layer.
        fc: Fully connected layer for output.
        dropout: Dropout layer to prevent overfitting.
        layer_norm: Layer normalization layer.
        activation: ReLU activation function.
    """
    
    def __init__(self, input_size, hidden_size, hidden_size2, output_size, 
                 dropout=0, dropout2=0, num_heads=7):
        """
        Initializes the LSTMWithAttentionAlt model.

        Args:
            input_size (int): Size of the input features.
            hidden_size (int): Size of the first LSTM hidden state.
            hidden_size2 (int): Size of the second LSTM hidden state.
            output_size (int): Size of the output layer.
            dropout (float): Dropout probability for the LSTM layers.
            dropout2 (float): Dropout probability for the output layer.
            num_heads (int): Number of heads in the attention layer.
        """
        super(LSTMWithAttentionAlt, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=dropout)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size2, batch_first=True, dropout=dropout)
        self.attention = nn.MultiheadAttention(hidden_size2, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(hidden_size2, output_size)
        self.dropout = nn.Dropout(p=dropout2)
        self.layer_norm = nn.LayerNorm(hidden_size2)
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the model with attention and layer normalization.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor: Output tensor after LSTM processing, attention, normalization, and fully connected layer.
        """
        lstm_out1, _ = self.lstm1(x)
        lstm_out2, _ = self.lstm2(lstm_out1)
        
        # Apply Multihead Attention
        attn_out, _ = self.attention(lstm_out2, lstm_out2, lstm_out2)
        
        # Apply layer normalization
        attn_out = self.layer_norm(attn_out)
        
        # Apply Dropout
        attn_out = self.dropout(attn_out)
        
        # Use only the last time step
        attn_out = attn_out[:, -1, :]
        
        # Forward through the fully connected layer
        out = self.fc(attn_out)
        out = self.activation(out)
        
        return out


import torch
import torch.nn as nn

class ConvLSTMWithAttention(nn.Module):
    """
    Convolutional LSTM model with attention mechanism and various normalizations.
    """

    def __init__(self, input_size, hidden_size, hidden_size2, output_size, 
                 dropout=0, dropout2=0, num_heads=3, conv_out_channels=64):
        """
        Initializes the ConvLSTMWithAttention model.

        Args:
            input_size (int): Size of the input features.
            hidden_size (int): Size of the first LSTM hidden state.
            hidden_size2 (int): Size of the second LSTM hidden state.
            output_size (int): Size of the output layer.
            dropout (float): Dropout probability for the LSTM layers.
            dropout2 (float): Dropout probability for the output layer.
            num_heads (int): Number of heads in the attention layer.
            conv_out_channels (int): Number of output channels for the convolutional layers.
        """
        super(ConvLSTMWithAttention, self).__init__()
        
        # Convolutional layers with Batch Normalization
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=conv_out_channels, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm1d(conv_out_channels)  # Batch normalization after conv1
        self.conv2 = nn.Conv1d(in_channels=conv_out_channels, out_channels=conv_out_channels, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm1d(conv_out_channels)  # Batch normalization after conv2
        self.conv_dropout = nn.Dropout(p=dropout2)
        self.conv_activation = nn.ReLU()
        
        # LSTM layers
        self.lstm1 = nn.LSTM(conv_out_channels, hidden_size, batch_first=True, dropout=dropout)
        self.lstm_layer_norm = nn.LayerNorm(hidden_size)  # Layer normalization after lstm1
        self.lstm2 = nn.LSTM(hidden_size, hidden_size2, batch_first=True, dropout=dropout)
        
        # Attention layer
        self.attention = nn.MultiheadAttention(hidden_size2, num_heads=num_heads, batch_first=True)
        
        # Fully connected layer and additional layers
        self.fc = nn.Linear(hidden_size2, output_size)
        self.dropout = nn.Dropout(p=dropout2)
        self.layer_norm = nn.LayerNorm(hidden_size2)
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the convolutional LSTM model with attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor: Output tensor after convolutional, LSTM processing, attention, normalization, and fully connected layer.
        """
        # Apply convolutional layers
        conv_out = x.permute(0, 2, 1)  # Change shape to (batch_size, channels, seq_len)
        
        conv_out = self.conv1(conv_out)
        conv_out = self.batch_norm1(conv_out)  # Apply BatchNorm after conv1
        conv_out = self.conv_activation(conv_out)
        conv_out = self.conv_dropout(conv_out)
        
        conv_out = self.conv2(conv_out)
        conv_out = self.batch_norm2(conv_out)  # Apply BatchNorm after conv2
        conv_out = self.conv_activation(conv_out)
        conv_out = self.conv_dropout(conv_out)
        
        # Prepare data for LSTM
        conv_out = conv_out.permute(0, 2, 1)  # Change shape back to (batch_size, seq_len, channels)
        
        lstm_out1, _ = self.lstm1(conv_out)
        lstm_out1 = self.lstm_layer_norm(lstm_out1)  # Apply LayerNorm after lstm1
        lstm_out2, _ = self.lstm2(lstm_out1)
        
        # Residual connection between LSTM layers
        lstm_out2 = lstm_out1 + lstm_out2  # Residual connection
        
        # Apply Multihead Attention
        attn_out, _ = self.attention(lstm_out2, lstm_out2, lstm_out2)
        
        # Apply layer normalization
        attn_out = self.layer_norm(attn_out)
        
        # Apply Dropout
        attn_out = self.dropout(attn_out)
        
        # Use only the last time step
        attn_out = attn_out[:, -1, :]
        
        # Forward through the fully connected layer
        out = self.fc(attn_out)
        
        return out

