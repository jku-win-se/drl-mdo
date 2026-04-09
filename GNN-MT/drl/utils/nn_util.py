import torch.nn as nn
import math


def init_layer(layer: nn.Module, gain=math.sqrt(2), bias=0) -> nn.Module:
    """
    Initialize a hidden linear layer with orthogonal weights and zero bias.
    Gain depends on activation function.

    Args:
        layer (nn.Module): The linear layer to initialize.
        activation_fn (str): Activation function ('relu' or 'tanh').

    Returns:
        nn.Module: Initialized layer.
    """
    assert isinstance(layer, nn.Linear), "Layer must be nn.Linear"

    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, bias)
    return layer


def init_rnn_orthogonal(rnn_layer: nn.Module) -> nn.Module:
    """
    Apply orthogonal initialization to all weights in an LSTM or GRU layer.
    Biases are zero-initialized.

    Args:
        rnn_layer (nn.Module): An instance of nn.LSTM or nn.GRU.

    Returns:
        nn.Module: Initialized RNN layer.
    """
    assert isinstance(rnn_layer, (nn.LSTM, nn.GRU)), "Must be nn.LSTM or nn.GRU"

    for name, param in rnn_layer.named_parameters():
        if "weight_ih" in name:
            # Input-hidden weights: orthogonal per gate
            hidden_size = param.shape[1]
            num_gates = param.shape[0] // hidden_size
            for i in range(num_gates):
                nn.init.orthogonal_(param.data[i * hidden_size : (i + 1) * hidden_size])
        elif "weight_hh" in name:
            # Hidden-hidden weights: orthogonal per gate
            hidden_size = param.shape[1]
            num_gates = param.shape[0] // hidden_size
            for i in range(num_gates):
                nn.init.orthogonal_(param.data[i * hidden_size : (i + 1) * hidden_size])
        elif "bias" in name:
            nn.init.constant_(param.data, 0)
            if isinstance(rnn_layer, nn.LSTM):
                hidden_size = param.shape[0] // 4
                param.data[hidden_size : hidden_size * 2] = 1.0
    return rnn_layer
