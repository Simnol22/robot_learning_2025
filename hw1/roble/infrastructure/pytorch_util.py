from typing import Union

import torch
from torch import nn

Activation = Union[str, nn.Module]

_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}
class MLP(nn.Module):
    def __init__(self, input_size, output_size, n_layers, activations, output_activation):
        super(MLP, self).__init__()
        layers = []
        in_dim = input_size

        # Add hidden layers
        for size, activation in zip(n_layers, activations):
            layers.append(nn.Linear(in_dim, size))
            layers.append(_str_to_activation[activation])
            in_dim = size
        
        # Add output layer
        layers.append(nn.Linear(in_dim, output_size))
        layers.append(output_activation)
        self.model = nn.Sequential(*layers)
        print("MLP model Succesfully created : ", self.model)

    def forward(self, x):
        return self.model(x)

def build_mlp(
        input_size: int,
        output_size: int,
        **kwargs
    ):
    """
    Builds a feedforward neural network

    arguments:
    n_layers: number of hidden layers
    size: dimension of each hidden layer
    activation: activation of each hidden layer
    input_size: size of the input layer
    output_size: size of the output layer
    output_activation: activation of the output layer

    returns:
        MLP (nn.Module)
    """
    try:
        params = kwargs["params"]
    except:
        params = kwargs
    if isinstance(params["output_activation"], str):
        output_activation = _str_to_activation[params["output_activation"]]

    return MLP(input_size, output_size, params["layer_sizes"], params["activations"], output_activation)

device = None

def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)

def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)

def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()
