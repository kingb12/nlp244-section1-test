# need to sub-class nn.Module
from torch import nn, Tensor


class MyLinear(nn.Module):

    # need to implement initialization:
    # what parameters do we need to accept? What fields do we use them to define?
    def __init__(self):
        super().__init__()
        pass

    # calculate our output y = wx + b and returnn
    def forward(self, x: Tensor) -> Tensor:
        pass
