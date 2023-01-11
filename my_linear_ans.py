# need to sub-class nn.Module
from torch import nn, Tensor
import torch


class MyLinear(nn.Module):

    weight: Tensor
    bias: Tensor

    # need to implement initialization:
    # what parameters do we need to accept? What fields do we use them to define?
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = torch.randn(out_dim, in_dim)  # MM (5 rows, 4 columns) * (1 column (len=4)) => 1 column (len=5))
        self.bias = torch.randn((out_dim,))

    def forward(self, x: Tensor) -> Tensor:
        return torch.matmul(self.weight, x) + self.bias


if __name__ == '__main__':
    my_linear = MyLinear(4, 5)
    real_linear = nn.Linear(4, 5)
    # surgery for equivalent weight and bias
    my_linear.weight, my_linear.bias = real_linear.weight, real_linear.bias
    x = torch.randn(4,)
    assert torch.equal(my_linear(x), real_linear(x))