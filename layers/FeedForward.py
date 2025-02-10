from torch.nn import Module, Linear, ReLU
from torch import Tensor

class FeedForward(Module):
    def __init__(this, d_model: int = 512, d_ff: int = 2048):
        super().__init__()
        this.d_model = d_model
        this.d_ff = d_ff
        this.first_layer = Linear(in_features=this.d_model, out_features=this.d_ff)
        this.relu = ReLU()
        this.second_layer = Linear(in_features=this.d_ff, out_features=this.d_model)
    
    def forward(this, input: Tensor) -> Tensor:
        return this.second_layer(this.relu(this.first_layer(input)))