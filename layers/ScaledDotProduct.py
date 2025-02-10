from torch.nn import Module, Softmax, Linear
from torch import Tensor, matmul
from math import sqrt
from typing import Tuple

class ScaledDotProduct(Module):
    
    def __init__(this, d_model: int=512, d_k:int=64, d_v=64 ):
        super().__init__()
        this.d_model: int = d_model
        this.d_k: int = d_k
        this.d_v: int = d_v
        this.fn_softmax = Softmax(dim=-1)

        this.queriesEmbedding = Linear(in_features=this.d_model, out_features=d_k) # Came from the input embedding, 512. d_k is for the attention input
        this.keysEmbedding = Linear(in_features=this.d_model, out_features=d_k)
        this.valuesEmbedding = Linear(in_features=this.d_model, out_features=d_v)

    def forward(this, queries: Tensor, keys: Tensor, values: Tensor) -> Tuple[Tensor, Tensor]:
        """
            Queries, Keys, Values: [Batch size, Seq Length, d_k]
        """
        Q: Tensor =  this.queriesEmbedding(queries)
        K: Tensor = this.keysEmbedding(keys)
        V: Tensor = this.valuesEmbedding(values)

        numerator: Tensor = matmul(Q, K.transpose(-2, -1))
        denominator: Tensor = sqrt(this.d_k)
        attention_weights: Tensor =  this.fn_softmax(numerator / denominator)
        outputs: Tensor = matmul(attention_weights, V)
        return attention_weights, outputs
    