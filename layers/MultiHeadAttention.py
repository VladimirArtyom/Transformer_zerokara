import sys
sys.path.insert(1, "..")
from torch.nn import Module, ModuleList, Linear
from torch import Tensor, concat
from layers.ScaledDotProduct import ScaledDotProduct
from typing import List

class MultiHeadAttention(Module):
    """
        Multi head attention from Vaswani et.al
    """
    def __init__(this, d_k: int = 64, d_v: int = 64, h: int = 8):
        super().__init__()
        this.d_k: int = d_k
        this.d_v: int = d_v
        this.h: int = h
        this.d_model: int = this.h * this.d_v
        this.head_attention_layers = ModuleList()
        this.w_o = Linear(in_features=this.d_v * this.h, out_features=this.d_model)

        for _ in range(0, this.h):
            new_scaled_attention = ScaledDotProduct(this.d_model, d_k=this.d_k, d_v=this.d_v)
            this.head_attention_layers.append(
                new_scaled_attention
            )
    
    def forward(this, queries: Tensor, keys: Tensor, values: Tensor):
        multiHeadRaw: List[Tensor] = []
        for indx in range(this.h):
            head_i = this.head_attention_layers[indx](queries, keys, values)[1] # Take the attention_outputs, not attention weights
            multiHeadRaw.append(head_i)
        concat_head: Tensor =  concat(multiHeadRaw, dim=-1) # [Batch_size, Seq-Len, d_v * h]
        return this.w_o(concat_head)

        
            