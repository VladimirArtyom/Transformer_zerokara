import sys
import unittest

sys.path.insert(1, "..")

from torch import rand, Tensor, manual_seed, allclose, ones_like
from layers.ScaledDotProduct import ScaledDotProduct

class TestScaledDotProductAttention(unittest.TestCase):

    def setUp(this):
        manual_seed(0)

        this.batch_size: int = 2
        this.seq_len: int = 5
        this.d_k: int = 64
        this.d_v: int = 64
        this.d_model: int = 512

        this.queries: Tensor = rand(this.batch_size,
                                    this.seq_len,
                                    this.d_model)
        this.values: Tensor =  rand(this.batch_size,
                                    this.seq_len,
                                    this.d_model)
        this.keys: Tensor =  rand(this.batch_size,
                                  this.seq_len,
                                  this.d_model)
        this.scaled_dot = ScaledDotProduct(d_model=this.d_model,
                                           d_k=this.d_k,
                                           d_v=this.d_v)

    def test_softmax_sum(this):
        attention_weights, _ = this.scaled_dot(this.queries, this.keys, this.values)
        attention_sum = attention_weights.sum(dim=-1)
        this.assertTrue(allclose(attention_sum, ones_like(attention_sum), atol=1e-5))



if __name__ == "__main__":
    unittest.main()