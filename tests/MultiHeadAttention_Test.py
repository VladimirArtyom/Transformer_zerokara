import sys
import unittest
sys.path.insert(1, "..")

from layers.MultiHeadAttention import MultiHeadAttention
from torch import rand, Tensor, manual_seed

class TestMultiHeadAttention(unittest.TestCase):

    def setUp(this):
        manual_seed(0)
        this.batch_size: int = 2
        this.seq_len: int = 4
        this.h: int = 8
        this.d_k: int = 64
        this.d_v: int = 64
        this.d_model: int = this.d_v * this.h
        this.queries: Tensor = rand(this.batch_size,
                                    this.seq_len,
                                    this.d_model)
        this.keys: Tensor = rand(this.batch_size, 
                                this.seq_len,
                                this.d_model)
        this.values: Tensor = rand(this.batch_size,
                                this.seq_len,
                                this.d_model)

        this.multiHead = MultiHeadAttention(d_k=this.d_k, d_v=this.d_v, h=this.h)


    def test_output_shape(this):
        multiHeadResult: Tensor = this.multiHead(this.queries, this.keys, this.values)
        expected_shape = (this.batch_size, this.seq_len, this.h * this.d_v)
        this.assertEqual(multiHeadResult.shape, expected_shape, f"Expected {expected_shape}, but got {multiHeadResult.shape}")

if __name__ == "__main__":
    unittest.main()