# -*- coding: utf-8 -*-
from typing import Optional
from transformers.configuration_utils import PretrainedConfig

class DeltaNetConfig(PretrainedConfig):
    model_type = "delta_net"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        # —— align with 8 heads × 16 head-dim = 2048 state size
        hidden_size:            int = 128,   # embedding dim = heads × head_dim
        num_hidden_layers:      int = 24,
        num_heads:              int = 8,     # so head_dim = 128 / 8 = 16
        intermediate_size:      Optional[int] = None,
        hidden_act:             str = "swish",
        dropout:                float = 0.1,
        chunk_size:             int = 16,    # size of each delta “chunk”
        use_cache:              bool = True,
        max_position_embeddings: int = 2048,
        # you can add other delta-specific flags here if needed
        **kwargs,
    ):
        # core Transformer settings
        self.hidden_size            = hidden_size
        self.num_hidden_layers      = num_hidden_layers
        self.num_heads              = num_heads
        self.intermediate_size      = intermediate_size or (4 * hidden_size)
        self.hidden_act             = hidden_act
        self.dropout                = dropout

        # delta-rule specific
        self.chunk_size             = chunk_size
        self.use_cache              = use_cache

        # positional
        self.max_position_embeddings = max_position_embeddings

        super().__init__(**kwargs)
