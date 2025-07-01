# -*- coding: utf-8 -*-
from typing import Optional
from transformers.configuration_utils import PretrainedConfig

class DeltaNetConfig(PretrainedConfig):
    model_type = "delta_net"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        # transformer
        self,
        hidden_size: int            = 128,       # 8 heads × 16 head-dim
        num_hidden_layers: int      = 24,
        num_heads: int              = 8,
        intermediate_size: Optional[int] = None,
        hidden_act: str             = "swish",
        dropout: float              = 0.1,

        # delta
        chunk_size: int             = 16,
        use_beta: bool              = True,
        use_gate: bool              = False,
        qk_activation: str          = "silu",
        qk_norm: str                = "l2",
        norm_eps: float             = 1e-6,

        # encoding & embedding
        max_position_embeddings: int= 2048,
        use_cache: bool             = True,
        pad_token_id: int           = 0,
        bos_token_id: int           = 1,
        eos_token_id: int           = 2,
        tie_word_embeddings: bool   = False,
        initializer_range: float    = 0.02,
        vocab_size: int             = 32000,
        **kwargs,
    ):
        # transformer
        self.hidden_size            = hidden_size
        self.num_hidden_layers      = num_hidden_layers
        self.num_heads              = num_heads
        self.intermediate_size      = intermediate_size or (4 * hidden_size)
        self.hidden_act             = hidden_act
        self.dropout                = dropout

        # delta
        self.chunk_size             = chunk_size
        self.use_beta               = use_beta
        self.use_gate               = use_gate
        self.qk_activation          = qk_activation
        self.qk_norm                = qk_norm
        self.norm_eps               = norm_eps

        # encoding & embedding
        self.max_position_embeddings= max_position_embeddings
        self.use_cache              = use_cache
        self.pad_token_id           = pad_token_id
        self.bos_token_id           = bos_token_id
        self.eos_token_id           = eos_token_id
        self.tie_word_embeddings    = tie_word_embeddings
        self.initializer_range      = initializer_range
        self.vocab_size             = vocab_size

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
