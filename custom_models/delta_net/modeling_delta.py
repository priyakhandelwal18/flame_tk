import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from .config_delta import DeltaNetConfig
from .delta_layer import DeltaAttention

class DeltaNetForCausalLM(PreTrainedModel):
    config_class = DeltaNetConfig
    base_model_prefix = "deltanet"

    def __init__(self, config: DeltaNetConfig):
        super().__init__(config)
        d = config.hidden_size

        self.embed = nn.Embedding(config.vocab_size, d)
        self.layers = nn.ModuleList([
            nn.Sequential(
                DeltaAttention(
                    d_model=d,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                    chunk_size=config.chunk_size,
                    use_beta=config.use_beta,
                    use_gate=config.use_gate,
                    qk_activation=config.qk_activation,
                    qk_norm=config.qk_norm,
                    norm_eps=config.norm_eps,
                ),
                nn.LayerNorm(d, eps=config.norm_eps),
                nn.Linear(d, config.intermediate_size),
                nn.SiLU(),
                nn.Linear(config.intermediate_size, d),
                nn.Dropout(config.dropout),
            )
            for _ in range(config.num_hidden_layers)
        ])
        self.lm_head = nn.Linear(d, config.vocab_size, bias=False)
        self.post_init()

    def forward(self, input_ids, **kwargs):
        x = self.embed(input_ids)
        for layer in self.layers:
            res = x
            x = layer(x, **kwargs)
            x = x + res
        logits = self.lm_head(x)
        return {"logits": logits}
