from transformers import AutoConfig, AutoModelForCausalLM
from .config_delta   import DeltaNetConfig
from .modeling_delta import DeltaNetForCausalLM

AutoConfig.register("delta_net", DeltaNetConfig)
AutoModelForCausalLM.register(DeltaNetConfig, DeltaNetForCausalLM)
