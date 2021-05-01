from typing import Any

class GeneralObject:
    pass

config = GeneralObject()
config.vocab_size = 0
config.hidden_size = 0
config.hidden_dropout_prob = 0
config.n_gate = 0
config.proj_dim = 0

print(config.n_gate)