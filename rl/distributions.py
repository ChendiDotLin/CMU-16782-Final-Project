import torch
import torch.nn as nn

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

FixedCategorical = torch.distributions.Categorical

old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)

log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: log_prob_cat(self, actions.squeeze(-1)).unsqueeze(-1)

FixedCategorical.mode = lambda self: self.probs.argmax(dim=1, keepdim=True)

class Categorical(nn.Module):
		def __init__(self, num_inputs, num_outputs):
				super(Categorical, self).__init__()

				init_ = lambda m: init(m,
							nn.init.orthogonal_,
							lambda x: nn.init.constant_(x, 0),
							gain=0.01)

				self.linear = init_(nn.Linear(num_inputs, num_outputs))

		def forward(self, x):
				x = self.linear(x)
				return FixedCategorical(logits=x)