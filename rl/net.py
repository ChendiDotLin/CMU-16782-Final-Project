import numpy as np
import pdb
import torch.nn.functional as F
import torch
import torch.nn as nn
from .distributions import *

class Net(nn.Module):
	def __init__(self,state_size, action_size, emb_size, class_size, non_lin = nn.ReLU):
		super(Net, self).__init__()
		self.state_size = state_size
		self.action_size = action_size
		self.emb_size = emb_size
		self.class_size = class_size
		self.non_lin = nn.ReLU
		self.is_recurrent = False

		#defining the policy and action heads.
		self._value = nn.Sequential(nn.Linear(self.emb_size, self.emb_size), self.non_lin(inplace=True), nn.Linear(self.emb_size,1))
		self._policy = nn.Sequential(nn.Linear(self.emb_size, self.emb_size), self.non_lin(inplace=True))
		self.dist = Categorical(self.emb_size,self.action_size)

		#defining the neural network
		self.layer_1 = nn.Sequential(nn.Linear(self.state_size, self.emb_size), self.non_lin(inplace=True))
		self.layer_2 = nn.Sequential(nn.Linear(self.emb_size, self.emb_size), self.non_lin(inplace=True))

	def _fwd(self, inp):
		#inp should be of the the size batch_size*self.num_states, one_hot should be of the size : batch_size*self.state_size*self.class_size
		# pdb.set_trace()
		# one_hot = F.one_hot(inp.to(torch.int64), self.class_size)
		# one_hot_inp = one_hot.view(-1, self.state_size*self.class_size)

		flatten_inp = torch.flatten(inp, start_dim=1)
		#passing through the network
		h = self.layer_1(flatten_inp)
		h = self.layer_2(h)

		return h

	def act(self, inp, state, mask=None, deterministic=False):
		x = self._fwd(inp)
		value = self._value(x)
		dist = self.dist(self._policy(x))
		if deterministic:
			action = dist.mode()
		else:
			action = dist.sample()
		action_log_probs = dist.log_probs(action).view(-1,1)

		return value,action,action_log_probs,state

	def evaluate_actions(self, inp, state, mask, action):
		x = self._fwd(inp)
		value = self._value(x)
		dist = self.dist(self._policy(x))
		action_log_probs = dist.log_probs(action)
		dist_entropy = dist.entropy().mean()
		
		return value,action_log_probs,dist_entropy,state

	def get_value(self, inp, state, mask):
		x = self._fwd(inp)
		value = self._value(x)
		
		return value