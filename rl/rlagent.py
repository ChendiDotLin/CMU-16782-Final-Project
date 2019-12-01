from learner.PPO import *
from learner.storage import *
import numpy as np

class Agent():
	def __init__(self, args, policy, obs_shape, action_space):
		super().__init__()

		self.obs_shape = obs_shape
		self.action_space = action_space
		self.actor_critic = policy
		self.args = args
		self.trainer = PPO(self.actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch, args.value_loss_coef, args.entropy_coef, lr=args.lr,max_grad_norm=args.max_grad_norm)
		self.rollouts = RolloutStorage(args.num_steps, args.num_processes, self.obs_shape, self.action_space, recurrent_hidden_state_size=1)

	def load_model(self, policy_state):
		self.actor_critic.load_state_dict(policy_state)

	def initialize_obs(self, obs):
		# this function is called at the start of episode
		self.rollouts.obs[0].copy_(obs)

	def update_rollout(self, obs, reward, mask):
		obs_t = torch.from_numpy(obs).float()
		reward_t = torch.from_numpy(np.stack(reward)).float()
		mask_t = torch.FloatTensor(mask)
		self.rollouts.insert(obs_t, self.states, self.action, self.action_log_prob, self.value, reward_t, mask_t)

	def act(self, step, deterministic=False):
		self.value, self.action, self.action_log_prob, self.states = self.actor_critic.act(self.rollouts.obs[step],self.rollouts.recurrent_hidden_states[step],self.rollouts.masks[step],deterministic=deterministic)
		return self.action

	def wrap_horizon(self):
		next_value = self.actor_critic.get_value(self.rollouts.obs[-1], self.rollouts.recurrent_hidden_states[-1], self.rollouts.masks[-1]).detach()
		self.rollouts.compute_returns(next_value, True, self.args.gamma, self.args.tau)

	def after_update(self):
		self.rollouts.after_update()

	def update(self):
		return self.trainer.update(self.rollouts)

	def to(self, device):
		self.actor_critic.to(device)
		self.rollouts.to(device)
		