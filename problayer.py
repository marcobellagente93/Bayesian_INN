import math
import sys
import torch.nn as nn
import numpy as np

class VBLinear(nn.Module):
	def __init__(self, in_features, out_features, prior_prec=1.0, map=False, local_reparam=True, freeze_weights=False):
		super(VBLinear, self).__init__()
		self.n_in = in_features
		self.n_out = out_features

		self.prior_prec = prior_prec
		self.map = map			     	# activate Maximum a posteriori: weighst are set to their mean value
		self.local_reparam = local_reparam   	# activate local reparameterization trick
		self.freeze_weights = freeze_weights 	# freezing random number to be able to evaluate several batches with same weight sample

		self.random = None
		self.random_local = None # different shape as self.random!
		self.bias = nn.Parameter(torch.Tensor(out_features))
		self.mu_w = nn.Parameter(torch.Tensor(out_features, in_features))
		self.logsig2_w = nn.Parameter(torch.Tensor(out_features, in_features))
		self.reset_parameters()

	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.mu_w.size(1))
		self.mu_w.data.normal_(0, stdv)
		self.logsig2_w.data.zero_().normal_(-9, 0.001)
		self.bias.data.zero_()

	def KL(self, loguniform=False):
		if loguniform:
			k1 = 0.63576; k2 = 1.87320; k3 = 1.48695
			log_alpha = self.logsig2_w - 2 * torch.log(self.mu_w.abs() + 1e-8)
			kl = -th.sum(k1 * torch.sigmoid(k2 + k3 * log_alpha) - 0.5 * F.softplus(-log_alpha) - k1)
		else:
			logsig2_w = self.logsig2_w.clamp(-11, 11)
			kl = 0.5 * (self.prior_prec * (self.mu_w.pow(2) + logsig2_w.exp()) - logsig2_w - 1 - np.log(self.prior_prec)).sum()
		return kl

	def forward(self, input):
		if self.training:
			if self.local_reparam:
				# local reparameterization trick is more efficient and leads to
				# an estimate of the gradient with smaller variance.
				# https://arxiv.org/pdf/1506.02557.pdf
				mu_out = F.linear(input, self.mu_w, self.bias)
				logsig2_w = self.logsig2_w.clamp(-11, 11)
				s2_w = logsig2_w.exp()
				var_out = F.linear(input.pow(2), s2_w) + 1e-8
				out = mu_out + var_out.sqrt() * torch.randn_like(mu_out)
			else:
				# 'non-local' reparameterization: not recommened for training
				logsig2_w = self.logsig2_w.clamp(-11, 11)
				s2_w = logsig2_w.exp()
				weight = self.mu_w + s2_w.sqrt() * torch.randn_like(s2_w)
				out = F.linear(input, weight, self.bias) + 1e-8

			return out

		else:
			if self.map:
				return F.linear(input, self.mu_w, self.bias)

			elif self.local_reparam:
				# local reparameterization trick for testing. Mostly useful for training.
				# Be carefull: this will lead to discontinous and unsmooth results for single weight samples as a function of the input
				mu_out = F.linear(input, self.mu_w, self.bias)
				logsig2_w = self.logsig2_w.clamp(-11, 11)
				s2_w = logsig2_w.exp()
				var_out = F.linear(input.pow(2), s2_w) + 1e-8

				if (not self.freeze_weights) or self.random_local is None: # None case: if this is first call
					self.random_local = torch.randn_like(mu_out) # a different random number for each input x_i!

				return mu_out + var_out.sqrt() * self.random_local


			else:
				if (not self.freeze_weights) or self.random is None: # None case: first call
					# draw a random number for each weight -> sharing random numbers for a batch of data
					self.random = torch.randn_like(self.logsig2_w)

				logsig2_w = self.logsig2_w.clamp(-11, 11)
				s2_w = logsig2_w.exp()
				weight = self.mu_w + s2_w.sqrt() * self.random
				out = F.linear(input, weight, self.bias) + 1e-8
				return out



	def __repr__(self):
		return self.__class__.__name__ + ' (' \
			   + str(self.n_in) + ' -> ' \
			   + str(self.n_out) + ')'
