import torch
import numpy as np

def pairwise_distance(x, y):

    if not len(x.shape) == len(y.shape) == 2:
        raise ValueError('Both inputs should be matrices.')

    x = torch.unsqueeze(x,2)
    y = y.T
    output = torch.sum((x - y)**2,1).T

    return output

def cauchy(X, Y, sigma):

    gamma = 1/sigma**2

    return 1/(1 + gamma * pairwise_distance(X, Y))


def breit_wigner(X, Y, gamma, reduce_mean=True):
    return cauchy(X, Y, gamma/2, reduce_mean)

def mmd(X, Y, sigma=1, kernel_type="summed", kernel_name="cauchy"):

	kernel = None
	if kernel_name == "cauchy":
		kernel = cauchy
	elif kernel_name == "breit_wigner":
		kernel = gaussian
	else:
		raise(NotImplementedError("Kernel {} is not available for MMD Loss"))
	if kernel_type == "standard":
		sigma = sigma
		return torch.sqrt(torch.abs(kernel(X, X, sigma) + kernel(Y, Y, sigma) - 2. * kernel(X, Y, sigma)))
	elif kernel_type == "summed":
		divergence = 0
		if not type(sigma) == list and not type(sigma) == np.ndarray and not type(sigma) == torch.Tensor:
			sigma = [sigma]
		for i, s in enumerate(sigma):
			a = torch.mean(kernel(X,X,s))
			b = torch.mean(kernel(Y,Y,s))
			c = torch.mean(kernel(X,Y,s))

			div = a + b - 2. * c
			#divergence += div
			divergence += torch.sqrt(torch.max(div, torch.ones(div.shape).to(div.device)*1e-10))

			if div <= 0:
				print("Warning, div is {}".format(div))
		return divergence
