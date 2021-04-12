from utils.train_utils import *
from utils.observables import *
from utils.plotting.distributions import *
from utils.plotting.plots import *
from load_data import *

from FrEIA.framework import *
from FrEIA.modules import *
from model import INN
from losses import * 

import sys, os

import config as c
import opts
opts.parse(sys.argv)
config_str = ""
config_str += "==="*30 + "\n"
config_str += "Config options:\n\n"

for v in dir(c):
	if v[0]=='_': continue
	s=eval('c.%s'%(v))
	config_str += " {:25}\t{}\n".format(v,s)

config_str += "==="*30 + "\n"

print(config_str)

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

train_loader, validate_loader, dataset_size, data_shape, scales = Loader(c.dataset, c.batch_size, c.test, c.scaler, c.on_shell, c.mom_cons, c.weighted)

train_params = {
	'lr': c.lr,
	'betas': [0.9, 0.999],
	'decay': c.weight_decay,
	'gamma': c.gamma,
	'on_shell': c.on_shell,
	'n_epochs': c.n_epochs,
	'batch_size': c.batch_size
	}

Flow = INN(num_coupling_layers=c.n_blocks, in_dim=data_shape, num_layers=c.n_layers, internal_size=c.n_units, params=train_params)
Flow.define_model_architecture()
Flow.set_optimizer()

print(Flow.model)
print('Total parameters: %d' % sum([np.prod(p.size()) for p in Flow.params_trainable]))

data = pd.read_hdf('./data/' + c.dataset + '.h5').values

if c.on_shell:
	data = remove_energies(data)
if c.mom_cons:
	data = remove_momenta(data)

if c.weighted:
	data_shape = data.shape[1] - 1
	scales = np.std(data[:,:-1],0)
else:
	data_shape = data.shape[1]
	#scales = np.std(data)
	scales = np.std(data,0)

log_dir = c.save_dir

checkpoint_path = log_dir + '/n_epochs_1/' + '/checkpoint_epoch_000.pth'

Flow, Flow.optim, init_epoch = load_checkpoint(checkpoint_path, Flow, Flow.optim)
Flow.model.eval()

size = 100000
noise = torch.randn(size, data_shape)

inv = Flow.model(noise.float(), rev=True).detach().numpy()
inv = inv.reshape(size, data_shape) * scales

real = get_real_data(c.dataset, c.test, size)

if c.mom_cons:
	inv = add_momenta(inv)
	real = add_momenta(real)

if c.on_shell:
	inv = add_energies(torch.from_numpy(inv).float()).detach().numpy()
	real = add_energies(torch.from_numpy(real).float()).detach().numpy()

#distributions = Distribution(noise, latent, 'latent', log_dir + '/epochs_' + str(c.n_epochs), args.dataset, latent=True)
#distributions.plot()
distributions = Distribution(real, inv, 'target', log_dir + '/n_epochs_' + str(c.n_epochs), c.dataset)
distributions.plot()
