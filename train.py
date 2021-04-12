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

if c.adversarial:
	# Target space discriminator
	D = netD(nz=data_shape, n_layers=5, n_units=30, params=train_params)
	#D = netD(nz=data_shape, n_layers=int(c.n_layers), n_units=int(c.n_units), params=train_params)
	D.set_optimizer()
	D.print_network()
	D.to(device)

	# Latent space discriminator
	#Dlat = netD(nz=data_shape, n_layers=int(c.n_layers), n_units=int(c.n_units), params=train_params)
	#Dlat.set_optimizer()
	#Dlat.print_network()
	#Dlat.to(device)
try:

	log_dir = c.save_dir

	if not os.path.exists(log_dir + '/n_epochs_' + str(c.n_epochs)):
		os.makedirs(log_dir + '/n_epochs_' + str(c.n_epochs))

	# setup some varibles
	Flow_loss_meter = AverageMeter()
	Val_loss_meter  = AverageMeter()
	if c.adversarial:
		Adv_loss_meter = AverageMeter()

	Flow_loss_list = []
	if c.adversarial:
		Adv_loss_list = []

	if c.load_model:
		checkpoint_path = log_dir 
		Flow, Flow.optim, init_epoch = load_checkpoint(checkpoint_path, Flow, Flow.optim)

	for epoch in range(c.n_epochs):

		for iteration in range(c.n_its_per_epoch):

			i=0
			j=0

			for events in train_loader:

				Flow.model.train()
				Flow.optim.zero_grad()

				if c.adversarial:
					D.train
					D.zero_grad()
					#Dlat.train
					#Dlat.zero_grad()
	
				if c.weighted:
					weights = events[:,-1] 
					events = events[:,:-1]

				events /= scales

				if c.train:
		
					gauss_output = Flow.model(events.float())

					if c.weighted:
						temp = torch.sum(gauss_output**2/2,1)
						#temp = torch.mean(gauss_output**2/2,1)
						loss = (torch.mean(weights * temp) - torch.mean(weights * Flow.model.log_jacobian(run_forward=False))) #/ gauss_output.shape[1]
					else:
						loss = torch.mean(gauss_output**2/2) - torch.mean(Flow.model.log_jacobian(run_forward=False)) / gauss_output.shape[1]

						noise = torch.randn(c.batch_size, data_shape)
						#loss += mmd(gauss_output, noise, [1])

					if c.mmd:
						noise = torch.randn(batch_size, data_shape)
						inv = Flow.model(noise.float(), rev=True).reshape(c.batch_size, data_shape)

						inv *= torch.tensor(scales).float()
						events *= scales

						if c.mom_cons:
							events = add_momenta(events)
							inv = add_momenta(inv)

						if c.on_shell:
							events = add_energies(events.float())
							inv = add_energies(inv.float())

						mass_real = get_masses(events, [[0,1]]).T
						mass_fake = get_masses(inv, [[0,1]]).T

						loss = (c.lambda_mmd + (epoch * 0.0005)) * mmd(mass_fake, mass_real, [100, 10, 1])

					if c.adversarial:
						label_real = torch.ones(batch_size).to(device)
						label_fake = torch.zeros(batch_size).to(device)
		
						for i in range(c.diter):
							noise = torch.randn(batch_size, data_shape)
							#inv = Flow.model(noise.float(), rev=True)
							#inv = inv.reshape(batch_size, data_shape)

							gauss_output = Flow.model(events.float())

							d_result_real = D(noise).view(-1)
							d_result_fake = D(gauss_output.detach()).view(-1)
							d_loss_real_ = torch.nn.ReLU()(1.0 - d_result_real).mean()
							d_loss_fake_ = torch.nn.ReLU()(1.0 + d_result_fake).mean()
						
							d_loss = d_loss_real_ + d_loss_fake_

							D.zero_grad()
							d_loss.backward()
							D.optim.step()

						#noise = torch.randn(batch_size, data_shape)
						gauss_output = Flow.model(events.float())

						d_result_lat = D(gauss_output).view(-1)
						g_loss = - d_result_lat.mean()

						loss += g_loss

						#d_result_real = Dlat(noise).view(-1)
						#d_result_fake = Dlat(gauss_output.detach()).view(-1)
						#d_loss_real_ = torch.nn.ReLU()(1.0 - d_result_real).mean()
						#d_loss_fake_ = torch.nn.ReLU()(1.0 + d_result_fake).mean()
						
						#d_loss = d_loss_real_ + d_loss_fake_

						#d_result_inv = Dlat(gauss_output).view(-1)
						#g_loss = - d_result_inv.mean()

						#loss += 0.1 * g_loss						
						#Adv_loss_meter.update(loss.item())
	
					if c.variational:
						kl = 0
						for l in range(len(Flow.model.module_list)):
							if Flow.model.module_list[l] is not None:
								if list(Flow.model.module_list[l].parameters()) != []:
									kl += (1/dataset_size) * Flow.model.module_list[l].s.layers[0].KL()
									kl += (1/dataset_size) * Flow.model.module_list[l].s.layers[3].KL()
									kl += (1/dataset_size) * Flow.model.module_list[l].s.layers[6].KL()
				
						loss += 1.0 * kl
				
					Flow_loss_meter.update(loss.item())

					loss.backward()
					Flow.optim.step()

				i += 1

			if epoch == 0 or epoch % c.show_interval == 0:
				print_log(epoch, c.n_epochs, i + 1, len(train_loader), Flow.scheduler.optimizer.param_groups[0]['lr'],
							   c.show_interval, Flow_loss_meter)

			elif (epoch + 1) == len(train_loader):
				print_log(epoch, c.n_epochs, i + 1, len(train_loader), Flow.scheduler.optimizer.param_groups[0]['lr'],
							   (i + 1) % c.show_interval, Flow_loss_meter)

			Flow_loss_meter.reset()
			if c.adversarial:
				Adv_loss_meter.reset()

			"""Validation test"""
			if epoch == 0 or epoch % c.show_interval == 0:
				for events in validate_loader:

					Flow.model.eval()
					if c.adversarial:
						D.eval()
						#Dlat.eval()

					events /= scales
				
					gauss_output = Flow.model(events.float())
					loss = torch.mean(gauss_output**2/2) - torch.mean(Flow.model.log_jacobian(run_forward=False)) / gauss_output.shape[1]
	
					if c.adversarial:
						noise = torch.randn(batch_size, data_shape)
						inv = Flow.model(noise.float(), rev=True)
						inv = inv.reshape(batch_size, data_shape)

						d_result_inv = D(inv).view(-1)
						g_loss = - d_result_inv.mean()
				
						loss += g_loss

					Val_loss_meter.update(loss.item())
					j += 1

				print('VALIDATION')
				print_log(epoch, c.n_epochs, j + 1, len(validate_loader), Flow.scheduler.optimizer.param_groups[0]['lr'],
							   (i + 1) % c.show_interval, Val_loss_meter)

			Val_loss_meter.reset()

		if epoch % c.save_interval == 0 or epoch + 1 == c.n_epochs:
			if c.save_model == True:
				checkpoint = {
					'epoch': epoch,
					'flow': Flow.model.state_dict(),
					'optimizerF': Flow.optim.state_dict(),
					}

				save_checkpoint(checkpoint, log_dir + '/n_epochs_' + str(c.n_epochs), 'checkpoint_epoch_%03d' % (epoch))

			if c.test == True:
				size = 50000
			else:
				size = 200000

			with torch.no_grad():
				real = get_real_data(c.dataset, c.test, size)

				if c.on_shell:
					real = remove_energies(real)

				if c.mom_cons:
					real = remove_momenta(real)

				if c.weighted:
					#real[:,:-1] = real[:,:-1][:size] / scales
					weights = torch.from_numpy(real[:,[-1]]).float() #* torch.normal(1,0.3, [size, 1])
				else:
					real = real[:size] / scales

				noise = torch.randn(size, data_shape)
				#latent = Flow.model(torch.from_numpy(real).float()).detach().numpy()
				inv = Flow.model(noise.float(), rev=True).detach().numpy()				

				inv = inv.reshape(size, data_shape)

				#if c.weighted:
				#	real = real
				#	real[:,:-1] *= scales
				#else:
				#	real *= scales
				inv *= scales
				real *= scales

				if c.mom_cons:
					real = add_momenta(real)
					inv = add_momenta(inv)

				if c.on_shell:
					real = add_energies(torch.from_numpy(real).float()).detach().numpy()
					inv = add_energies(torch.from_numpy(inv).float()).detach().numpy()

			#distributions = Distribution(noise, latent, 'epoch_%03d' % (epoch) + '_latent', log_dir + '/n_epochs_' + str(c.n_epochs), c.dataset, latent=True)
			#distributions.plot()
			distributions = Distribution(real, inv, 'epoch_%03d' % (epoch) + '_target', log_dir + '/n_epochs_' + str(c.n_epochs), c.dataset)
			distributions.plot()

		Flow.scheduler.step()

		#print('current lr:', Flow.optim.param_groups[0]['lr'])

except:
	if c.checkpoint_on_error:
		model.save(c.filename + '_ABORT')
	raise 

#if __name__ == '__main__':
#	os.system('mkdir -p logs')

	#c = parse()
	#print(c)

#	if c.train:
#		train_net(c)
#	else:
#		predict(c)
