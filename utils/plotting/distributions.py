############################################
# Imports
############################################

from utils.plotting.plots import *
from utils.observables import Observable

import numpy as np

import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

############################################
# Plotting
############################################

class Distribution(Observable):
	"""Custom Distribution class.

	Defines which Observables will be plotted depending on the
	specified dataset.
	"""
	def __init__(self,
				 real_data,
				 gen_data,
				 name,
				 log_dir,
				 dataset,
				 mean=[],
				 std=[],
				 latent=False):
		super(Distribution, self).__init__()
		self.real_data = real_data
		self.gen_data = gen_data
		self.name = name
		self.log_dir = log_dir
		self.dataset = dataset
		self.latent = latent
		self.mean = mean
		self.std = std

	def plot(self):
		if self.latent == True:
			self.latent_distributions()
		else:
			if self.dataset == 'Drell_Yan':
				self.drell_yan_distributions()
			elif self.dataset == 'Drell_Yan_Remake':
				self.drell_yan_distributions()
			elif self.dataset == '2d_ring_gaussian':
				self.basic_2d_distributions()
			elif self.dataset == '2d_linear':
				self.basic_2d_distributions()
			else:
				self.basic_2d_distributions()

		if True:
			plt.rc('text', usetex=True)
			plt.rc('font', family='serif')
			plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

		if False:
			with PdfPages(self.log_dir + '/' + self.dataset + '_' + self.name + '.pdf') as pp:
				for observable in self.args.keys():
					fig, axs = plt.subplots(1)
					plot_distribution(fig, axs, self.real_data, self.gen_data, self.args[observable])
					fig.savefig(pp, bbox_inches='tight', format='pdf', pad_inches=0.05)
					plt.close()

		if True:
			with PdfPages(self.log_dir + '/' + self.dataset + '_' + self.name + '_ratio.pdf') as pp:
				for observable in self.args.keys():
					fig, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios' : [4, 1], 'hspace' : 0.00})
					plot_distribution_ratio(fig, axs, self.real_data, self.gen_data,  self.args[observable])
					fig.savefig(pp, bbox_inches='tight', format='pdf', pad_inches=0.05)
					plt.close()

		if True:
			with PdfPages(self.log_dir + '/' + self.dataset + '_' + self.name + '_2d.pdf') as pp:
				for i, observable in enumerate(list(self.args2.keys())):
					for observable2 in list(self.args2.keys())[i+1:]:
						fig, axs = plt.subplots(1,3, figsize=(20,5))
						#fig, axs = plt.subplots(1)
						plot_2d_distribution(fig, axs, self.real_data, self.gen_data, self.args2[observable], self.args2[observable2])
						plt.subplots_adjust(wspace=0.45, hspace=0.25)
						fig.savefig(pp, bbox_inches='tight', format='pdf', pad_inches=0.05)
						plt.close()

	def basic_2d_distributions(self):
		 # Particle_id, observable, bins, range, x_label, log_scale

		args = {			 
			'x' : ([0], self.coordinate_0, 50, (-0.2,1.2) ,r'$x$', r'x',False),
			'y' : ([0], self.coordinate_1, 50, (-0.2,1.2) ,r'$y$', r'y',False),
		}	 

		args2 = {			 
			'x' : ([0], self.coordinate_0, 50, (-0.2,1.2) ,r'$x$', r'x',False),
			'y' : ([0], self.coordinate_1, 50, (-0.2,1.2) ,r'$y$', r'y',False),
		}
	 
		self.args = args
		self.args2 = args2

	def drell_yan_distributions(self):
		 # Particle_id, observable, bins, range, x_label, log_scale

		args = {			 
			#'pte1' : ([0], self.transverse_momentum, 40, (0,60) ,r'$p_{T, e^-}$ [GeV]', r'p_{T, e^-}',False),
		 	'pxe1' : ([0], self.x_momentum, 50, (-80,80), r'$p_{\mathrm{x}, e^-}$ [GeV]', r'p_{x, e^-}',False),
			'pye1' : ([0], self.y_momentum, 50, (-80,80), r'$p_{\mathrm{y}, e^-}$ [GeV]', r'p_{y, e^-}',False),
			'pze1' : ([0], self.z_momentum, 50, (-400,400), r'$p_{\mathrm{z}, e^-}$ [GeV]', r'p_{z, e^-}',False),
			#'Ee1'  : ([0], self.energy, 40, (0,300), r'$E_{e^-}$ [GeV]', r'E_{e^-}',False),
			#---------------------#		
			#'pte2' : ([1], self.transverse_momentum, 40, (0,60) ,r'$p_{T, e^+}$ [GeV]', r'p_{T, e^+}',False),
		 	#'pxe2' : ([1], self.x_momentum, 40, (-80,80), r'$p_{\mathrm{x}, e^+}$ [GeV]', r'p_{x, e^+}',False),
			#'pye2' : ([1], self.y_momentum, 40, (-80,80), r'$p_{\mathrm{y}, e^+}$ [GeV]', r'p_{y, e^+}',False),
			'pze2' : ([1], self.z_momentum, 50, (-400,400), r'$p_{\mathrm{z}, e^+}$ [GeV]', r'p_{z, e^+}',False),
			#'Ee2'  : ([1], self.energy, 40, (0,300), r'$E_{e^+}$ [GeV]', r'E_{e^+}',False),
			#---------------------#			
			#'ptZ' : ([0,1], self.transverse_momentum, 40, (-2,2) ,r'$p_{T, e^- e^+}$ [GeV]', r'p_{T, e^- e^+}',False),
			#'pxZ' : ([0,1], self.x_momentum, 40, (-2,2), r'$p_{\mathrm{x}, e^- e^+}$ [GeV]', r'p_{x, e^- e^+}',False),
			#'pyZ' : ([0,1], self.y_momentum, 40, (-200,200), r'$p_{\mathrm{y}, e^- e^+}$ [GeV]', r'p_{y, e^- e^+}',False),
			#'pzZ' : ([0,1], self.z_momentum, 40, (-750,750), r'$p_{\mathrm{z}, e^- e^+}$ [GeV]', r'p_{z, e^- e^+}',False),
			#'EZ'  : ([0,1], self.energy, 40, (40,500), r'$E_{e^- e^+}$ [GeV]', r'E_{e^- e^+}',False),
			'MZ'  : ([0,1], self.invariant_mass, 40, (82,100), r'$m_{e^- e^+}$ [GeV]', r'm_{e^- e^+}',False),
			#---------------------#			
		}	 

		args2 = {			 
			'pte1' : ([0], self.transverse_momentum, 40, (0,60) ,r'$p_{T, e^-}$ [GeV]', r'p_{T, e^-}',False),
		 	'pxe1' : ([0], self.x_momentum, 40, (-80,80), r'$p_{\mathrm{x}, e^-}$ [GeV]', r'p_{x, e^-}',False),
			'pye1' : ([0], self.y_momentum, 40, (-80,80), r'$p_{\mathrm{y}, e^-}$ [GeV]', r'p_{y, e^-}',False),
			'pze1' : ([0], self.z_momentum, 40, (-400,400), r'$p_{\mathrm{z}, e^-}$ [GeV]', r'p_{z, e^-}',False),
			'Ee1'  : ([0], self.energy, 40, (0,300), r'$E_{e^-}$ [GeV]', r'E_{e^-}',False),
			#---------------------#		
			#'pte2' : ([1], self.transverse_momentum, 40, (0,60) ,r'$p_{T, e^+}$ [GeV]', r'p_{T, e^+}',False),
		 	#'pxe2' : ([1], self.x_momentum, 40, (-80,80), r'$p_{\mathrm{x}, e^+}$ [GeV]', r'p_{x, e^+}',False),
			#'pye2' : ([1], self.y_momentum, 40, (-80,80), r'$p_{\mathrm{y}, e^+}$ [GeV]', r'p_{y, e^+}',False),
			'pze2' : ([1], self.z_momentum, 40, (-350,350), r'$p_{\mathrm{z}, e^+}$ [GeV]', r'p_{z, e^+}',False),
			#'Ee2'  : ([1], self.energy, 40, (0,300), r'$E_{e^+}$ [GeV]', r'E_{e^+}',False),
			#---------------------#			
			'ptZ' : ([0,1], self.transverse_momentum, 40, (-2,2) ,r'$p_{T, e^- e^+}$ [GeV]', r'p_{T, e^- e^+}',False),
			#'pxZ' : ([0,1], self.x_momentum, 40, (-2,2), r'$p_{\mathrm{x}, e^- e^+}$ [GeV]', r'p_{x, e^- e^+}',False),
			#'pyZ' : ([0,1], self.y_momentum, 40, (-200,200), r'$p_{\mathrm{y}, e^- e^+}$ [GeV]', r'p_{y, e^- e^+}',False),
			#'pzZ' : ([0,1], self.z_momentum, 40, (-750,750), r'$p_{\mathrm{z}, e^- e^+}$ [GeV]', r'p_{z, e^- e^+}',False),
			#'EZ'  : ([0,1], self.energy, 40, (40,500), r'$E_{e^- e^+}$ [GeV]', r'E_{e^- e^+}',False),
			'MZ'  : ([0,1], self.invariant_mass, 40, (82,100), r'$m_{e^- e^+}$ [GeV]', r'm_{e^- e^+}',False),
			#---------------------#			
		}	 
	 
		self.args = args
		self.args2 = args2
