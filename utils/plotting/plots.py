import numpy as np
import pandas as pd

import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.stats import kde
import sys
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import ScalarFormatter

import matplotlib.colors as mcolors
import matplotlib.cm as cm

from matplotlib.legend_handler import HandlerBase
from mpl_toolkits.axes_grid1 import make_axes_locatable

gcolor = '#3b528b'
dcolor = '#e41a1c'
teal = '#10AB75'

FONTSIZE = 20 #18, (20 2d) 12 (if latex is false in distributions)

############################################
# Functions
############################################

class ScalarFormatterForceFormat(ScalarFormatter):
	def _set_format(self):	# Override function that finds format to use.
		self.format = "%1.1f"  # Give format her

class AnyObjectHandler(HandlerBase):
	def create_artists(self, legend, orig_handle,
					   x0, y0, width, height, fontsize, trans):
		l1 = plt.Line2D([x0,y0+width], [0.7*height,0.7*height],
						   linestyle=orig_handle[1], color=orig_handle[0])
		l2 = plt.Line2D([x0,y0+width], [0.3*height,0.3*height],
						   linestyle = orig_handle[2], color = orig_handle[0])
		return [l1, l2]

def plot_distribution_ratio(fig, axs, y_train, y_predict, args):
	# Particle_id, observable, bins, range, x_label, log_scale

	y_train = args[1](y_train, args[0])
	y_predict = args[1](y_predict, args[0])

	if args[6]:
		axs[0].set_yscale('log')
	
	y_t, x_t = np.histogram(y_train, args[2], density=True, range=args[3])
	y_p, x_p = np.histogram(y_predict, args[2], density=True, range=args[3])

	axs[0].step(x_t[:args[2]], y_t, '#f03b20', label='Truth', linewidth=1.0, where='mid')
	axs[0].step(x_t[:args[2]], y_p, '#2c7fb8', label='BINN', linewidth=1.0, where='mid')

	for j in range(2):
		for label in ( [axs[j].yaxis.get_offset_text()] +
					  axs[j].get_xticklabels() + axs[j].get_yticklabels()):
			label.set_fontsize(FONTSIZE)

	yfmt = ScalarFormatterForceFormat()
	yfmt.set_powerlimits((0,0))
	axs[0].yaxis.set_major_formatter(yfmt)

	axs[0].set_ylabel('Normalized', fontsize = FONTSIZE)
	axs[0].legend(loc='upper left', prop={'size': int(FONTSIZE-4)}, frameon=False)

	axs[1].set_ylabel(r'$\frac{\text{Truth}}{\text{BINN}}$', fontsize = FONTSIZE-2)

	y_r = (y_p)/y_t

	y_r [np.isnan(y_r )==True]=1
	y_r [y_r==np.inf]=1

	axs[1].step(x_t[:args[2]], y_r, 'black', linewidth=1.0, where='mid')
	axs[1].set_ylim((0.72,1.28))
	axs[1].set_yticks([0.8, 1.0, 1.2])
	axs[1].set_yticklabels([r'$0.8$', r'$1.0$', "$1.2$"])

	axs[1].axhline(y=1,linewidth=1, linestyle='--', color='grey')
	axs[1].axhline(y=2,linewidth=1, linestyle='--', color='grey')
	#axs[1].axhline(y=1.1,linewidth=1, linestyle='--', color='grey')
	#axs[1].axhline(y=0.9,linewidth=1, linestyle='--', color='grey')
	axs[1].set_xlabel(args[4], fontsize = FONTSIZE)

def plot_distribution_ratio_uncertainties(fig, axs, y_train, y_predict, args, mean, std):
	# Particle_id, observable, bins, range, x_label, log_scale

	y_train = args[1](y_train, args[0])
	#y_predict = args[1](y_predict, args[0])

	if args[6]:
		axs[0].set_yscale('log')
	
	marg = []
	marg2 = []
	unc = []
	unc2 = []

	y_t, x_t = np.histogram(y_train, args[2], density=True, range=args[3])
	#y_p, x_p = np.histogram(y_predict, args[2], density=True, range=args[3])
	"""
	# FUCKING HELL
	marg = np.empty([args[2],args[2]])
	unc  = np.empty([args[2],args[2]])

	x = np.linspace(-8, 8, args[2]+1)
	y = np.linspace(-0.5, 7.5, args[2]+1)

	for i in range(args[2]):
		for j in range(args[2]):
				idx = (y_predict[:,0] < x[i+1]) * (y_predict[:,0] > x[i]) * (y_predict[:,1] < y[j+1]) * (y_predict[:,1] > y[j]) 
				m = mean[idx]
				s = std[idx]
				marg[i][j] = np.sum(m)
				unc[i][j] = np.sum(s)

	y_predict = args[1](y_predict, args[0])
	y_p, x_p = np.histogram(y_predict, args[2], density=True, range=args[3])
	
	if args[3][0] == -8.0:
		marg = np.nansum(marg,1)
		#unc=unc**2
		#unc = np.sqrt(np.nansum(unc,1))
		unc = np.nansum(unc,1)
	else:
		marg = np.nansum(marg,0)
		#unc=unc**2
		#unc = np.sqrt(np.nansum(unc,0))
		unc = np.nansum(unc,0)
	
	NORM = (np.mean(marg)/np.mean(y_p))
	marg /= NORM
	unc /= NORM	

	rel_unc = unc/marg
	"""
	y_predict = args[1](y_predict, args[0])
	y_p, x_p = np.histogram(y_predict, args[2], density=True, range=args[3])


	# SIMPLIFIED
	for i in range(args[2]):
		idx = (y_predict < x_p[i+1]) * (y_predict > x_p[i])
		m = mean[idx]
		s = std[idx]
		#w = map_weights[idx]/np.mean(map_weights[idx])
		#print(np.sum(np.isnan(s)))
		#print(np.sum(np.isnan(m)))
		unc.append(np.mean(s))
		marg.append(np.mean(m))

	marg = np.asarray(marg)
	unc = np.asarray(unc)

	#unc = np.asarray(unc) / (10000 * (x_p[1] - x_p[0]))
	rel_unc = unc/marg
	#rel_unc[rel_unc > 1] = 1

	#s = pd.HDFStore('unc_10.h5')
	#s.append('data', pd.DataFrame(rel_unc))
	#s.close()
	
	axs[0].step(x_t[:args[2]], y_t, '#f03b20', label='Truth', linewidth=1.0, where='mid')
	axs[0].step(x_t[:args[2]], y_p, '#2c7fb8', label='BINN', linewidth=1.0, where='mid')
	#axs[0].step(x_t[:args[2]], marg, '#2c7fb8', label='BINN', linewidth=1.0, where='mid')
	#axs[0].fill_between(x_t[:args[2]], y_p - y_p * unc2, y_p + y_p * unc2, color= '#2c7fb8', step='mid', alpha=0.5)
	axs[0].fill_between(x_t[:args[2]], y_p - y_p * rel_unc, y_p + y_p * rel_unc, color= '#2c7fb8', step='mid', alpha=0.5)
	#axs[0].bar(x_t[:args[2]], height = y_p * rel_unc, bottom = y_p, width = np.diff(x_t), alpha=0.5, zorder=-1, color='#2c7fb8')
	#axs[0].bar(x_t[:args[2]], height = y_p * rel_unc, bottom = y_p - y_p * rel_unc, width = np.diff(x_t), alpha=0.5, zorder=-1, color='#2c7fb8')
	#axs[0].step(x_t[:args[2]], y_p + y_p * rel_unc, '#2c7fb8', label='BINN', linewidth=1.0, where='mid')
	#axs[0].step(x_t[:args[2]], y_p - y_p * rel_unc, '#2c7fb8', label='BINN', linewidth=1.0, where='mid')

	for j in range(2):
		for label in ( [axs[j].yaxis.get_offset_text()] +
					  axs[j].get_xticklabels() + axs[j].get_yticklabels()):
			label.set_fontsize(FONTSIZE)

	yfmt = ScalarFormatterForceFormat()
	yfmt.set_powerlimits((0,0))
	axs[0].yaxis.set_major_formatter(yfmt)

	axs[0].set_ylabel('Normalized', fontsize = FONTSIZE)
	#axs[0].legend(loc='lower center', prop={'size': int(FONTSIZE-4)}, frameon=False)
	axs[0].legend(loc='lower center', bbox_to_anchor=[0.4, 0.1], prop={'size': int(FONTSIZE-4)}, frameon=False)

	axs[1].set_ylabel(r'$\frac{\text{Truth}}{\text{BINN}}$', fontsize = FONTSIZE-2)

	y_r = (y_p)/y_t
	y_r_up = (y_p + y_p * rel_unc)/y_t
	y_r_do = (y_p - y_p * rel_unc)/y_t
	#y_r_up = (y_p + y_p * unc2)/y_t
	#y_r_do = (y_p - y_p * unc2)/y_t

	y_r [np.isnan(y_r )==True]=1
	y_r [y_r==np.inf]=1

	axs[1].step(x_t[:args[2]], y_r, 'black', linewidth=1.0, where='mid')
	axs[1].fill_between(x_t[:args[2]], y_r_do, y_r_up, step='mid', alpha=0.5)
	#axs[1].step(x_t[:args[2]], y_r_up, 'black', linewidth=1.0, where='mid')
	#axs[1].step(x_t[:args[2]], y_r_do, 'black', linewidth=1.0, where='mid')
	axs[1].set_ylim((0.72,1.28))
	axs[1].set_yticks([0.8, 1.0, 1.2])
	axs[1].set_yticklabels([r'$0.8$', r'$1.0$', "$1.2$"])

	axs[1].axhline(y=1,linewidth=1, linestyle='--', color='grey')
	axs[1].axhline(y=2,linewidth=1, linestyle='--', color='grey')
	axs[1].set_xlabel(args[4], fontsize = FONTSIZE)

def plot_distribution(fig, axs, y_train, y_predict, args):
	"""Plot the distributions"""
	# Particle_id, observable, bins, range, x_label, log_scale
	y_train = args[1](y_train, args[0])

	y_predict = args[1](y_predict, args[0])

	if args[6]:
		axs.set_yscale('log')

	y_t, x_t = np.histogram(y_train, args[2], density=True, range=args[3])
	y_p, x_p = np.histogram(y_predict, args[2], density=True, range=args[3])
	#y_p, x_p = np.histogram(y_predict, args[2], density=True, range=args[3], weights=w)

	axs.step(x_t[:args[2]], y_t, '#f03b20', label='Truth', linewidth=1.0, where='mid')
	axs.step(x_p[:args[2]], y_p, '#2c7fb8', label='BINN', linewidth=1.0, where='mid')

	for label in ( [axs.yaxis.get_offset_text()] +
				  axs.get_xticklabels() + axs.get_yticklabels()):
		label.set_fontsize(FONTSIZE)

	yfmt = ScalarFormatterForceFormat()
	yfmt.set_powerlimits((0,0))
	axs.yaxis.set_major_formatter(yfmt)

	axs.set_ylabel('Normalized', fontsize = FONTSIZE)
	axs.legend(loc='upper right', prop={'size':(FONTSIZE-4)}, frameon=False)
	axs.set_xlabel(r'$%s$' %(args[5]), fontsize = FONTSIZE)

def plot_distribution_uncertainties(fig, axs, y_train, y_predict, args, mean, std):
	"""Plot the distributions"""
	# Particle_id, observable, bins, range, x_label, log_scale
	y_train = args[1](y_train, args[0])[:30000]
	y_predict = args[1](y_predict, args[0])

	if args[6]:
		axs[0].set_yscale('log')
	
	marg = []
	unc = []
	unc2 = []

	y_t, x_t = np.histogram(y_train, args[2], density=False, range=args[3])
	y_p, x_p = np.histogram(y_predict, args[2], density=True, range=args[3])

	for i in range(args[2]):
		idx = (y_predict < x_p[i+1]) * (y_predict > x_p[i])
		m = mean[idx]
		s = std[idx]
		#print(np.sum(np.isnan(s)))
		#print(np.sum(np.isnan(m)))
		marg.append(np.mean(m))
		unc.append(np.sum(s))
		unc2.append(np.mean(s/m))

	marg = np.asarray(marg)
	unc = np.asarray(unc)
	rel_unc = unc/marg
	rel_unc[rel_unc > 1] = 1

	u300 = pd.read_hdf('unc_300.h5').values
	u20 = pd.read_hdf('unc_10.h5').values
	#y300 = pd.read_hdf('y_p_300.h5').values
	#y20 = pd.read_hdf('y_p_20.h5').values

	#axs.step(x_t[:args[2]], rel_unc, '#f03b20', label='Truth', linewidth=1.0, where='mid')
	#axs.step(x_p[:args[2]], unc2, '#2c7fb8', label='Predicted uncertainty', linewidth=1.0, where='mid')
	#axs.step(x_p[:args[2]], y300[50:], '#2c7fb8', label='BINN, 300 epochs', linewidth=1.0, where='mid')
	#axs.step(x_p[:args[2]], y20[50:], teal, label='BINN, 20 epochs', linewidth=1.0, where='mid')
	axs.step(x_p[:args[2]], u300[50:], '#2c7fb8', label='Predicted uncertainty, 300 epochs', linewidth=1.0, where='mid')
	axs.step(x_p[:args[2]], u20[50:], teal, label='Predicted uncertainty, 20 epochs', linewidth=1.0, where='mid')
	axs.step(x_p[:args[2]], 1/np.sqrt(y_t), color='#e41a1c', label='Training statistics', linewidth=1.0, where='mid')

	for label in ( [axs.yaxis.get_offset_text()] +
				  axs.get_xticklabels() + axs.get_yticklabels()):
		label.set_fontsize(FONTSIZE)

	yfmt = ScalarFormatterForceFormat()
	yfmt.set_powerlimits((0,0))
	axs.yaxis.set_major_formatter(yfmt)

	axs.set_ylabel('Relative uncertainty', fontsize = FONTSIZE)
	axs.legend(loc='upper center', prop={'size':(FONTSIZE-5)}, frameon=False)
	axs.set_xlabel(r'$%s$' %(args[5]), fontsize = FONTSIZE)

def save_distribution(y_train, y_predict, args):
	"""Plot the distributions"""
	# Particle_id, observable, bins, range, x_label, log_scale

	y_train = args[1](y_train, args[0])
	y_predict = args[1](y_predict, args[0])

	y_t, x_t = np.histogram(y_train, args[2], density=True, range=args[3])
	y_p, x_p = np.histogram(y_predict, args[2], density=True, range=args[3])
	x_p = x_p[0:args[2]]

	histo = np.vstack((x_p,y_t,y_p)).T

	return histo

def plot_2d_distribution(fig, axs, y_train, y_predict, args1, args2):
	"""Plot the distributions"""
	fontsize=FONTSIZE+2
	data = [[0.,0.], [0.,0.]]
	h = [[0.,0.], [0.,0.]]

	# Fill (x,y) with data
	data[0][0] = args1[1](y_train, args1[0])
	data[1][0] = args2[1](y_train, args2[0])

	data[0][1] = args1[1](y_predict, args1[0])
	data[1][1] = args2[1](y_predict, args2[0])

	h[0][0], xedges, yedges = np.histogram2d(data[0][0], data[1][0], bins=args1[2], range=(args1[3],args2[3]), density=True)
	h[0][1], xedges, yedges = np.histogram2d(data[0][1], data[1][1], bins=args1[2], range=(args1[3],args2[3]), density=True)
	h[1][0] = (h[0][1]-h[0][0])/(h[0][1]+h[0][0])

	h[1][0][np.isnan(h[1][0])==True]=0 #1
	h[1][0][h[1][0]==np.inf]=0 #1
#	 h[1][0][h[1][0]==0]=1

	for j in range(3):
		for label in ( [axs[j].yaxis.get_offset_text()] +
					  axs[j].get_xticklabels() + axs[j].get_yticklabels()):
			label.set_fontsize(fontsize)

	Z = h[0][0].T
	yfmt = ScalarFormatterForceFormat()
	yfmt.set_powerlimits((0,0))
	im = axs[0].pcolormesh(xedges, yedges, Z, rasterized=True)
	divider = make_axes_locatable(axs[0])
	cax = divider.append_axes('right', size='5%', pad=0.05)
	axs[0].set_xlabel(args1[4], fontsize = fontsize)
	axs[0].set_ylabel(args2[4], fontsize = fontsize)
	cbar = fig.colorbar(im, cax=cax, orientation='vertical')
	cbar.ax.tick_params(labelsize=fontsize)
	cbar.ax.yaxis.set_major_formatter(yfmt)

	Z = h[0][1].T
	yfmt = ScalarFormatterForceFormat()
	yfmt.set_powerlimits((0,0))
	im = axs[1].pcolormesh(xedges, yedges, Z, rasterized=True)
	divider = make_axes_locatable(axs[1])
	cax = divider.append_axes('right', size='5%', pad=0.05)
	axs[1].set_xlabel(args1[4], fontsize = fontsize)
	axs[1].set_ylabel(args2[4], fontsize = fontsize)
	cbar = fig.colorbar(im, cax=cax, orientation='vertical')
	cbar.ax.tick_params(labelsize=fontsize)
	cbar.ax.yaxis.set_major_formatter(yfmt)

	yfmt = ScalarFormatterForceFormat()
	yfmt.set_powerlimits((0,0))
	im = axs[2].pcolormesh(xedges, yedges, h[1][0].T, rasterized=True)
	divider = make_axes_locatable(axs[2])
	cax = divider.append_axes('right', size='5%', pad=0.05)
	axs[2].set_xlabel(args1[4], fontsize = fontsize)
	axs[2].set_ylabel(args2[4], fontsize = fontsize)
	cbar = fig.colorbar(im, cax=cax, orientation='vertical')
	cbar.ax.tick_params(labelsize=fontsize)
	cbar.ax.yaxis.set_major_formatter(yfmt)

def plot_2d_distribution_single(fig, axs, y_train, y_predict, args1, args2):

	fontsize = FONTSIZE#+10
	"""Plot the distributions"""
	data = [[0.,0.], [0.,0.]]
	h    = [[0.,0.], [0.,0.]]

	# Fill (x,y) with data
	data[0][0] = args1[1](y_train, args1[0])
	data[1][0] = args2[1](y_train, args2[0])

	data[0][1] = args1[1](y_predict, args1[0])
	data[1][1] = args2[1](y_predict, args2[0])

	h[0][1], xedges, yedges = np.histogram2d(data[0][0], data[1][0], bins=args1[2], range=(args1[3],args2[3]), density=True)
	#h[0][1], xedges, yedges = np.histogram2d(data[0][1], data[1][1], bins=args1[2], range=(args1[3],args2[3]), density=True)
	#h[1][0] = (h[0][1]-h[0][0])/(h[0][1]+h[0][0])

	if args1[6]:
		axs.set_xscale('log')
	if args2[6]:
		axs.set_yscale('log')

	for label in ([axs.yaxis.get_offset_text()] +
				  axs.get_xticklabels() + axs.get_yticklabels()):
		label.set_fontsize(fontsize)
	yfmt = ScalarFormatterForceFormat()
	yfmt.set_powerlimits((0,0))
	im = axs.pcolormesh(xedges, yedges, h[0][1].T, rasterized=True)
	divider = make_axes_locatable(axs)
	cax = divider.append_axes('right', size='5%', pad=0.05)
	axs.set_xlabel(args1[4], fontsize = fontsize)
	axs.set_ylabel(args2[4], fontsize = fontsize)
	cbar = fig.colorbar(im, cax=cax, orientation='vertical')
	cbar.ax.tick_params(labelsize=fontsize)
	cbar.ax.yaxis.set_major_formatter(yfmt)
	for label in ([cbar.ax.yaxis.get_offset_text()]):
		label.set_fontsize(fontsize)

def plot_slice(fig, axs, y_train, y_predict, args1, args2):

	fontsize = FONTSIZE+10

	data1 = args1[1](y_train, args1[0])
	data2 = args1[1](y_predict, args1[0])

	mean = 0.2
	delta = 0.05
	lim_max = mean+delta
	lim_min = mean-delta

	y_train = data2[(data1 < lim_max) * (data1 > lim_min)]

	bins1d = 30

	y_t, x_t = np.histogram(y_train, bins1d, density=True, range=args1[3])
	#y_p, x_p = np.histogram(y_predict, bins1d, density=True, range=args1[3])

	for label in ( [axs.yaxis.get_offset_text()] + axs.get_yticklabels() + axs.get_xticklabels()):
		label.set_fontsize(fontsize)

	axs.text(1.0, 4.0, r'slice at $\mu= $' + str(mean), fontsize = fontsize)
	yfmt = ScalarFormatterForceFormat()
	yfmt.set_powerlimits((0,0))
	axs.yaxis.set_major_formatter(yfmt)
	axs.step(x_t[:bins1d], y_t, '#f03b20', linewidth=1.0, where='pre')
	axs.set_ylabel(r'$\text{Normalized}$', fontsize = fontsize)
	#axs.legend(loc='upper right', prop={'size':(fontsize-2)}, frameon=False)
	axs.set_xlabel(args1[4], fontsize = fontsize)
	#axs.set_ylim((0,0.020))
