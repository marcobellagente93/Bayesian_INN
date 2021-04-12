from utils.train_utils import *
from utils.observables import *

import os
import pandas as pd

def read_files(DATAPATH, dataset, verbose=True):
	"""Read the files based on whether the dataset exists in .txt or .h5 files
	DATAPATH: folder with the dataset files"""
	events = []
	for file in os.listdir(DATAPATH):
		if dataset in file:
			if verbose:
				print("Reading data from {}".format(file))
				events = pd.read_hdf(os.path.join(DATAPATH, file)).values

	return events

def Loader(dataset, batch_size, test, scaler, on_shell, mom_cons, weighted):

	datapath = './data/'
	data = read_files(datapath, dataset)	

	if test == True:
		split = int(len(data) * 0.01)
	else:	
		split = int(len(data) * 0.9)

	validate_split = int(len(data) * 0.95)

	events=data
	if on_shell:
		events = remove_energies(events)
	if mom_cons:
		events = remove_momenta(events)

	"""Select a single global scale or one for each direction""" 
	#scales = np.std(events)
	scales = np.std(events,0)

	events_train = events[:split]
	events_validate = events[validate_split:]

	shape = events_train.shape[1]
	print(events_train.shape)

	"""Prepare train and validate data loaders"""
	train_loader = torch.utils.data.DataLoader(
			torch.from_numpy(events_train).to(device),
			batch_size = batch_size,
			shuffle = True,
			drop_last = True,
			)
	
	validate_loader = torch.utils.data.DataLoader(
			torch.from_numpy(events_validate).to(device),
			batch_size = batch_size,
			shuffle = False,
			drop_last = True,
			)

	return train_loader, validate_loader, split, shape, scales
