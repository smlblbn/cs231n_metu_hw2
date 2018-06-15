import scipy.io as sio # you may use scipy for loading your data
import cPickle as pickle # or Pickle library
import numpy as np
import os
from scipy import signal as sg

def normalize(data):
	mean = np.mean(data, axis=0)
	std = np.std(data, axis=0)
	return (data - mean)/std

def load_dataset(filename):
 	"""Load your 'PPG to blood pressure' dataset"""
	# TODO: Fill this function so that your version of the data is loaded from a file into vectors
	T = sio.loadmat(filename)
	T = T['Part_1'][0]

	window = 1000
	instances = 10000
	index = 0
	i = 0
	j = 0
	sum_ind = 0
	X = np.ndarray(shape=(instances, window), dtype=float)
	Y = np.ndarray(shape=(instances, 2), dtype=float)
	while index < instances:
		length = T[i].shape[1] / window
		ppg = T[i][0]
		abp = T[i][1]
		j = 0
		
		while j < length and index < instances:
			X[index] = ppg[j*window : (j+1)*window]
			
			abp_max_peak_index = sg.find_peaks_cwt(abp[j*window : (j+1)*window], np.arange(40, 50))
			Y[index][0] = np.mean(abp[j*window : (j+1)*window][abp_max_peak_index])

			abp_min_peak_index = sg.find_peaks_cwt(1.0 / abp[j*window : (j+1)*window], np.arange(40, 50))
			Y[index][1] = np.mean(abp[j*window : (j+1)*window][abp_min_peak_index])

			index+=1
			j+=1
		i+=1
	
	X = normalize(X)
	return X, Y


if __name__ == '__main__':
	# TODO: You can fill in the following part to test your function(s)/dataset from the command line
	filename='metu/dataset/part1_dataset.mat'
	X, Y = load_dataset(filename)