import numpy as np
import h5py
from scipy import signal as sg


# min-max scaling
def min_max_scaling(data):
    min_values = np.amin(data, axis=0)
    max_values = np.amax(data, axis=0)
    return (data - min_values) / (max_values - min_values)


# normalizing
def normalize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std


def load_dataset(filename):
    """
    Load your 'PPG to blood pressure' dataset
    """
    window = 1000
    instances = 10000
    index = 0

    X = np.ndarray(shape=(instances, window), dtype=float)
    Y = np.ndarray(shape=(instances, 2), dtype=float)

    with h5py.File(filename, 'r') as file:
        data = file['Part_1']
        data_length = data.shape[0]

        i = 0
        while i < data_length and index < instances:
            sample_length = int(file[data[i, 0]][()].shape[0] / window)
            ppg = file[data[i, 0]][()][:, 0]
            abp = file[data[i, 0]][()][:, 1]

            j = 0
            while j < sample_length and index < instances:
                X[index] = ppg[j * window: (j + 1) * window]

                abp_max_peak_index = sg.find_peaks_cwt(abp[j * window: (j + 1) * window], np.arange(40, 50))
                Y[index][0] = np.mean(abp[j * window: (j + 1) * window][abp_max_peak_index])

                abp_min_peak_index = sg.find_peaks_cwt(1.0 / abp[j * window: (j + 1) * window], np.arange(40, 50))
                Y[index][1] = np.mean(abp[j * window: (j + 1) * window][abp_min_peak_index])

                index += 1
                j += 1
            i += 1

        #X = normalize(X)
        X = min_max_scaling(X)
        return X, Y
