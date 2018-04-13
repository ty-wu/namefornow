import numpy as np
import pandas as pd


def load_data(fname):
	'''
	Loads the data in file specified by fname. The file specified is a txt file

	Returns X: an nx(d-1) array, where n is the number of examples and d is the dimensionality.    
            Y: an nx1 array, where n is the number of examples
	'''
	data = pd.read_csv(fname)
	# original_data = data.copy()
	# print data.head(1)
	
	# dealing with missing data
	data = data.replace('?', np.nan)
	# map make column
	data['make'] = data['make'].map({'alfa-romero':0, 'audi':1, 'bmw':2, 'chevrolet':3, 'dodge':4, 'honda':5,\
                               'isuzu':6, 'jaguar':7, 'mazda':8, 'mercedes-benz':9, 'mercury':10,\
                               'mitsubishi':11, 'nissan':12, 'peugot':13, 'plymouth':14, 'porsche':15,\
                               'renault':16, 'saab':17, 'subaru':18, 'toyota':19, 'volkswagen':20, 'volvo':21})
	# map fuel-type
	data['fuel-type'] = data['fuel-type'].map({'gas':0, 'diesel':1})
	# map aspiration
	data['aspiration'] = data['aspiration'].map({'std':0, 'turbo':1})
	# map num-of-doors
	data['num-of-doors'] = data['num-of-doors'].map({'four':0, 'two':1})
	# map body-style
	data['body-style'] = data['body-style'].map({'hardtop':0, 'wagon':1, 'sedan':2, 'hatchback':3, \
											'convertible':4})
	# map drive-wheels
	data['drive-wheels'] = data['drive-wheels'].map({'4wd':0, 'fwd':1, 'rwd':2})
	# map engine-location
	data['engine-location'] = data['engine-location'].map({'front':0, 'rear':1})
	# map wheel-base 
	# binning
	small_bin_labels = [i for i in range(5)]
	large_bin_labels = [i for i in range(10)]
	data['wheel-base'] = pd.qcut(data['wheel-base'], 10, labels=large_bin_labels)
	# map length
	data['length'] = pd.qcut(data['length'], 10, labels=large_bin_labels)
	# map width
	data['width'] = pd.qcut(data['width'], 5, labels=small_bin_labels)
	# map height
	data['height'] = pd.qcut(data['height'], 5, labels=small_bin_labels)
	# map curb-weight

	data['curb-weight'] = pd.qcut(data['curb-weight'], 10, labels=large_bin_labels)
	# map engine-type
	data['engine-type'] = data['engine-type'].map({'dohc':0, 'dohcv':1, 'l':2, 'ohc':3, 'ohcf':4, \
												'ohcv':5, 'rotor':6})
	# map num-of-cylinders
	data['num-of-cylinders'] = data['num-of-cylinders'].map({'eight':0, 'five':1, 'four':2, \
														'six':3, 'three':4, 'twelve':5, 'two':6})
	# map engine-size
	data['engine-size'] = pd.qcut(data['engine-size'], 10, labels=large_bin_labels)
	# map fuel-system
	data['fuel-system'] = data['fuel-system'].map({'1bbl':0, '2bbl':1, '4bbl':2, 'idi':3, \
												'mfi':4, 'mpfi':5, 'spdi':6, 'spfi':7})
	# map bore
	data['bore'] = pd.qcut(data['bore'].astype(float), 5, labels=small_bin_labels)
	# map stroke
	data['stroke'] = pd.qcut(data['stroke'].astype(float), 5, labels=small_bin_labels)
	# map compression-ratio
	data['compression-ratio'] = pd.qcut(data['compression-ratio'], 5, labels=small_bin_labels)
	# map horsepower
	data['horsepower'] = pd.qcut(data['horsepower'].astype(float), 10, labels=large_bin_labels)
	# map peak-rpm
	data['peak-rpm'] = pd.qcut(data['peak-rpm'].astype(float), 5, labels=small_bin_labels)
	# map city-mpg
	data['city-mpg'] = pd.qcut(data['city-mpg'].astype(float), 10, labels=large_bin_labels)
	# map highway-mpg
	data['highway-mpg'] = pd.qcut(data['highway-mpg'], 10, labels=large_bin_labels)
	# map price
	data['price'] = pd.qcut(data['price'].astype(float), 10, labels=large_bin_labels)
	
	data = data.drop(['normalized-losses'], axis=1) # only consider specification as feature
	data = data.dropna(axis=0, how='any') # drop the sample if it contains any missing value
	y = np.array(data['symboling'])
	data = data.drop(['symboling'],axis=1)
	X = np.array(data)
	# map the label to binary using threshold=0
	thrs = 0
	for i in range(len(y)):
		if y[i] > 0:
			y[i] = 1
		else:
			y[i] = 0
	print 'pos samples:', np.count_nonzero(y), 'neg examples:', len(y)-np.count_nonzero(y)
	# print X[0],y[0]
	return X,y
