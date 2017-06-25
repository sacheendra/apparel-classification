from load_data import ImageLoader
from feature_extractors import SURFExtractor, HOGExtractor
from vector_quantizer import VectorQuantizer
import numpy as np
import os

def main():
	loader = ImageLoader('fashion-data')
	train_data = loader.get_train_data()

	if os.path.isfile(os.path.join('fashion-data', 'surf-features.npy')):
		print('loading SURF features for training data from numpy file')
		surf_features = loader.load_data('surf-features.npy')
	else:
		print('extracting SURF features for training data')
		surf_features = SURFExtractor(train_data, 400).get_features()
		loader.store_data(surf_features, 'surf-features.npy')

	reshaped_surf_features = np.ndarray.reshape(surf_features, (surf_features.shape[0]*surf_features.shape[1]*surf_features.shape[2], surf_features.shape[3]))

	if os.path.isfile(os.path.join('fashion-data', 'quantized-surf-features.npy')):
		print('loading quantized SURF features for training data from numpy file')
		quantized_surf = loader.load_data('quantized-surf-features.npy')
	else:
		print('extracting quantized SURF features for training data')
		quantized_surf = VectorQuantizer(reshaped_surf_features, num_centroids=1024).get_features()
		loader.store_data(quantized_surf, 'quantized-surf-features.npy')

	original_shape_quantized_surf = np.reshape(quantized_surf, surf_features.shape[:-1])

	if os.path.isfile(os.path.join('fashion-data', 'hog-features.npy')):
		print('loading HOG features for training data from numpy file')
		hog_features = loader.load_data('hog-features.npy')
	else:
		print('extracting HOG features for training data')
		hog_features = HOGExtractor(train_data).get_features()
		loader.store_data(surf_features, 'hog-features.npy')

	reshaped_hog_features = np.ndarray.flatten(hog_features)

	if os.path.isfile(os.path.join('fashion-data', 'quantized-hog-features.npy')):
		print('loading quantized HOG features for training data from numpy file')
		quantized_hog = loader.load_data('quantized-hog-features.npy')
	else:
		print('extracting quantized HOG features for training data')
		quantized_hog = VectorQuantizer(reshaped_hog_features, num_centroids=1024).get_features()
		loader.store_data(quantized_hog, 'quantized-hog-features.npy')

	original_shape_quantized_hog = np.reshape(quantized_hog, hog_features.shape)

	train_features = (np.zeros((train_data.shape[0], quantized_surf.shape[1]*quantized_surf.shape[2],
							quantized_surf.shape[3] + quantized_hog.shape[3])))

	train_features[:][:][:quantized_surf.shape[3]] = np.reshape(quantized_surf.shape[:][:][:][:], (train_data.shape[0], quantized_surf.shape[1]*quantized_surf.shape[2],
							quantized_surf.shape[3]))
	train_features[:][:][quantized_surf.shape[3]:] = np.reshape(quantized_surf.shape[:][:][:][:], (train_data.shape[0], quantized_surf.shape[1]*quantized_surf.shape[2],
							quantized_hog.shape[3]))

		
if __name__ == '__main__':
	main()