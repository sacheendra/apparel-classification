from load_data import ImageLoader
from feature_extractors import SURFExtractor
from scipy.cluster import vq
import os
import numpy as np

class VectorQuantizer(object):
	"""vector quantize features"""
	def __init__(self, imagelist, num_centroids=128):
		super(VectorQuantizer, self).__init__()
		whitened = vq.whiten(imagelist)
		(codebook, _) = vq.kmeans2(whitened, num_centroids, iter=1000, minit='points')
		self.encoded_features = vq.vq(whitened, codebook)

	def get_features(self):
		return self.encoded_features

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

	print(surf_features.shape)
	print('vector quantizing SURF features for training data')
	encoded_surf_features = VectorQuantizer(surf_features).get_features()

	print(encoded_surf_features.shape)

if __name__ == '__main__':
	main()