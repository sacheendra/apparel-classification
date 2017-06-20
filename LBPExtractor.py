import numpy as np
from skimage import color
from skimage.feature import local_binary_pattern

import load_data

class LBPExtractor(object):
	"""Extracts Local Binary Pattern descriptor 
	from a list of images."""
	def __init__(self, images):
		super(LBPExtractor, self).__init__()

		self.lbp_features = [np.ndarray(0)] * images.shape[0]
		for i in range(len(images)):
			image = color.rgb2gray(images[i])
			lbp_image = local_binary_pattern(image, 8, 2, method='uniform')
			self.lbp_features[i] = np.ndarray.flatten(lbp_image)

	def get_features(self):
		return self.lbp_features

def run():
	il = load_data.ImageLoader('fashion-data')
	test_images = il.get_test_data()

	e = LBPExtractor(test_images)
	lbp_features = e.get_features()
	print(lbp_features[0].shape)
	print(np.shape(lbp_features))

if __name__ == '__main__':
	run()