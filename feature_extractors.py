import cv2
import numpy as np
from load_data import ImageLoader
from skimage.feature import hog
from skimage import data, color, exposure

class SURFExtractor(object):
	"""get SURF features"""
	def __init__(self, image_list, hessian_threshold):
		super(SURFExtractor, self).__init__()
		surf = cv2.SURF(hessian_threshold)

		self.surf_features = [np.ndarray(0)] * image_list.shape[0]
		for i in range(len(image_list)):
			dense = cv2.FeatureDetector_create("Dense")
			kp = dense.detect(image_list[i])
			(_, surf_descriptors) = surf.compute(image_list[i],kp)

			self.surf_features[i] = surf_descriptors

	def get_features(self):
		return self.surf_features

class HOGExtractor(object):
	"""get HOG features"""
	def __init__(self, image_list):
		super(HOGExtractor, self).__init__()

		self.hog_features = [np.ndarray(0)] * image_list.shape[0]
		for i in range(len(image_list)):
			grayscale_image = color.rgb2gray(image_list[i])
			hog_feature_array = hog(grayscale_image, block_norm='L2-Hys')

			self.hog_features[i] = np.ndarray.flatten(hog_feature_array)

	def get_features(self):
		return self.hog_features

class DimensionNormaliser(object):
	"""make all images have same bumber of dimensions"""
	def __init__(self, image_list):
		super(DimensionNormaliser, self).__init__()
		max_dimensions = max(image_list, key=lambda x: x.shape[0]).shape[0]
		self.feature_matrix = np.zeros(shape=(len(image_list), max_dimensions), dtype=np.float32)
		for i in range(len(image_list)):
			self.feature_matrix[i, :image_list[i].shape[0]] = image_list[i]

	def get_features(self):
		return self.feature_matrix

def main():
	loader = ImageLoader('fashion-data')
	train_data = loader.get_train_data()
	surf_features = SURFExtractor(train_data, 400).get_features()
	print(len(surf_features))
	print(surf_features[3].shape)
	# dim_normalised_surf_features = DimensionNormaliser(surf_features).get_features()
	# print(dim_normalised_surf_features.shape)
	# hog_features = HOGExtractor(train_data).get_features()
	# print(len(hog_features))
	# print(hog_features[0].shape)
		
if __name__ == '__main__':
	main()