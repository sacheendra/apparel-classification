import cv2
import numpy as np
from load_data import ImageLoader
from skimage.feature import hog
from skimage import data, color, exposure

class SURFExtractor(object):
	"""get SURF features"""
	def __init__(self, image_list, hessian_threshold, step_size=10):
		super(SURFExtractor, self).__init__()

		if hasattr(cv2, 'SURF'):
			surf = cv2.SURF(hessian_threshold)
		else:
			surf = cv2.xfeatures2d.SURF_create()

		self.keypoints = None
		self.surf_features = None
		for i in range(image_list.shape[0]):
			gray = cv2.cvtColor(image_list[i][:][:][:], cv2.COLOR_BGR2GRAY)

			if self.keypoints is None:
				self.keypoints = ([cv2.KeyPoint(x, y, step_size) 
									for y in range(0, gray.shape[0], step_size) 
									for x in range(0, gray.shape[1], step_size)])

			(kps, surf_descriptors) = surf.compute(image_list[i][:][:][:], self.keypoints)

			if self.surf_features is None:
				self.surf_features = np.zeros((image_list.shape[0], gray.shape[0]/step_size, gray.shape[1]/step_size, surf_descriptors.shape[1]))

			for j, kp in enumerate(kps):
				self.surf_features[i][int(kp.pt[0]/step_size)][int(kp.pt[1]/step_size)][:] = surf_descriptors[j][:]

	def get_features(self):
		return self.surf_features

class HOGExtractor(object):
	"""get HOG features"""
	def __init__(self, image_list):
		super(HOGExtractor, self).__init__()

		self.hog_features = None
		for i in range(image_list.shape[0]):
			grayscale_image = color.rgb2gray(image_list[i])
			hog_feature_array = hog(grayscale_image, block_norm='L2-Hys')

			if self.hog_features is None:
				self.hog_features = np.zeros((image_list.shape[0], hog_feature_array.shape[0]))

			self.hog_features[i][:] = np.flatten(hog_feature_array)

	def get_features(self):
		return self.hog_features

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