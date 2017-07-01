from load_data import ImageLoader
from feature_extractors import SURFExtractor, HOGExtractor
from vector_quantizer import VectorQuantizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os

loader = ImageLoader('fashion-data')
train_data = loader.get_train_data()
test_data = loader.get_test_data()
line = loader.get_train_labels()
actual_test_labels = loader.get_test_labels()

def calculate_features(train_data, step_size, num_centroids):	# with poorer accuracy (ironically)
	print('extracting SURF features for training data')
	surf_features = SURFExtractor(train_data, 400, step_size).get_features()

	reshaped_surf_features = np.ndarray.reshape(surf_features, (surf_features.shape[0]*surf_features.shape[1]*surf_features.shape[2], surf_features.shape[3]))

	print('extracting quantized SURF features for training data')
	quantized_surf = VectorQuantizer(reshaped_surf_features, num_centroids).get_features()

	original_shape_quantized_surf = np.reshape(quantized_surf[0], (surf_features.shape[0], surf_features.shape[1]*surf_features.shape[2]))
	return original_shape_quantized_surf

def perform_random_forest(train_val, test_val, num_iter):
	rec_accuracy = [0] * num_iter
	for j in range(num_iter):	# you can change the number of iterations if you want to
		score = 0
		clf = RandomForestClassifier(n_jobs=2)
		clf.fit(train_val, line)
		# Done with training the classifier
		result = clf.predict(test_val)
		for i in range(len(result)):
			score += np.sum(actual_test_labels[i] == result[i])
		rec_accuracy[j] = (score * 100)/(len(actual_test_labels))
	accuracy = sum(rec_accuracy)/len(rec_accuracy)
	return accuracy

def spm_tri_level(train_matrix, num_centroids):
	dim = len(train_matrix)
	dim1 = dim/2
	dim2 = dim1/2
	# level 0 of the pyramid
	level_0 = np.zeros(num_centroids)
	for i in range(dim):
		for j in range(dim):
			level_0[train_matrix[i][j]] += 1
	level_hist = 0.25 * level_0
	# level 1 of the pyramid
	train_matrix = np.reshape(train_matrix,(4,dim1,dim1))
	for k in range(4):
		level_0 = np.zeros(num_centroids)
		for i in range(dim1):
			for j in range(dim1):
				level_0[train_matrix[k][i][j]] += 1
		level_0 = 0.25 * level_0
		level_hist = np.append(level_hist, level_0)
	# level 2 of the pyramid
	train_matrix = np.reshape(train_matrix,(16,dim2,dim2))
	for k in range(16):
		level_0 = np.zeros(num_centroids)
		for i in range(dim2):
			for j in range(dim2):
				level_0[train_matrix[k][i][j]] += 1
		level_0 = 0.5 * level_0
		level_hist = np.append(level_hist, level_0)
	return level_hist

def main():
	potential_step_size = [40]
	potential_num_centroids = [32]

	for i in range(len(potential_step_size)):
		for j in range(len(potential_num_centroids)):
			DSIFT_STEP_SIZE = 320/potential_step_size[i]
			train_features = calculate_features(train_data, potential_step_size[i], potential_num_centroids[j])
			test_features = calculate_features(test_data, potential_step_size[i], potential_num_centroids[j])
			x_train = [np.reshape(train_features[k], (DSIFT_STEP_SIZE, DSIFT_STEP_SIZE)) for k in range(len(train_data))]
			x_test = [np.reshape(test_features[k], (DSIFT_STEP_SIZE, DSIFT_STEP_SIZE)) for k in range(len(test_data))]
			train_histogram = [spm_tri_level(x_train[k], potential_num_centroids[j]) for k in range(len(train_data))]
			test_histogram = [spm_tri_level(x_test[k], potential_num_centroids[j]) for k in range(len(test_data))]
			accuracy = perform_random_forest(train_histogram, test_histogram, num_iter = 50)
			print accuracy

if __name__ == '__main__':
	main()