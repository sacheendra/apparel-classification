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

def right_approach(step_size, num_centroids):	# with poorer accuracy (ironically)
	print('extracting SURF features for training data')
	surf_features = SURFExtractor(train_data, 400, step_size).get_features()
	np.save('fashion-data\surf-features-train.npy', surf_features)

	reshaped_surf_features = np.ndarray.reshape(surf_features, (surf_features.shape[0]*surf_features.shape[1]*surf_features.shape[2], surf_features.shape[3]))

	print('extracting quantized SURF features for training data')
	quantized_surf = VectorQuantizer(reshaped_surf_features, num_centroids).get_features()
	np.save('quantized-surf-features-train.npy', quantized_surf)

	tuple_0 = np.reshape(quantized_surf[0], (surf_features.shape[0], surf_features.shape[1]*surf_features.shape[2]))
	tuple_1 = np.reshape(quantized_surf[1], (surf_features.shape[0], surf_features.shape[1]*surf_features.shape[2]))
	# print tuple_0.shape
	# print tuple_1.shape
	original_shape_quantized_surf = np.concatenate((tuple_0, tuple_1), axis = 1)
	# print original_shape_quantized_surf.shape

	np.save('fashion-data\surf_train.npy', original_shape_quantized_surf)

	print('extracting SURF features for testing data')
	surf_features = SURFExtractor(test_data, 400, step_size).get_features()
	np.save('fashion-data\surf-features-test.npy', surf_features)

	reshaped_surf_features = np.ndarray.reshape(surf_features, (surf_features.shape[0]*surf_features.shape[1]*surf_features.shape[2], surf_features.shape[3]))

	print('extracting quantized SURF features for testing data')
	quantized_surf = VectorQuantizer(reshaped_surf_features, num_centroids).get_features()
	np.save('quantized-surf-features-test.npy', quantized_surf)

	tuple_0 = np.reshape(quantized_surf[0], (surf_features.shape[0], surf_features.shape[1]*surf_features.shape[2]))
	tuple_1 = np.reshape(quantized_surf[1], (surf_features.shape[0], surf_features.shape[1]*surf_features.shape[2]))
	# print tuple_0.shape
	# print tuple_1.shape
	original_shape_quantized_surf = np.concatenate((tuple_0, tuple_1), axis = 1)
	# print original_shape_quantized_surf.shape

	np.save('fashion-data\surf_test.npy', original_shape_quantized_surf)

def wrong_approach(step_size, num_centroids):	# with nice accuracy at step size = 64 or 80
	print('extracting SURF features for training data')
	surf_features = SURFExtractor(train_data, 400, step_size).get_features()
	np.save('fashion-data\surf-features-train.npy', surf_features)

	reshaped_surf_features = np.ndarray.reshape(surf_features, (surf_features.shape[0]*surf_features.shape[1]*surf_features.shape[2], surf_features.shape[3]))

	print('extracting quantized SURF features for training data')
	quantized_surf = VectorQuantizer(reshaped_surf_features, num_centroids).get_features()
	np.save('quantized-surf-features-train.npy', quantized_surf)

	original_shape_quantized_surf = np.reshape(quantized_surf, (surf_features.shape[0], 2*surf_features.shape[1]*surf_features.shape[2]))
	# print original_shape_quantized_surf.shape

	np.save('fashion-data\surf_train.npy', original_shape_quantized_surf)

	print('extracting SURF features for testing data')
	surf_features = SURFExtractor(test_data, 400, step_size).get_features()
	np.save('fashion-data\surf-features-test.npy', surf_features)

	reshaped_surf_features = np.ndarray.reshape(surf_features, (surf_features.shape[0]*surf_features.shape[1]*surf_features.shape[2], surf_features.shape[3]))

	print('extracting quantized SURF features for testing data')
	quantized_surf = VectorQuantizer(reshaped_surf_features, num_centroids).get_features()
	np.save('quantized-surf-features-test.npy', quantized_surf)

	original_shape_quantized_surf = np.reshape(quantized_surf, (surf_features.shape[0], 2*surf_features.shape[1]*surf_features.shape[2]))
	# print original_shape_quantized_surf.shape

	np.save('fashion-data\surf_test.npy', original_shape_quantized_surf)

def perform_random_forest(num_iter):
	train_val = np.load('fashion-data\surf_train.npy')
	test_val = np.load('fashion-data\surf_test.npy')
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

def main():
	potential_step_size = [80, 64, 40, 32, 20, 10, 8]
	potential_num_centroids = [8, 16, 32, 64, 128]

	for i in range(len(potential_step_size)):
		for j in range(len(potential_num_centroids)):
			right_approach(potential_step_size[i], potential_num_centroids[j])
			accuracy = perform_random_forest(num_iter = 50)
			print "Right approach: for step size = {0} and #(centroids) = {1}, the average accuracy = {2}".format(potential_step_size[i], potential_num_centroids[j], accuracy)

			wrong_approach(potential_step_size[i], potential_num_centroids[j])
			accuracy = perform_random_forest(num_iter = 50)
			print "Wrong approach: for step size = {0} and #(centroids) = {1}, the average accuracy = {2}".format(potential_step_size[i], potential_num_centroids[j], accuracy)
		
if __name__ == '__main__':
	main()