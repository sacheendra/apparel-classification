from load_data import ImageLoader
from feature_extractors import SURFExtractor, HOGExtractor
from vector_quantizer import VectorQuantizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
import numpy as np
import os

loader = ImageLoader('fashion-data', train_images_per_category=100, test_images_per_category=10, num_categories=15)

def spm_tri_level(train_matrix, num_centroids):
	dim = train_matrix.shape[0]
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

def extract_surf_features(dataset_name, input_data, num_centroids=16, step_size=40):
	if os.path.isfile(os.path.join('fashion-data', dataset_name + '-surf-features.npy')):
		print('loading SURF features for '+dataset_name+' from numpy file')
		surf_features = loader.load_data(dataset_name + '-surf-features.npy')
	else:
		print('extracting SURF features for '+dataset_name)
		surf_features = SURFExtractor(input_data, 400, step_size=step_size).get_features()
		loader.store_data(surf_features, dataset_name + '-surf-features.npy')

	return np.ndarray.reshape(surf_features, (surf_features.shape[0], surf_features.shape[1]*surf_features.shape[2]*surf_features.shape[3]))
	# reshaped_surf_features = np.ndarray.reshape(surf_features, (surf_features.shape[0]*surf_features.shape[1]*surf_features.shape[2], surf_features.shape[3]))

	# if os.path.isfile(os.path.join('fashion-data', dataset_name + '-quantized-surf-features.npy')):
	# 	print('loading quantized SURF features for '+dataset_name+' from numpy file')
	# 	quantized_surf = loader.load_data(dataset_name + '-quantized-surf-features.npy')
	# else:
	# 	print('extracting quantized SURF features for '+dataset_name)
	# 	quantized_surf = VectorQuantizer(reshaped_surf_features, num_centroids=num_centroids).get_features()
	# 	loader.store_data(quantized_surf, dataset_name + '-quantized-surf-features.npy')

	# original_shape_quantized_surf = np.reshape(quantized_surf, (surf_features.shape[0], surf_features.shape[1], surf_features.shape[2]))

	# return original_shape_quantized_surf

def extract_quantized_surf_features(dataset_name, input_data, num_centroids=16, step_size=40):
	if os.path.isfile(os.path.join('fashion-data', dataset_name + '-surf-features.npy')):
		print('loading SURF features for '+dataset_name+' from numpy file')
		surf_features = loader.load_data(dataset_name + '-surf-features.npy')
	else:
		print('extracting SURF features for '+dataset_name)
		surf_features = SURFExtractor(input_data, 400, step_size=step_size).get_features()
		loader.store_data(surf_features, dataset_name + '-surf-features.npy')

	# return np.ndarray.reshape(surf_features, (surf_features.shape[0], surf_features.shape[1]*surf_features.shape[2]*surf_features.shape[3]))
	reshaped_surf_features = np.ndarray.reshape(surf_features, (surf_features.shape[0]*surf_features.shape[1]*surf_features.shape[2], surf_features.shape[3]))

	if os.path.isfile(os.path.join('fashion-data', dataset_name + '-quantized-surf-features.npy')):
		print('loading quantized SURF features for '+dataset_name+' from numpy file')
		quantized_surf = loader.load_data(dataset_name + '-quantized-surf-features.npy')
	else:
		print('extracting quantized SURF features for '+dataset_name)
		quantized_surf = VectorQuantizer(reshaped_surf_features, num_centroids=num_centroids).get_features()
		loader.store_data(quantized_surf, dataset_name + '-quantized-surf-features.npy')

	original_shape_quantized_surf = np.reshape(quantized_surf, (surf_features.shape[0], surf_features.shape[1], surf_features.shape[2]))

	return original_shape_quantized_surf

def extract_hog_features(dataset_name, input_data, num_centroids=16, pixels_per_cell=16, cells_per_block=3):
	if os.path.isfile(os.path.join('fashion-data', dataset_name + '-hog-features.npy')):
		print('loading HOG features for '+dataset_name+' from numpy file')
		hog_features = loader.load_data(dataset_name + '-hog-features.npy')
	else:
		print('extracting HOG features for '+dataset_name)
		hog_features = HOGExtractor(input_data, pixels_per_cell=pixels_per_cell, cells_per_block=pixels_per_cell).get_features()
		loader.store_data(hog_features, dataset_name + '-hog-features.npy')

	return hog_features

def extract_quantized_hog_features(dataset_name, input_data, num_centroids=16, pixels_per_cell=16, cells_per_block=3):
	if os.path.isfile(os.path.join('fashion-data', dataset_name + '-hog-features.npy')):
		print('loading HOG features for '+dataset_name+' from numpy file')
		hog_features = loader.load_data(dataset_name + '-hog-features.npy')
	else:
		print('extracting HOG features for '+dataset_name)
		hog_features = HOGExtractor(input_data, pixels_per_cell=pixels_per_cell, cells_per_block=pixels_per_cell).get_features()
		loader.store_data(hog_features, dataset_name + '-hog-features.npy')

	print hog_features.shape
	reshaped_hog_features = np.reshape(hog_features, (hog_features.shape[0]*hog_features.shape[1], 1))

	if os.path.isfile(os.path.join('fashion-data', dataset_name + '-quantized-hog-features.npy')):
		print('loading quantized HOG features for '+dataset_name+' from numpy file')
		quantized_hog = loader.load_data(dataset_name + '-quantized-hog-features.npy')
	else:
		print('extracting quantized HOG features for '+dataset_name)
		quantized_hog = VectorQuantizer(reshaped_hog_features, num_centroids=num_centroids).get_features()
		loader.store_data(quantized_hog, dataset_name + '-quantized-hog-features.npy')

	original_shape_quantized_hog = np.reshape(quantized_hog, hog_features.shape)

	return original_shape_quantized_hog

def main():
	train_data = loader.get_train_data()
	train_labels = loader.get_train_labels()
	test_data = loader.get_test_data()
	test_labels = loader.get_test_labels()

	# potential_step_size = [64, 32, 16]
	potential_step_size = [80, 40, 20]
	potential_num_centroids = [16, 32, 64, 128]
	# potential_step_size = [20]
	# potential_num_centroids = [32]

	print "just SURF"
	j = 0
	for i in range(len(potential_step_size)):
		# for j in range(len(potential_num_centroids)):
			train_features = extract_surf_features('train_'+str(potential_num_centroids[j])+'_'+str(potential_step_size[i]), train_data, num_centroids=potential_num_centroids[j], step_size=potential_step_size[i])
			test_features = extract_surf_features('test_'+str(potential_num_centroids[j])+'_'+str(potential_step_size[i]), test_data, num_centroids=potential_num_centroids[j], step_size=potential_step_size[i])
			# train_histogram = [spm_tri_level(train_features[k], potential_num_centroids[j]) for k in range(train_features.shape[0])]
			# test_histogram = [spm_tri_level(test_features[k], potential_num_centroids[j]) for k in range(test_features.shape[0])]

			accuracy = 0
			for x in xrange(50):
				clf = RandomForestClassifier(n_jobs=2)
				# clf = SVC()
				clf.fit(train_features, train_labels)
				accuracy = (accuracy*x + clf.score(test_features, test_labels))/(x+1)
			print "for step size = {0} and #(centroids) = {1}, the average accuracy = {2}".format(potential_step_size[i], potential_num_centroids[j], accuracy)


	print "SURF with kmeans"
	for i in range(len(potential_step_size)):
		for j in range(len(potential_num_centroids)):
			train_features = extract_quantized_surf_features('train_'+str(potential_num_centroids[j])+'_'+str(potential_step_size[i]), train_data, num_centroids=potential_num_centroids[j], step_size=potential_step_size[i])
			test_features = extract_quantized_surf_features('test_'+str(potential_num_centroids[j])+'_'+str(potential_step_size[i]), test_data, num_centroids=potential_num_centroids[j], step_size=potential_step_size[i])
			# train_histogram = [spm_tri_level(train_features[k], potential_num_centroids[j]) for k in range(train_features.shape[0])]
			# test_histogram = [spm_tri_level(test_features[k], potential_num_centroids[j]) for k in range(test_features.shape[0])]
			reshaped_train_features = np.reshape(train_features, (train_features.shape[0], train_features.shape[1]*train_features.shape[2]))
			reshaped_test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1]*test_features.shape[2]))

			accuracy = 0
			for x in xrange(50):
				clf = RandomForestClassifier(n_jobs=2)
				# clf = SVC()
				clf.fit(reshaped_train_features, train_labels)
				accuracy = (accuracy*x + clf.score(reshaped_test_features, test_labels))/(x+1)
			print "for step size = {0} and #(centroids) = {1}, the average accuracy = {2}".format(potential_step_size[i], potential_num_centroids[j], accuracy)


	print "SURF with histograms"
	for i in range(len(potential_step_size)):
		for j in range(len(potential_num_centroids)):
			train_features = extract_quantized_surf_features('train_'+str(potential_num_centroids[j])+'_'+str(potential_step_size[i]), train_data, num_centroids=potential_num_centroids[j], step_size=potential_step_size[i])
			test_features = extract_quantized_surf_features('test_'+str(potential_num_centroids[j])+'_'+str(potential_step_size[i]), test_data, num_centroids=potential_num_centroids[j], step_size=potential_step_size[i])
			train_histogram = [spm_tri_level(train_features[k], potential_num_centroids[j]) for k in range(train_features.shape[0])]
			test_histogram = [spm_tri_level(test_features[k], potential_num_centroids[j]) for k in range(test_features.shape[0])]

			accuracy = 0
			for x in xrange(50):
				clf = RandomForestClassifier(n_jobs=2)
				# clf = SVC()
				clf.fit(train_histogram, train_labels)
				accuracy = (accuracy*x + clf.score(test_histogram, test_labels))/(x+1)
			print "for step size = {0} and #(centroids) = {1}, the average accuracy = {2}".format(potential_step_size[i], potential_num_centroids[j], accuracy)

	# potential_pixels_per_cell = [16, 8]
	# potential_cells_per_block = [4, 3, 2, 1]
	# potential_num_centroids = [16, 32, 64, 128]

	# k = 0
	# for i in range(len(potential_pixels_per_cell)):
	# 	for j in range(len(potential_cells_per_block)):
	# 		# for k in range(len(potential_num_centroids)):
	# 			train_features = (extract_hog_features('train_'+str(potential_num_centroids[k])+'_'+str(potential_pixels_per_cell[i])+'_'+str(potential_cells_per_block[j]), 
	# 								train_data, num_centroids=potential_num_centroids[k], 
	# 								pixels_per_cell=potential_pixels_per_cell[i], 
	# 								cells_per_block=potential_cells_per_block[j]))
	# 			print(train_features.shape)
	# 			test_features = (extract_hog_features('test_'+str(potential_num_centroids[k])+'_'+str(potential_pixels_per_cell[i])+'_'+str(potential_cells_per_block[j]), 
	# 								test_data, num_centroids=potential_num_centroids[k], 
	# 								pixels_per_cell=potential_pixels_per_cell[i], 
	# 								cells_per_block=potential_cells_per_block[j]))		

	# 			clf = RandomForestClassifier(n_jobs=2)
	# 			clf.fit(train_features, train_labels)
	# 			accuracy = clf.score(test_features, test_labels)
	# 			print "for pixels per cell = {0}, cells per block = {1} and #(centroids) = {2}, the average accuracy = {3}".format(potential_pixels_per_cell[i], potential_cells_per_block[j], potential_num_centroids[k], accuracy)
		
if __name__ == '__main__':
	main()