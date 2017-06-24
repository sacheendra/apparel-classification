import cv2
import numpy as np
from load_data import ImageLoader
from feature_extractors import HOGExtractor
from LABExtractor import LABExtractor
from LBPExtractor import LBPExtractor
from sklearn.cluster import KMeans
import scipy as sc
import scipy.cluster.vq as vq

loader = ImageLoader('fashion-data')
train_data = loader.get_train_data()
level = 0

def input_vector_encoder(feature, codebook):
    """
    Input all the local feature of the image
    Pooling (encoding) by codebook and return
    """
    code, _ = vq.vq(feature, codebook)
    word_hist, bin_edges = np.histogram(code, bins=range(codebook.shape[0] + 1), normed=True)
    return word_hist

def build_codebook(X, voc_size):
    """
    Inupt a list of feature descriptors
    voc_size is the "K" in K-means, k is also called vocabulary size
    Return the codebook/dictionary
    """
    features = np.vstack((X[i] for i in range(len(X))))
    # print features.shape
    kmeans = KMeans(n_clusters=voc_size)
    kmeans.fit(features)
    codebook = kmeans.cluster_centers_.squeeze()
    return codebook

def build_spatial_pyramid(image_shape, descriptor, h, w, level):
    """
    Rebuild the descriptors according to the level of pyramid
    """
    assert 0 <= level <= 2, "Level Error"
    idx_crop = np.array(range(len(descriptor))).reshape(h,w)
    size = idx_crop.itemsize
    height, width = idx_crop.shape
    bh, bw = 2**(3-level), 2**(3-level)
    shape = (height/bh, width/bw, bh, bw)
    strides = size * np.array([width*bh, bw, width, 1])
    crops = np.lib.stride_tricks.as_strided(
            idx_crop, shape=shape, strides=strides)
    des_idxs = [col_block.flatten().tolist() for row_block in crops
                for col_block in row_block]
    pyramid = []
    for idxs in des_idxs:
        pyramid.append(np.asarray([descriptor[idx] for idx in idxs]))
    return pyramid

def spatial_pyramid_matching(image_shape, descriptor, codebook, h, w, level):
    pyramid = []
    if level == 0:
        pyramid += build_spatial_pyramid(image_shape, descriptor, h, w, level=0)
        code = [input_vector_encoder(crop, codebook) for crop in pyramid]
        return np.asarray(code).flatten()
    if level == 1:
        pyramid += build_spatial_pyramid(image_shape, descriptor, h, w, level=0)
        pyramid += build_spatial_pyramid(image_shape, descriptor, h, w, level=1)
        code = [input_vector_encoder(crop, codebook) for crop in pyramid]
        code_level_0 = 0.5 * np.asarray(code[0]).flatten()
        code_level_1 = 0.5 * np.asarray(code[1:]).flatten()
        return np.concatenate((code_level_0, code_level_1))
    if level == 2:
        pyramid += build_spatial_pyramid(image_shape, descriptor, h, w, level=0)
        pyramid += build_spatial_pyramid(image_shape, descriptor, h, w, level=1)
        pyramid += build_spatial_pyramid(image_shape, descriptor, h, w, level=2)
        code = [input_vector_encoder(crop, codebook) for crop in pyramid]
        code_level_0 = 0.25 * np.asarray(code[0]).flatten()
        code_level_1 = 0.25 * np.asarray(code[1:5]).flatten()
        code_level_2 = 0.5 * np.asarray(code[5:]).flatten()
        return np.concatenate((code_level_0, code_level_1, code_level_2))

def extract_hog_descriptors(image):
	"""
	Extracting dense HOG features
	"""
	winSize = (64,64)
	blockSize = (64,64)
	blockStride = (8,8)
	cellSize = (32,32)
	nbins = 4
	derivAperture = 1
	winSigma = 4.
	histogramNormType = 0
	L2HysThreshold = 2.0000000000000001e-01
	gammaCorrection = 0
	nlevels = 64
	hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
	                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
	winStride = (8,8)
	padding = (8,8)
	locations = ((160,160),)
	descriptors = hog.compute(train_data[0],winStride,padding,locations)
	# print descriptors.shape
	return descriptors

def extract_surf_descriptors(image, DSIFT_STEP_SIZE):
	"""
	Extracting dense SURF features
	"""
	surf = cv2.SURF(400)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	disft_step_size = DSIFT_STEP_SIZE
	keypoints = [cv2.KeyPoint(x, y, disft_step_size) for y in range(0, gray.shape[0], disft_step_size) for x in range(0, gray.shape[1], disft_step_size)]
	keypoints, descriptors = surf.compute(gray, keypoints)
	return descriptors

def main():
	# # SURF feature
	VOC_SIZE = 1024
	DSIFT_STEP_SIZE = 64
	h = train_data[0].shape[0] / DSIFT_STEP_SIZE
	w = train_data[0].shape[1] / DSIFT_STEP_SIZE
	surf_train_feature = [extract_surf_descriptors(image, DSIFT_STEP_SIZE) for image in train_data]
	codebook = build_codebook(surf_train_feature, VOC_SIZE)
	print codebook.shape
	# Pyramid level 0
	PYRAMID_LEVEL = 0
	surf_train = [spatial_pyramid_matching(train_data[i].shape, surf_train_feature[i], codebook, h, w, level=PYRAMID_LEVEL) for i in range(len(train_data))]
	surf_level_0 = [np.reshape(surf_train[i], (len(surf_train[0]), 1)) for i in range(len(surf_train))]
	final_array = np.array(surf_level_0)
	np.savez('fashion-data\surf0.npz', *final_array)
	# Pyramid level 1
	PYRAMID_LEVEL = 1
	surf_train = [spatial_pyramid_matching(train_data[i].shape, surf_train_feature[i], codebook, h, w, level=PYRAMID_LEVEL) for i in range(len(train_data))]
	surf_level_1 = [np.reshape(surf_train[i], (len(surf_train[0]), 1)) for i in range(len(surf_train))]
	final_array = np.array(surf_level_1)
	np.savez('fashion-data\surf1.npz', *final_array)
	# Pyramid level 2
	PYRAMID_LEVEL = 2
	surf_train = [spatial_pyramid_matching(train_data[i].shape, surf_train_feature[i], codebook, h, w, level=PYRAMID_LEVEL) for i in range(len(train_data))]
	surf_level_2 = [np.reshape(surf_train[i], (len(surf_train[0]), 1)) for i in range(len(surf_train))]
	final_array = np.array(surf_level_2)
	np.savez('fashion-data\surf2.npz', *final_array)
	# HOG feature
	VOC_SIZE = 1024
	DSIFT_STEP_SIZE = 80
	h = train_data[0].shape[0] / DSIFT_STEP_SIZE
	w = train_data[0].shape[1] / DSIFT_STEP_SIZE
	hog_train_feature = [extract_hog_descriptors(image) for image in train_data]
	codebook = build_codebook(hog_train_feature, VOC_SIZE)
	codebook = np.reshape(codebook, (len(codebook), 1))
	print codebook.shape
	# Pyramid level 0
	PYRAMID_LEVEL = 0
	hog_train = [spatial_pyramid_matching(train_data[i].shape, hog_train_feature[i], codebook, h, w, level=PYRAMID_LEVEL) for i in range(len(train_data))]
	hog_level_0 = [np.reshape(hog_train[i], (len(hog_train[0]), 1)) for i in range(len(hog_train))]
	final_array = np.array(hog_level_0)
	np.savez('fashion-data\hog0.npz', *final_array)
	# Pyramid level 1
	PYRAMID_LEVEL = 1
	hog_train = [spatial_pyramid_matching(train_data[i].shape, hog_train_feature[i], codebook, h, w, level=PYRAMID_LEVEL) for i in range(len(train_data))]
	hog_level_1 = [np.reshape(hog_train[i], (len(hog_train[0]), 1)) for i in range(len(hog_train))]
	final_array = np.array(hog_level_1)
	np.savez('fashion-data\hog1.npz', *final_array)
	# Pyramid level 2
	PYRAMID_LEVEL = 2
	hog_train = [spatial_pyramid_matching(train_data[i].shape, hog_train_feature[i], codebook, h, w, level=PYRAMID_LEVEL) for i in range(len(train_data))]
	hog_level_2 = [np.reshape(hog_train[i], (len(hog_train[0]), 1)) for i in range(len(hog_train))]
	final_array = np.array(hog_level_2)
	np.savez('fashion-data\hog2.npz', *final_array)
	# LAB feature
	VOC_SIZE = 128
	DSIFT_STEP_SIZE = 80
	h = train_data[0].shape[0] / DSIFT_STEP_SIZE
	w = (train_data[0].shape[1] * 3) / (DSIFT_STEP_SIZE * 2)
	train_feature = LABExtractor(train_data).get_features()
	lab_train_feature = [np.reshape(train_feature[i], (len(train_feature[0]), 1)) for i in range(len(train_feature))]
	codebook = build_codebook(lab_train_feature, VOC_SIZE)
	codebook = np.reshape(codebook, (len(codebook), 1))
	print codebook.shape
	# Pyramid level 0
	PYRAMID_LEVEL = 0
	lab_train = [spatial_pyramid_matching(train_data[i].shape, lab_train_feature[i], codebook, h, w, level=PYRAMID_LEVEL) for i in range(len(train_data))]
	lab_level_0 = [np.reshape(lab_train[i], (len(lab_train[0]), 1)) for i in range(len(lab_train))]
	final_array = np.array(lab_level_0)
	np.savez('fashion-data\lab0.npz', *final_array)
	# Pyramid level 1
	PYRAMID_LEVEL = 1
	lab_train = [spatial_pyramid_matching(train_data[i].shape, lab_train_feature[i], codebook, h, w, level=PYRAMID_LEVEL) for i in range(len(train_data))]
	lab_level_1 = [np.reshape(lab_train[i], (len(lab_train[0]), 1)) for i in range(len(lab_train))]
	final_array = np.array(lab_level_1)
	np.savez('fashion-data\lab1.npz', *final_array)
	# Pyramid level 2
	PYRAMID_LEVEL = 2
	lab_train = [spatial_pyramid_matching(train_data[i].shape, lab_train_feature[i], codebook, h, w, level=PYRAMID_LEVEL) for i in range(len(train_data))]
	lab_level_2 = [np.reshape(lab_train[i], (len(lab_train[0]), 1)) for i in range(len(lab_train))]
	final_array = np.array(lab_level_2)
	np.savez('fashion-data\lab2.npz', *final_array)
	#LBP feature
	train_feature = LBPExtractor(train_data).get_features()
	lbp_train_feature = [np.reshape(train_feature[i], (len(train_feature[0]), 1)) for i in range(len(train_feature))]
	final_array = np.array(lbp_train_feature)
	np.savez('fashion-data\lbp.npz', *final_array)
	print final_array.shape

if __name__ == '__main__':
	main()