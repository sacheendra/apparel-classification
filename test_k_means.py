from sklearn.cluster import KMeans
import numpy as np
import cv2
import numpy as np
from load_data import ImageLoader
import matplotlib.pyplot as plt
import skimage.measure
import scipy.cluster.vq as vq

def input_vector_encoder(feature, codebook):
    """
    Input all the local feature of the image
    Pooling (encoding) by codebook and return
    """
    code, _ = vq.vq(feature, codebook)
    word_hist, bin_edges = np.histogram(code, bins=range(codebook.shape[0] + 1), normed=True)
    return word_hist

def build_spatial_pyramid(image, descriptor, level):
    """
    Rebuild the descriptors according to the level of pyramid
    """
    assert 0 <= level <= 2, "Level Error"
    step_size = DSIFT_STEP_SIZE
    # from utils import DSIFT_STEP_SIZE as s
    assert s == step_size, "step_size must equal to DSIFT_STEP_SIZE\
                            in utils.extract_DenseSift_descriptors()"
    # h = image.shape[0] / step_size 	# need to find a generic allocation of h and w
    # w = image.shape[1] / step_size
    h = 12
    w = 18
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

def spatial_pyramid_matching(image, descriptor, codebook, level):
    pyramid = []
    if level == 0:
        pyramid += build_spatial_pyramid(image, descriptor, level=0)
        code = [input_vector_encoder(crop, codebook) for crop in pyramid]
        return np.asarray(code).flatten()
    if level == 1:
        pyramid += build_spatial_pyramid(image, descriptor, level=0)
        pyramid += build_spatial_pyramid(image, descriptor, level=1)
        code = [input_vector_encoder(crop, codebook) for crop in pyramid]
        code_level_0 = 0.5 * np.asarray(code[0]).flatten()
        code_level_1 = 0.5 * np.asarray(code[1:]).flatten()
        return np.concatenate((code_level_0, code_level_1))
    if level == 2:
        pyramid += build_spatial_pyramid(image, descriptor, level=0)
        pyramid += build_spatial_pyramid(image, descriptor, level=1)
        pyramid += build_spatial_pyramid(image, descriptor, level=2)
        code = [input_vector_encoder(crop, codebook) for crop in pyramid]
        code_level_0 = 0.25 * np.asarray(code[0]).flatten()
        code_level_1 = 0.25 * np.asarray(code[1:5]).flatten()
        code_level_2 = 0.5 * np.asarray(code[5:]).flatten()
        return np.concatenate((code_level_0, code_level_1, code_level_2))


loader = ImageLoader('fashion-data')
surf = cv2.SURF(400)
train_data = loader.get_train_data()

codebook = []
descriptor = []
step_size = 4
level = 0
VOC_SIZE = 100
PYRAMID_LEVEL = 1

DSIFT_STEP_SIZE = 4
s = 4

for i in range(1):	# replace 1 with len(train_data) for the entire training set
	(surf_keypoints, surf_descriptors) = surf.detectAndCompute(train_data[i],None)
	kmeans = KMeans(n_clusters=8).fit(surf_descriptors)
	codebook.append(np.asarray(kmeans.cluster_centers_[kmeans.labels_]))
	descriptor.append(np.asarray(surf_descriptors))

x_train = [spatial_pyramid_matching(train_data[i],
                                        descriptor[i],
                                        codebook[i],
                                        level=PYRAMID_LEVEL)
                                        for i in range(1)]

print x_train[0].shape