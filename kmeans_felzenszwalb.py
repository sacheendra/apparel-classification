from skimage.segmentation import felzenszwalb
from skimage.segmentation import mark_boundaries
from skimage import img_as_float
from sklearn.cluster import KMeans
from colors.colors import load_colors
import matplotlib.pyplot as plt
import numpy as np
import load_data
import operator
import math


def get_felzenswalb_segment(image):
	# Get segments
	img = img_as_float(image)
	segments_fz = felzenszwalb(img, scale=100, sigma=0.93, min_size=200)
	
	# Find largest segment
	counts = {}
	flat = segments_fz.flatten()
	for i in range(flat.shape[0]):
		if flat[i] in counts:
			counts[flat[i]] += 1
		else:
			counts[flat[i]] = 1
	segment = max(counts.iteritems(), key=operator.itemgetter(1))[0]

	return segment, counts[segment], segments_fz

def get_segment_colors(largest_segment, count, segments, image):
	colors = np.zeros(shape=(count, 3))
	count = 0
	for i in range(segments.shape[0]):
		for j in range(segments.shape[1]):
			if segments[i][j] == largest_segment:
				colors[count] = image[i][j]
				count += 1
	return colors

def get_largest_kmeans_cluster(colors):
	kmeans = KMeans(n_clusters=8, random_state=0).fit(colors)

	elements_per_cluster = np.bincount(kmeans.labels_)

	#NOTE: returns the first largest value not all the largest
	largest_pos = np.argmax(elements_per_cluster)
	largest_cluster = kmeans.cluster_centers_[largest_pos] 

	return largest_cluster

def get_color(color):
	# http://www.rapidtables.com/web/color/RGB_Color.htm
	# red (255,0,0)
	# orange (255,165,0)
	# yellow (255,255,0)
	# green (0,255,0)
	# light blue (0,255,255)
	# blue (0,0,255)
	# purple (128,0,128)
	# pink (255,0,255)
	# dirt laundry (192,192,192)
	r = color[0]
	g = color[1]
	b = color[2]
	#colors = {
	#	'red': ((255,0,0), 500),
	#	'orange': ((255,165,0), 500),
	#	'yellow': ((255,255,0), 500),
	#	'green': ((0,255,0), 500),
	#	'light blue': ((0, 255,255), 500),
	#	'blue': ((0,0,255), 500),
	#	'purple': ((128,0,128), 500),
	#	'pink': ((255,0,255), 500),
	#	'dirty laundry': ((192,192,192), 500),
	#	}
	colors = load_colors('colors/data.json')
	closest = ('', 500)
	for color, vals in colors.iteritems():
		# euclidean distance
		distance = math.sqrt(pow(r - vals[0][0], 2) + pow(g - vals[0][1], 2) + pow(b - vals[0][2], 2))
		colors[color] = (colors[color][0], distance)
		if distance < closest[1]:
			closest = (color, distance)

	return closest[0]


def run():
	il = load_data.ImageLoader('fashion-data')
	test_images = il.get_test_data()

	for i in range(100):
		largest_segment, count, segments = get_felzenswalb_segment(test_images[i])
		colors = get_segment_colors(largest_segment, count, segments, test_images[i])
		# print(colors)
		quantized_common_color = get_largest_kmeans_cluster(colors)
		print(quantized_common_color)

		# http://gauth.fr/2011/09/get-a-color-name-from-any-rgb-combination/
		color = get_color(quantized_common_color)
		print(color)

		imgplot = plt.imshow(mark_boundaries(test_images[i], segments))
		plt.show()


if __name__ == '__main__':
	run()