from sklearn.cluster import KMeans
import numpy as np
from load_data import ImageLoader
from LABExtractor import LABExtractor
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def get_image_lab_histograms(images):
	e = LABExtractor(images)
	lab_histograms = e.get_features()
	return lab_histograms

def get_color_kmeans(images, n_clusters=8):
	histograms = get_image_lab_histograms(images)
	kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(histograms)
	return kmeans, histograms

def run():
	il = ImageLoader('fashion-data')
	train_images = il.get_train_data()
	kmeans, histograms = get_color_kmeans(train_images)
	closest = kmeans.predict(histograms)
	print("Centroids:")
	for i in range(np.shape(histograms)[0]):

		# fig1 = plt.figure(1)
		fig1, axarr = plt.subplots(1,2)
		axarr[0].imshow(train_images[i])
		# plt.imshow(train_images[i])
		title = "Image #{} Closest centroid: {}".format(
			i,
			closest[i])
		fig1.suptitle(title, fontsize=14, fontweight='bold')

		# fig2, ax = plt.subplots(figsize=(12, 1.5))
		axarr[1].set_title("Histogram")
		axarr[1].bar(range(len(histograms[i])), histograms[i], color='r', width=1)

		plt.show()

	print(enumerate(kmeans.cluster_centers_))
	

if __name__ == '__main__':
	run()