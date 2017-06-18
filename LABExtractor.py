import numpy as np
from skimage import color, exposure
import cv2

import load_data
import self_similarity
import matplotlib.pyplot as plt



def ipynb_show_color_histogram(histogram, plot_title=''):
	fig, ax = plt.subplots(figsize=(12, 1.5))
	ax.set_title(plot_title)
	ax.bar(range(len(histogram)), histogram, color='r', width=1)
	plt.show()

def compute_histogram(image, channels, bins, ranges):
  # We return the histogram as a single vector, in which the three sub-histograms are concatenated.
  histogram = np.zeros(np.sum(bins))
  
  # We generate a histogram per channel, and then add it to the single-vector histogram.
  for i in range(0, len(channels)):
    channel = channels[i]
    channel_bins = bins[i]
    channel_range = ranges[i]
    channel_histogram = cv2.calcHist(
        [image],
        [channel],
        None, # one could specify an optional mask here (we don't use this here),
        [channel_bins],
        channel_range
        )
    
    # We normalize values in the histogram, such that the values sum up to 1.0.
    channel_histogram_normalized = channel_histogram / np.sum(channel_histogram)
    
    # Now we copy these values to the right indices in our single-vector histogram.
    start_index = int(np.sum(bins[0:channel]))
    end_index = start_index + channel_bins
    histogram[start_index:end_index] = channel_histogram_normalized.flatten()

  return histogram

def compute_lab_histogram(image, bins_per_channel=[8, 8, 8]):
	channels = [0, 1, 2]
	# Channel ranges L: 0 to 100, a: -127 to 128, b: -128 to 127.
	# From: https://stackoverflow.com/questions/25294141/cielab-color-range-for-scikit-image
	ranges = [[0, 100], [-127, 128], [-128, 127]]
	return compute_histogram(image, channels, bins_per_channel, ranges)

class LABExtractor(object):
	"""Extracts L*a*b* color space
	from a list of images."""
	def __init__(self, images):
		super(LABExtractor, self).__init__()

		self.lab_histograms = [np.ndarray(0)] * images.shape[0]

		for i in range(len(images)):
			# image = color.rgb2lab(images[i]).astype(np.float32)
			# image = images[i].astype(np.float32)
			# histogram = self.caclulate_color_histogram(image)
			# ipynb_show_color_histogram(histogram)
			# print(histogram)
			# self.lab_histograms[i] = histogram

			# The way done in the lab
			bins_per_channel = [8, 8, 8]
			self.lab_histograms[i] = compute_lab_histogram(color.rgb2lab(images[i]).astype(np.float32), bins_per_channel)
			# ipynb_show_color_histogram(self.lab_histograms[i])

	def caclulate_color_histogram(self, image):
		histogram = np.zeros(24)
		start = 0
		end = 8
		step = 8
		for channel in [0, 1, 2]:
			add = 127 if channel == 1 else 128 if channel == 2 else 0 
			res = exposure.histogram(image[:,:,channel], nbins=8)[0] + add
			normalized = res.astype(np.float32) / np.sum(res)
			histogram[start:end] = normalized.flatten()
			start = end
			end += step
		return histogram

	def get_features(self):
		return self.lab_histograms

def run():
	il = load_data.ImageLoader('fashion-data')
	test_images = il.get_test_data()

	e = LABExtractor(test_images)
	lab_features = e.get_features()
	print(lab_features)
	print(np.shape(lab_features))

if __name__ == '__main__':
	run()
