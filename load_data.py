import os
import skimage
from skimage import io
from skimage import transform
import pandas as pd
import numpy as np

class ImageLoader(object):
	"""load train and test data"""
	def __init__(self, folderpath, train_images_per_category=100, test_images_per_category=10, num_categories=14):
		super(ImageLoader, self).__init__()
		partialtestbinpath = 'partialtest.npy'
		partialtrainbinpath = 'partialtrain.npy'

		self.folderpath = folderpath
		self.train_images_per_category = train_images_per_category
		self.test_images_per_category = test_images_per_category
		self.num_categories = num_categories

		if os.path.isfile(os.path.join(folderpath, partialtestbinpath)):
			(self.test_images, self.test_labels, _) = self.load_data(partialtestbinpath)
			print "partial test data loaded from numpy file"
		else:
			self.test_filenames = pd.read_csv(os.path.join(folderpath, 'test.txt'))
			(self.test_images, self.test_labels) = self.load_images(self.test_filenames, test_images_per_category, num_categories)
			# https://stackoverflow.com/questions/25552741/python-numpy-not-saving-array
			# extra array of zeroes is a hack to make numpy save work.
			self.store_data((self.test_images, self.test_labels, np.zeros(0)), partialtestbinpath)
			print "partial test data loaded"

		if os.path.isfile(os.path.join(folderpath, partialtrainbinpath)):
			(self.train_images, self.train_labels, _) = self.load_data(partialtrainbinpath)
			print "partial train data loaded from numpy file"
		else:
			self.train_filenames = pd.read_csv(os.path.join(folderpath, 'train.txt'))
			(self.train_images, self.train_labels) = self.load_images(self.train_filenames,  train_images_per_category, num_categories)
			self.store_data((self.train_images, self.train_labels, np.zeros(0)), partialtrainbinpath)
			print "partial train data loaded"

	def load_images(self, filenames, images_per_category, num_categories):
		filenames_by_category = [[] for i in range(num_categories)]
		for index, row in filenames.iterrows():
			imagename = row.iloc[0]
			label = int(imagename.split('/', 1)[0])
			if label < num_categories:
				filenames_by_category[label].append(imagename)

		images = np.zeros((images_per_category * num_categories, 320, 320, 3), dtype=np.uint8)
		labels = np.zeros(images_per_category * num_categories, dtype=np.uint8)

		for i, category in enumerate(filenames_by_category):
			images_to_pick = np.random.randint(0, len(category), images_per_category)
			count = 0
			for imageindex in images_to_pick:
				imagename = category[imageindex]
				index_in_dataset = i*images_per_category + count
				images[index_in_dataset][:][:][:] = transform.resize(io.imread(
					os.path.join(self.folderpath, 'images', imagename + '.jpg')), (320, 320), mode='reflect')
				labels[index_in_dataset] = i
				count = count + 1
		return (images, labels)

	def store_data(self, data, filename):
		with open(os.path.join(self.folderpath, filename), 'wb+') as fh:
			np.save(fh, data)

	def load_data(self, filename):
		return np.load(os.path.join(self.folderpath, filename))

	def get_test_data(self):
		return self.test_images

	def get_test_labels(self):
		return self.test_labels

	def get_train_data(self):
		return self.train_images

	def get_train_labels(self):
		return self.train_labels


def main():
	il = ImageLoader('fashion-data')
	test_images = il.get_test_data()
	test_labels = il.get_test_labels()
	print(len(test_images[0]))
	print(len(test_images[0][0]))
	print(len(test_labels))
		
if __name__ == '__main__':
	main()