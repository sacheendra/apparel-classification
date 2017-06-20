import os
import skimage
from skimage import io
import pandas as pd
import numpy as np

class ImageLoader(object):
	"""load train and test data"""
	def __init__(self, folderpath, loadall=False):
		super(ImageLoader, self).__init__()
		testbinpath = 'test.npy'
		trainbinpath = 'train.npy'
		partialtestbinpath = 'partialtest.npy'
		partialtrainbinpath = 'partialtrain.npy'

		self.folderpath = folderpath

		if ~loadall and os.path.isfile(os.path.join(folderpath, partialtestbinpath)):
			(self.test_images, self.test_labels) = self.load_data(partialtestbinpath)
			print "partial test data loaded from numpy file"
		elif loadall and os.path.isfile(os.path.join(folderpath, testbinpath)):
			(self.test_images, self.test_labels) = self.load_data(testbinpath)
			print "test data loaded from numpy file"
		elif ~loadall and os.path.isfile(os.path.join(folderpath, testbinpath)):
			(all_test_images, all_test_labels) = self.load_data(testbinpath)
			self.test_images = all_test_images[:100]
			self.test_labels = all_test_labels[:100]
			self.store_data((self.test_images, self.test_labels), partialtestbinpath)
		else:
			self.test_filenames = pd.read_csv(os.path.join(folderpath, 'test.txt'))
			(self.test_images, self.test_labels) = self.load_images(self.test_filenames)
			self.store_data((self.test_images, self.test_labels), testbinpath)
			self.store_data((self.test_images[:100], self.test_labels[:100]), partialtestbinpath)

		if ~loadall and os.path.isfile(os.path.join(folderpath, partialtrainbinpath)):
			(self.train_images, self.train_labels) = self.load_data(partialtrainbinpath)
			print "partial train data loaded from numpy file"
		elif loadall and os.path.isfile(os.path.join(folderpath, trainbinpath)):
			(self.train_images, self.train_labels) = self.load_data(trainbinpath)
			print "train data loaded from numpy file"
		elif ~loadall and os.path.isfile(os.path.join(folderpath, trainbinpath)):
			(all_train_images, all_train_labels) = self.load_data(trainbinpath)
			self.train_images = all_train_images[:500]
			self.train_labels = all_train_labels[:500]
			self.store_data((self.train_images, self.train_labels), partialtrainbinpath)
		else:
			self.train_filenames = pd.read_csv(os.path.join(folderpath, 'train.txt'))
			(self.train_images, self.train_labels) = self.load_images(self.train_filenames)
			self.store_data((self.train_images, self.train_labels), trainbinpath)
			self.store_data((self.train_images[:500], self.train_labels[:500]), partialtrainbinpath)

	def load_images(self, filenames):
		images = [np.ndarray(np.ndarray(0))] * filenames.shape[0]
		labels = np.zeros(filenames.shape[0], dtype=np.uint8)
		for index, row in filenames.iterrows():
			imagename = row.iloc[0]
			images[index] = io.imread(
				os.path.join(self.folderpath, 'images', imagename + '.jpg'))
			labels[index] = imagename.split('/', 1)[0]
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