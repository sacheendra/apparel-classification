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
			self.test_images = self.load_data(partialtestbinpath)
			print "partial test data loaded from numpy file"
		elif loadall and os.path.isfile(os.path.join(folderpath, testbinpath)):
			self.test_images = self.load_data(testbinpath)
			print "test data loaded from numpy file"
		elif ~loadall and os.path.isfile(os.path.join(folderpath, testbinpath)):
			self.test_images = self.load_data(testbinpath)[:100]
			self.store_data(self.test_images, partialtestbinpath)
		else:
			self.test_filenames = pd.read_csv(os.path.join(folderpath, 'test.txt'))
			self.test_images = self.load_images(self.test_filenames)
			self.store_data(self.test_images, testbinpath)
			self.store_data(self.test_images[:100], partialtestbinpath)

		if ~loadall and os.path.isfile(os.path.join(folderpath, partialtrainbinpath)):
			self.train_images = self.load_data(partialtrainbinpath)
			print "partial train data loaded from numpy file"
		elif loadall and os.path.isfile(os.path.join(folderpath, trainbinpath)):
			self.train_images = self.load_data(trainbinpath)
			print "train data loaded from numpy file"
		elif ~loadall and os.path.isfile(os.path.join(folderpath, trainbinpath)):
			self.train_images = self.load_data(trainbinpath)[:500]
			self.store_data(self.train_images, partialtrainbinpath)
		else:
			self.train_filenames = pd.read_csv(os.path.join(folderpath, 'train.txt'))
			self.train_images = self.load_images(self.train_filenames)
			self.store_data(self.train_images, trainbinpath)
			self.store_data(self.train_images[:500], partialtrainbinpath)

	def load_images(self, filenames):
		images = [np.ndarray(np.ndarray(0))]*filenames.shape[0]
		for index, row in filenames.iterrows():
			images[index] = io.imread(
				os.path.join(self.folderpath, 'images', row.iloc[0] + '.jpg'))
		return images

	def store_data(self, data, filename):
		with open(os.path.join(self.folderpath, filename), 'wb+') as fh:
			np.save(fh, data)

	def load_data(self, filename):
		return np.load(os.path.join(self.folderpath, filename))

	def get_test_data(self):
		return self.test_images

	def get_train_data(self):
		return self.train_data


def main():
	il = ImageLoader('fashion-data')
	test_images = il.get_test_data()
	print(len(test_images[0]), len(test_images[0][0]))
	print(test_images[0][0])
		
if __name__ == '__main__':
	main()