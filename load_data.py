import os
import skimage
from skimage import io
import pandas as pd
import numpy as np
from scipy.io import savemat, loadmat
import h5py

class ImageLoader(object):
	"""load train and test data"""
	def __init__(self, folderpath):
		super(ImageLoader, self).__init__()
		testbinpath = 'test.npy'
		trainbinpath = 'train.npy'

		self.folderpath = folderpath

		if os.path.isfile(os.path.join(folderpath, testbinpath)):
			self.test_images = self.load_data(testbinpath)
			print "test data loaded from numpy file"
		else:
			self.test_filenames = pd.read_csv(os.path.join(folderpath, 'test.txt'))
			self.test_images = self.load_images(self.test_filenames)
			self.store_data(self.test_images, testbinpath)

		if os.path.isfile(os.path.join(folderpath, trainbinpath)):
			self.train_images = self.load_data(trainbinpath)
			print "train data loaded from numpy file"
		else:
			self.train_filenames = pd.read_csv(os.path.join(folderpath, 'train.txt'))
			self.train_images = self.load_images(self.train_filenames)
			self.store_data(self.train_images, trainbinpath)

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


def main():
	il = ImageLoader('fashion-data')
		
if __name__ == '__main__':
	main()