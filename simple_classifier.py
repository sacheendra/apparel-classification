from load_data import ImageLoader
from feature_extractors import SURFExtractor, HOGExtractor, DimensionNormalizer
from scipy.cluster import vq

def main():
	loader = ImageLoader('fashion-data')
	train_data = loader.get_train_data()

	surf_features = SURFExtractor(train_data, 400).get_features()
	dim_normalised_surf_features = DimensionNormaliser(surf_features).get_features()

	hog_features = HOGExtractor(train_data).get_features()
	dim_normalised_hog_features = DimensionNormaliser(surf_features).get_features()


		
if __name__ == '__main__':
	main()