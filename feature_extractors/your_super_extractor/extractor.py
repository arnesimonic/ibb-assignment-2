import cv2, sys
from skimage import feature, filters
import numpy as np


class Super:
	def __init__(self, resize=100):
		self.resize=resize

	def extract(self, img):
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, (self.resize, self.resize))

		prewitt_h = filters.prewitt_h(img)
		prewitt_v = filters.prewitt_v(img)

		cv2.imshow("", prewitt_v)
		cv2.waitKey(0)
		
		return img


if __name__ == '__main__':
	fname = sys.argv[1]
	img = cv2.imread(fname)
	extractor = Super()
	features = extractor.extract(img)
	print(features)
