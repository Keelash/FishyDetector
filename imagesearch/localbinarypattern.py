from skimage import feature
import numpy as np
 
class LocalBinaryPatterns:
	def __init__(self, numPoints, radius):
		self.numPoints = numPoints
		self.radius = radius
 
 	#Return the LBP histogramme of the image
	def describe(self, image, eps=1e-7):
		lbp = feature.local_binary_pattern(image, self.numPoints,
			self.radius, method="uniform")
		(hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, self.numPoints + 3),
			range=(0, self.numPoints + 2))
 
		# normalize the histogram
		hist = hist.astype("float")
		hist /= (hist.sum() + eps)
 
		return hist