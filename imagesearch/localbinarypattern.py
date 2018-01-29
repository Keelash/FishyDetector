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

	def lbp_from_bBox(self, full_image, boundingBox)
		x1 = boundingBox[0]
		x2 = boundingBox[1]
		y1 = boundingBox[2]
		y2 = boundingBox[3]
		image = full_image[y1:y2,x1:x2]
		hist = describe(image)
		return hist