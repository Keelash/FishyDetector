from sklearn.svm import LinearSVC

from imagesearch.helpers import slidingWindow
from imagesearch.helpers import pyramid
from imagesearch.localbinarypattern import LocalBinaryPatterns
from utils.percentbar import PercentBar

import imutils
import cv2
import sys

def trainSVC(trainingData,
			 stepSize = 8,
			 windowSize=(128, 64),
			 scale = 2.0,
			 epsilon = 0.6
			 ) :
	svcData = []
	svcLabel = []

	bar = PercentBar("Data Training")
	bar.setPercent(int(0))
	bar.show()

	desc = LocalBinaryPatterns(24, 8) #Must be user defined later
	it = 0.0

	for (i, (filePath, posWindow)) in enumerate(trainingData) :

		#We reduce the image because there are too big for quick test. But it can
		#be set as an option for the user.

		imageTrain = cv2.cvtColor(cv2.imread(filePath), cv2.COLOR_BGR2GRAY)
		imageTrain = imutils.resize(imageTrain, width=int(imageTrain.shape[1]/4))
		(y, x, w, h) = [ int(value/4) for value in posWindow]

		for resized in pyramid(imageTrain, scale=scale) :
			for (winX, winY, window) in slidingWindow(resized, stepSize=stepSize, windowSize=windowSize):

				if window.shape[0] != windowSize[1] or window.shape[1] != windowSize[0]:
					continue

				hist = desc.describe(window)

				#We determine if the chunk of the picture contain the fish based on the proximity
				#with his real position. For this, we use the percent of fish covered by the 
				#the sliding window
				x_overlap = max(0, min(x+w, winX+windowSize[0]) - max(x, winX));
				y_overlap = max(0, min(y+h, winY+windowSize[1]) - max(y, winY));
				overlapAreaPercent = float(x_overlap * y_overlap) / float(w*h);

				if overlapAreaPercent > epsilon and overlapAreaPercent != 1.0 :
					svcLabel.append("Fishy")
				else :
					svcLabel.append("NotAFishy")

				svcData.append(hist)

			(x, y, w, h) = [ int(value / scale) for value in (x, y, w, h)]

		it += 1.0
		bar.setPercent(int((it/len(trainingData))*100))
		bar.show()

	model = LinearSVC(C=100.0, random_state=42)
	model.fit(svcData, svcLabel)

	return model