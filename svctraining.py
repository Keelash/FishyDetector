from sklearn.svm import LinearSVC

from imagesearch.helpers import slidingWindow
from imagesearch.localbinarypattern import LocalBinaryPatterns
from utils.percentbar import PercentBar

import imutils
import cv2

def trainSVC(trainingData) :
	svcData = []
	svcLabel = []
	svcWeight = []

	bar = PercentBar("Data Training");
	bar.show()
	it = 0.0

	desc = LocalBinaryPatterns(24, 8) #Must be user defined later

	#We need to compute the mean of all the window

	for (filePath, posWindow) in trainingData :

		#We reduce the image because there are two big for quick test. But it can
		#be set as an option for the user.

		imageTrain = cv2.cvtColor(cv2.imread(filePath), cv2.COLOR_BGR2GRAY)
		imageTrain = imutils.resize(imageTrain, width=imageTrain.shape[0]/4)

		(x, y, w, h) = [ value / 4 for value in posWindow]
		hist = desc.describe(imageTrain[y:y+h, x:x+w])

		for (winX, winY, window) in slidingWindow(imageTrain, stepSize=32, windowSize=(w, h)):

			if window.shape[0] != h or window.shape[1] != w:
				continue

			hist = desc.describe(window)

			x_overlap = max(0, min(x+w, winX+w) - max(x, winX));
			y_overlap = max(0, min(y+h, winY+h) - max(y, winY));
			overlapAreaPercent = float(x_overlap * y_overlap) / float(w * h);

			if overlapAreaPercent > 0.8 :
				svcLabel.append("Fishy")
				svcWeight.append(overlapAreaPercent)
			else :
				svcLabel.append("NotAFishy")
				svcWeight.append(1.0 - overlapAreaPercent)

			svcData.append(hist)

		it += 1.0
		bar.setPercent(int((it/len(trainingData))*100))
		bar.show()

	model = LinearSVC(C=100.0, random_state=42)
	model.fit(svcData, svcLabel, svcWeight)

	return model