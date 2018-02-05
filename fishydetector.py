from imagesearch.helpers import pyramid
from imagesearch.helpers import slidingWindow
from imagesearch.localbinarypattern import LocalBinaryPatterns
from svctraining import trainSVC

import joblib
import argparse
import json
import imutils
import cv2


# When you load an image using OpenCV it loads that image into BGR color space by default. To show the colored image using `matplotlib` we have to convert it to RGB space. Following is a helper function to do exactly that. 
def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


ap = argparse.ArgumentParser()
ap.add_argument("classifier", help="Path to the classifier to use or create")
ap.add_argument( '-ss', '--stepSize', nargs='?', const='16', default='16', type=int, help='The step of the sliding window')
ap.add_argument( '-s', '--scale', nargs='?', const='1.5', default='1.5', type=float, help='The scale between two step of the image pyramid')


subap = ap.add_subparsers(dest='subcommand')
apTraining = subap.add_parser('train')
apTraining.add_argument( 'trainingFile', help='The path to the training data json file')
ap.add_argument( '-e', '--epsilon', nargs='?', const='0.8', default='0.8', type=float, help='The percent the sliding window must cover to be valid, only used in training mode' )
apDetect = subap.add_parser('detect')
apDetect.add_argument('image', nargs='+', help='The path to an Image')

args = vars(ap.parse_args())

if args["subcommand"] == "train" :
	trainingData = []
	windowRatioList = []

	path = args["trainingFile"].rsplit('/', 1)[0] + '/';
	with open(args["trainingFile"]) as data_file :
		data = json.load(data_file)

	for fileName in data :
		window = list(map(int, data[fileName]["outer"]))
		windowRatioList.append(float(window[3])/float(window[2]))
		trainingData.append((path + fileName, window))

	windowRatio = sum(windowRatioList) / len(windowRatioList)
	model = trainSVC(
					trainingData,
					windowSize=(128, int(128*windowRatio)),
					stepSize=args["stepSize"],
					epsilon=args["epsilon"],
					scale=args["scale"]
					)

	joblib.dump((windowRatio, model), args["classifier"])


if args["subcommand"] == "detect" :
	(winRatio, model) = joblib.load(args["classifier"])
	desc = LocalBinaryPatterns(24, 8)


	for imagePath in args["image"] :
		imageDetect = cv2.imread(imagePath)
		imageDetectGray = cv2.cvtColor(imageDetect, cv2.COLOR_BGR2GRAY)
		imageDetectGray = imutils.resize(imageDetectGray, width=int(imageDetectGray.shape[0]/4))

		for resized in pyramid(imageDetectGray, scale=args["scale"]):
			for (x, y, window) in slidingWindow(resized, stepSize=args["stepSize"], windowSize=(128, int(128*winRatio))):

				if window.shape[0] != int(128*winRatio) or window.shape[1] != 128:
					continue
				
				hist = desc.describe(window)
				prediction = model.predict(hist.reshape(1, -1))[0]

				if prediction == "Fishy" :
					print ("Found ! at [{}, {}, {}, {}] in {}".format(x, y, window.shape[0], window.shape[1], imagePath))

				clone = resized.copy()
				cv2.rectangle(clone, (x, y), (x + 128, y + int(128*winRatio)), (0, 255, 0), 2)
				cv2.imshow("Window", clone)
				cv2.waitKey(1)