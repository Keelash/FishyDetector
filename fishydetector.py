from imagesearch.helpers import pyramid
from imagesearch.helpers import slidingWindow
from svctraining import trainSVC

import joblib
import argparse
import json
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("classifier", help="Path to the classifier to use")
ap.add_argument("-v", "--verbose", action="store_true", help="Make the function verbose")

subap = ap.add_subparsers(dest='subcommand')
apTraining = subap.add_parser('train')
apTraining.add_argument( 'trainingFile', help='The path to the training data json file')
apDetect = subap.add_parser('detect')
apDetect.add_argument('Image', help='The path to an Image')

args = vars(ap.parse_args())

verbose = args["verbose"]

if args["subcommand"] == "train" :
	trainingData = []

	with open(args["trainingFile"]) as data_file :
		data = json.load(data_file)

	for filePath in data :
		window = list(map(int, data[filePath]["outer"]))
		trainingData.append((filePath, window))

	if verbose :
		print("Training an SVC based on the {} data : ".format(args["trainingFile"].split("/")[1]))

	model = trainSVC(trainingData)

	if verbose :
		print("Saving the SVC in file {}".format(args["classifier"]))

	joblib.dump(model, args["classifier"])


if args["subcommand"] == "detect" :
	imageDetect = cv2.imread(args["detect"])
	imageDetectGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	#TODO : Load the SVC

	#TODO : We must save and load the mean ratio of the training window
	(winW, winH) = (128, 128)

	for resized in pyramid(image, scale=1.5):
		for (x, y, window) in slidingWindow(resized, stepSize=32, windowSize=(winW, winH)):

			if window.shape[0] != winH or window.shape[1] != winW:
				continue

			#TODO : The detection code in each window

			clone = resized.copy()
			cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
			cv2.imshow("Window", clone)
			cv2.waitKey(1)