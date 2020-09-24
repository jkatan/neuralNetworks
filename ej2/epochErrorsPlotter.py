import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def graph(trainingFilename, testingFilename):
	trainingFile = open(trainingFilename, "r")
	trainingLines = trainingFile.readlines()

	trainingXData = []
	trainingYData = []

	for trainingLine in trainingLines:
		sample = trainingLine.split(",")
		trainingXData.append(int(sample[0]))
		trainingYData.append(float(sample[1]))

	testingFile = open(testingFilename, "r")
	testingLines = testingFile.readlines()

	testingXData = []
	testingYData = []

	for testingLine in testingLines:
		sample = testingLine.split(",")
		testingXData.append(int(sample[0]))
		testingYData.append(float(sample[1]))

	xTesting = np.array(testingXData)
	yTesting = np.array(testingYData)

	xTraining = np.array(trainingXData)
	yTraining = np.array(trainingYData)

	fig, ax = plt.subplots()
	ax.set(xlabel='epochs', ylabel='error',
	       title='epochs vs error')
	ax.grid()
	ax.plot(xTesting, yTesting, color='black')
	ax.plot(xTraining, yTraining, color='red')
	plt.show()

graph("training-errors-per-epoch.csv", "testing-errors-per-epoch.csv")

