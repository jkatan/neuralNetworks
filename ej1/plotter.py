import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np

def graphData(trainingDataFilename, weightsFilename):
	trainingDataFile = open(trainingDataFilename, "r")
	lines = trainingDataFile.readlines()

	redClassXpoints = []
	redClassYpoints = []

	blueClassXpoints = []
	blueClassYpoints = []

	for line in lines:
		trainingSample = line.split(",")
		sampleClass = float(trainingSample[2])
		if sampleClass == 1.0:
			blueClassXpoints.append(float(trainingSample[0]))
			blueClassYpoints.append(float(trainingSample[1]))
		else:
			redClassXpoints.append(float(trainingSample[0]))
			redClassYpoints.append(float(trainingSample[1]))

	plt.scatter(redClassXpoints, redClassYpoints, color="red")
	plt.scatter(blueClassXpoints, blueClassYpoints, color="blue")

	weightsDataFile = open(weightsFilename, "r")
	lines = weightsDataFile.readlines()
	weights = lines[0].split(",")
	m = (-float(weights[0]))/(float(weights[1]))
	b = (-float(weights[2]))/(float(weights[1]))
	x = np.linspace(-5,5,100)
	y = x * m + b

	plt.plot(x, y, color="black", label='Hiperplano generado por el perceptron simple')
	plt.xlabel('x', color='#1C2833')
	plt.ylabel('y', color='#1C2833')
	plt.legend(loc='upper left')
	plt.grid()
	plt.show()

graphData("AND-trainingData.csv", "AND-weights.csv")
graphData("XOR-trainingData.csv", "XOR-weights.csv")