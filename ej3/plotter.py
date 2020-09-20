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
	x = np.linspace(-5,5,100)

	weightsNeuron1 = lines[0].split(",")
	m = (-float(weightsNeuron1[0]))/(float(weightsNeuron1[1]))
	b = (-float(weightsNeuron1[2]))/(float(weightsNeuron1[1]))
	y = x * m + b

	weightsNeuron2 = lines[1].split(",")
	m1 = (-float(weightsNeuron2[0]))/(float(weightsNeuron2[1]))
	b1 = (-float(weightsNeuron2[2]))/(float(weightsNeuron2[1]))
	y1 = x * m1 + b1

	plt.plot(x, y, color="black", label='Hiperplano neurona 1')
	plt.plot(x, y1, color="green", label='Hiperplano neurona 2')
	plt.xlabel('x', color='#1C2833')
	plt.ylabel('y', color='#1C2833')
	plt.legend(loc='upper left')
	plt.grid()
	plt.show()

graphData("XOR-TrainingData.csv", "XOR-neuron-weights.csv")