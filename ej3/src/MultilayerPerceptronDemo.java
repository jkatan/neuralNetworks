import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

public class MultilayerPerceptronDemo {

    public static void main(String[] args) throws IOException {
        xorExercise();
        imagesExercise();
    }

    private static void imagesExercise() throws FileNotFoundException {
        File file = new File("mapaDeDigitos.txt");
        Scanner reader = new Scanner(file);
        int lineNumber = 1;
        List<List<Double>> trainingData = new ArrayList<>();
        List<List<Double>> expectedOutputs = new ArrayList<>();
        int currentDigit = 0;
        List<Double> digitMap = new ArrayList<>();
        while (reader.hasNextLine()) {
            String data = reader.nextLine();
            for (int i = 0; i < data.length(); i++) {
                if (data.charAt(i) != '\n' && data.charAt(i) != ' ') {
                    digitMap.add(Double.parseDouble(String.valueOf(data.charAt(i))));
                }
            }
            if (lineNumber % 7 == 0) {
                List<Double> output = new ArrayList<>();
                if (currentDigit % 2 == 0) {
                    output.add(1.0); // even -> 1.0
                } else {
                    output.add(-1.0); // odd -> -1.0
                }
                expectedOutputs.add(output);
                currentDigit++;
                trainingData.add(digitMap);
                digitMap = new ArrayList<>();
            }
            lineNumber += 1;
        }

        reader.close();

        // We use 6 digits to train and 4 digits to test, all random
        List<List<Double>> testData = new ArrayList<>();
        List<List<Double>> expectedOutputsTest = new ArrayList<>();
        Random rand = new Random();
        for (int i = 0; i < 4; i++) {
            int randIndex = rand.nextInt(trainingData.size());
            testData.add(trainingData.remove(randIndex));
            expectedOutputsTest.add(expectedOutputs.remove(randIndex));
        }

        List<List<Neuron>> neuralNetwork = createANeuralNetwork(trainingData.get(0).size(), 2, 1);
        System.out.println("TRAIN IMAGES NEURAL NETWORK");
        trainNeuralNetwork(neuralNetwork, trainingData, expectedOutputs, 0.001, 0.01);
        System.out.println("Neural network layers: ");
        for (List<Neuron> layer : neuralNetwork) {
            System.out.println(layer);
        }

        System.out.println();

        System.out.println("Test results using training data:");
        printTestResultsImageExercise(trainingData, neuralNetwork, expectedOutputs);

        System.out.println();

        System.out.println("Tests results using testing data:");
        printTestResultsImageExercise(testData, neuralNetwork, expectedOutputsTest);
        System.out.println();
    }

    private static void printData(List<List<Double>> trainingData, List<List<Double>> expectedOutputs) {
        int digit = 0;
        for (int i=0; i<trainingData.size(); i++) {
            for (int j = 0; j < trainingData.get(i).size(); j++) {
                System.out.print(trainingData.get(i).get(j) + " ");
                if ((j + 1) % 5 == 0) {
                    System.out.println();
                }
            }
            System.out.println("Digit: " + digit);
            System.out.println("Expected output: " + expectedOutputs.get(i));
            System.out.println();
            digit += 1;
        }
    }

    private static void xorExercise() throws IOException {
        List<List<Double>> trainingData = new ArrayList<>();

        List<Double> sampleOne = new ArrayList<>();
        sampleOne.add(-1.0);
        sampleOne.add(1.0);
        List<Double> sampleTwo = new ArrayList<>();
        sampleTwo.add(1.0);
        sampleTwo.add(-1.0);
        List<Double> sampleThree = new ArrayList<>();
        sampleThree.add(-1.0);
        sampleThree.add(-1.0);
        List<Double> sampleFour = new ArrayList<>();
        sampleFour.add(1.0);
        sampleFour.add(1.0);

        trainingData.add(sampleOne);
        trainingData.add(sampleTwo);
        trainingData.add(sampleThree);
        trainingData.add(sampleFour);

        List<List<Double>> expectedOutputs = new ArrayList<>();

        List<Double> expectedOutputOne = new ArrayList<>();
        expectedOutputOne.add(1.0);
        List<Double> expectedOutputTwo = new ArrayList<>();
        expectedOutputTwo.add(1.0);
        List<Double> expectedOutputThree = new ArrayList<>();
        expectedOutputThree.add(-1.0);
        List<Double> expectedOutputFour = new ArrayList<>();
        expectedOutputFour.add(-1.0);

        expectedOutputs.add(expectedOutputOne);
        expectedOutputs.add(expectedOutputTwo);
        expectedOutputs.add(expectedOutputThree);
        expectedOutputs.add(expectedOutputFour);

        List<List<Neuron>> neuralNetwork = createANeuralNetwork(trainingData.get(0).size(), 2, 1);
        System.out.println("TRAIN XOR NEURAL NETWORK");
        trainNeuralNetwork(neuralNetwork, trainingData, expectedOutputs, 0.01, 0.0015);

        System.out.println("Neural network layers: ");
        for (List<Neuron> layer : neuralNetwork) {
            System.out.println(layer);
        }

        System.out.println();

        System.out.println("XOR test results");
        printTestResults(trainingData, neuralNetwork, expectedOutputs);
        System.out.println();

        List<List<Double>> hiddenLayerNeuronsWeights = new ArrayList<>();
        for (Neuron hiddenNeuron : neuralNetwork.get(0)) {
            hiddenLayerNeuronsWeights.add(hiddenNeuron.getWeights());
        }
        writeTrainingDataToFile("XOR-TrainingData.csv", trainingData, expectedOutputs);
        writeWeightsToFile("XOR-neuron-weights.csv", hiddenLayerNeuronsWeights);
    }

    private static void printTestResults(List<List<Double>> trainingData, List<List<Neuron>> neuralNetwork, List<List<Double>> expectedOutputs) {
        System.out.println("Results: ");
        for (int i = 0; i < trainingData.size(); i++) {
            List<Double> outputs = forwardPropagate(neuralNetwork, trainingData.get(i));
            System.out.println("(Inputs: " + trainingData.get(i) + ") -> (Outputs: " + outputs + ")");
            System.out.println("Expected outputs: " + expectedOutputs.get(i));
        }
    }

    private static void printTestResultsImageExercise(List<List<Double>> trainingData, List<List<Neuron>> neuralNetwork, List<List<Double>> expectedOutputs) {
        System.out.println("Results: ");
        int correct = 0;
        int incorrect = 0;
        for (int i = 0; i < trainingData.size(); i++) {
            List<Double> outputs = forwardPropagate(neuralNetwork, trainingData.get(i));
            if ((outputs.get(0) > 0 && expectedOutputs.get(i).get(0) > 0) || ((outputs.get(0) < 0 && expectedOutputs.get(i).get(0) < 0))) {
                correct += 1;
            } else {
                incorrect += 1;
            }
        }
        System.out.println("Correct predictions: " + correct);
        System.out.println("Incorrect predictions: " + incorrect);
    }

    private static void trainNeuralNetwork(List<List<Neuron>> neuralNetwork, List<List<Double>> trainingSamples, List<List<Double>> expectedOutputs, double eta, double minError) {
        int currentEpoch = 1;
        double currentMaxError = minError+1;
        Random random = new Random();
        while (currentMaxError > minError) {
            List<Double> errors = new ArrayList<>();
            for (int j = 0; j < trainingSamples.size(); j++) {
                int randomIndex = random.nextInt(trainingSamples.size());
                List<Double> trainingInput = trainingSamples.get(randomIndex);
                List<Double> expectedTrainingOutput = expectedOutputs.get(randomIndex);
                double error = 0.0;
                List<Double> outputs = forwardPropagate(neuralNetwork, trainingInput);
                for (int i = 0; i < outputs.size(); i++) {
                    error += Math.abs(outputs.get(i) - expectedTrainingOutput.get(i));
                }
                errors.add(error);
                backPropagate(neuralNetwork, expectedTrainingOutput);
                updateWeights(neuralNetwork, trainingInput, eta);
            }

            currentMaxError = Collections.max(errors);
            currentEpoch += 1;
        }

        System.out.println("Epochs: " + currentEpoch);
        System.out.println("Max error: " + currentMaxError);
    }

    private static List<List<Neuron>> createANeuralNetwork(int entryLayerInputs, int hiddenLayerNeurons, int exitLayerNeurons) {
        List<List<Neuron>> neuralNetwork = new ArrayList<>();

        List<Neuron> hiddenLayer = createALayer(hiddenLayerNeurons, entryLayerInputs);
        neuralNetwork.add(hiddenLayer);

        List<Neuron> outputLayer = createALayer(exitLayerNeurons, hiddenLayer.size());
        neuralNetwork.add(outputLayer);

        return neuralNetwork;
    }

    // Given a neural network that has passed through the forward and backpropagation phases, update it's weights
    private static void updateWeights(List<List<Neuron>> neuralNetwork, List<Double> inputs, double eta) {
        List<Double> nextInputs = inputs;
        for (int i = 0; i < neuralNetwork.size(); i++) {
            if (i > 0) {
                nextInputs = neuralNetwork.get(i-1).stream().map(Neuron::getOutput).collect(Collectors.toList());
            }

            List<Neuron> currentLayer = neuralNetwork.get(i);
            for (Neuron neuron : currentLayer) {
                for (int j = 0; j < nextInputs.size(); j++) {
                    Double weight = neuron.getWeights().get(j);
                    //System.out.println("j: " + j);
                    //System.out.println("nextInputsSize: " + nextInputs.size());
                    weight += eta * neuron.getDelta() * nextInputs.get(j);
                    neuron.getWeights().set(j, weight);
                }

                int weightsAmount = neuron.getWeights().size();
                Double bias = neuron.getWeights().get(weightsAmount-1);
                bias += eta * neuron.getDelta();
                neuron.getWeights().set(weightsAmount-1, bias);
            }
        }
    }

    private static List<Double> forwardPropagate(List<List<Neuron>> neuralNetwork, List<Double> initialInputs) {
        List<Double> currentInputs = initialInputs;
        for (List<Neuron> layer : neuralNetwork) {
            List<Double> nextInputs = new ArrayList<>();
            for (Neuron neuron : layer) {
                nextInputs.add(neuron.activate(currentInputs));
            }
            currentInputs = nextInputs;
        }
        return currentInputs;
    }

    private static void backPropagate(List<List<Neuron>> neuralNetwork, List<Double> expectedOutputs) {
        // We iterate the layers in reverse order
        for (int i = neuralNetwork.size()-1; i >= 0; i--) {
            List<Neuron> upperLayerErrors = new ArrayList<>();
            List<Neuron> currentLayer = neuralNetwork.get(i);
            // In this case we calculate the delta of the exit layer
            if (i == neuralNetwork.size() - 1) {
                for (int j = 0; j < expectedOutputs.size(); j++) {
                    Neuron exitNeuron = currentLayer.get(j);
                    Double error = expectedOutputs.get(j) - exitNeuron.getOutput();
                    Double delta = exitNeuron.activationDerivative() * error;
                    exitNeuron.setDelta(delta);
                }
            } else {
                // Here we update the deltas of the hidden layer neurons
                List<Neuron> upperLayer = neuralNetwork.get(i+1);
                for (int j=0; j<currentLayer.size(); j++) {
                    double error = 0.0;
                    Neuron currentLayerNeuron = currentLayer.get(j);
                    for (Neuron upperNeuron : upperLayer) {
                        error += upperNeuron.getWeights().get(j) * upperNeuron.getDelta();
                    }
                    currentLayerNeuron.setDelta(error * currentLayerNeuron.activationDerivative());
                }
            }
        }
    }

    private static List<Double> generateRandomWeights(int amountToGenerate) {
        Random random = new Random();
        return random.doubles(amountToGenerate).boxed().collect(Collectors.toList());
    }

    private static List<Neuron> createALayer(int neuronsAmount, int previousLayerNeuronsAmount) {
        List<Neuron> hiddenLayer = new ArrayList<>(neuronsAmount);
        for (int i = 0; i < neuronsAmount; i++) {
            Neuron neuron = new Neuron(generateRandomWeights(previousLayerNeuronsAmount+1));
            hiddenLayer.add(neuron);
        }
        return hiddenLayer;
    }

    private static void writeTrainingDataToFile(String filename, List<List<Double>> dataToWrite, List<List<Double>> expectedOutputs) throws IOException {
        FileWriter writer = new FileWriter(filename);
        for (int i = 0; i < dataToWrite.size(); i++) {
            for (int j = 0; j < dataToWrite.get(i).size(); j++) {
                writer.write(String.valueOf(dataToWrite.get(i).get(j)));
                writer.write(",");
            }
            writer.write(String.valueOf(expectedOutputs.get(i).get(0)));
            writer.write("\n");
        }

        writer.close();
    }

    private static void writeWeightsToFile(String filename, List<List<Double>> neuronsWeights) throws IOException {
        FileWriter writer = new FileWriter(filename);
        for (List<Double> neuronsWeight : neuronsWeights) {
            for (int j = 0; j < neuronsWeight.size(); j++) {
                writer.write(String.valueOf(neuronsWeight.get(j)));
                if (j < neuronsWeight.size() - 1) {
                    writer.write(",");
                }
            }

            writer.write("\n");
        }

        writer.close();
    }
}
