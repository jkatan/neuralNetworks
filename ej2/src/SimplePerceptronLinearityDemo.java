import java.io.*;
import java.util.Random;
import java.util.Scanner;

public class SimplePerceptronLinearityDemo {
    public static void main(String[] args) throws IOException {
        int linesForTraining = 160;
        String pathToEj2 = "C:\\Users\\Nico\\Documents\\ITBA\\Sistemas de Inteligencia Artificial (SIA)\\TP3\\neuralNetworks\\ej2\\";
        float[][] trainingSet = parseTrainingSetFromFile(pathToEj2 + "TP3-ej2-Conjunto-entrenamiento.txt", linesForTraining);
        float[] expectedOutputs = parseExpectedOutputsFromFile(pathToEj2 + "TP3-ej2-Salida-deseada.txt", linesForTraining);

        writeTrainingDataToFile("linear-trainingData.csv", trainingSet, expectedOutputs);
        System.out.println("Linear test:");
        float[] linearWeights = simpleLinearPerceptron(trainingSet, expectedOutputs, 0.01f, 50);
        writeWeightsToFile("linear-weights.csv", linearWeights);

        System.out.println();

        writeTrainingDataToFile("nonLinear-trainingData.csv", trainingSet, expectedOutputs);
        System.out.println("Non linear test:");
        float[] nonLinearWeights = simpleNonLinearPerceptron(trainingSet, expectedOutputs, 0.01f, 50);
        writeWeightsToFile("nonLinear-weights.csv", nonLinearWeights);
    }

    public static float[] simpleLinearPerceptron(float[][] trainingSet, float[] expectedOutputs, float eta, int maxIterations) {
        int i = 0;
        float[] weights = new float[3];
        float[] deltaWeights;
        float error = 1.0f;

        while (error > 0 && i < maxIterations) {
            Random random = new Random();
            int randomIndex = random.nextInt(trainingSet.length);
            float[] randomTrainingSample = trainingSet[randomIndex];
            // in simple perceptron linear, excitement is equal to activation
            float activation = calculatePerceptronExcitement(randomTrainingSample, weights);
            // and activation impacts in delta error
            float errorProportion = eta * (expectedOutputs[randomIndex] - activation);
            // which indirectly impacts on the cost function
            deltaWeights = calculateDeltaWeights(errorProportion, randomTrainingSample);
            weights = updateWeights(weights, deltaWeights, errorProportion);
            error = calculateLinearError(trainingSet, expectedOutputs, weights);
            i++;
        }

        System.out.println("Iterations: " + i);
        System.out.println("Total error: " + error);
        System.out.println("Weights: ");
        printArray(weights);

        return weights;
    }

    public static float[] simpleNonLinearPerceptron(float[][] trainingSet, float[] expectedOutputs, float eta, int maxIterations) {
        int i = 0;
        float[] weights = new float[3];
        float[] deltaWeights;
        float error = 1.0f;
        double beta = 0.5;

        while (error > 0 && i < maxIterations) {
            Random random = new Random();
            int randomIndex = random.nextInt(trainingSet.length);
            float[] randomTrainingSample = trainingSet[randomIndex];
            // in simple perceptron linear, excitement is equal to activation
            double excitement = calculatePerceptronExcitement(randomTrainingSample, weights);
            float activation = (float) Math.tanh(beta * excitement);
            // and activation impacts in delta error, because now I also need g', which in tanh it's equal to beta*(1-g^2)
            float errorProportion = (float) (eta * (expectedOutputs[randomIndex] - activation) * (beta * (1 - Math.pow(activation, 2))));
            // which indirectly impacts on the cost function
            deltaWeights = calculateDeltaWeights(errorProportion, randomTrainingSample);
            weights = updateWeights(weights, deltaWeights, errorProportion);
            error = calculateNonLinearError(trainingSet, expectedOutputs, weights, beta);
            i++;
        }

        System.out.println("Iterations: " + i);
        System.out.println("Total error: " + error);
        System.out.println("Weights: ");
        printArray(weights);

        return weights;
    }

    private static void writeTrainingDataToFile(String filename, float[][] dataToWrite, float[] expectedOutputs) throws IOException {
        FileWriter writer = new FileWriter(filename);
        for (int i = 0; i < dataToWrite.length; i++) {
            for (int j = 0; j < dataToWrite[i].length; j++) {
                writer.write(String.valueOf(dataToWrite[i][j]));
                writer.write(",");
            }
            writer.write(String.valueOf(expectedOutputs[i]));
            writer.write("\n");
        }

        writer.close();
    }

    private static void writeWeightsToFile(String filename, float[] weights) throws IOException {
        FileWriter writer = new FileWriter(filename);
        for (int i = 0; i < weights.length; i++) {
            writer.write(String.valueOf(weights[i]));
            if (i < weights.length -1) {
                writer.write(",");
            }
        }
        writer.close();
    }

    private static float[] updateWeights(float[] weights, float[] deltaWeights, float errorProportion) {
        float[] newWeights = new float[weights.length];
        for (int i = 0; i < deltaWeights.length; i++) {
            newWeights[i] += weights[i] + deltaWeights[i];
        }
        // The last weight is the bias, so we just sum the errorProportion (equivalent to multiply it by 1)
        newWeights[weights.length-1] += errorProportion;

        return newWeights;
    }

    private static float[] calculateDeltaWeights(float errorProportion, float[] trainingSample) {
        float[] result = new float[trainingSample.length];
        for (int i = 0; i < trainingSample.length; i++) {
            result[i] = trainingSample[i] * errorProportion;
        }
        return result;
    }

    private static float calculatePerceptronExcitement(float[] trainingSample, float[] weights) {
        // The last weight is the bias
        float result = weights[weights.length-1];

        for (int i = 0; i < trainingSample.length; i++) {
            result += trainingSample[i] * weights[i];
        }
        return result;
    }

    private static void printArray(float[] array) {
        System.out.print("(");
        for (int i = 0; i < array.length; i++) {
            System.out.print(array[i]);
            if (i < array.length - 1) {
                System.out.print(", ");
            }
        }
        System.out.println(")");
    }

    private static float calculateLinearError(float[][] trainingSet, float[] expectedOutputs, float[] weights) {
        float totalError = 0.0f;
        int length = expectedOutputs.length;
        for (int i = 0; i < length; i++) {
            float actualOutput = calculatePerceptronExcitement(trainingSet[i], weights);
            totalError += Math.abs(actualOutput - expectedOutputs[i]);
        }
        return totalError;
    }

    private static float calculateNonLinearError(float[][] trainingSet, float[] expectedOutputs, float[] weights, double beta) {
        float totalError = 0.0f;
        for (int i = 0; i < expectedOutputs.length; i++) {
            float actualOutput = (float) Math.tanh(beta * calculatePerceptronExcitement(trainingSet[i], weights));
            totalError += Math.abs(actualOutput - expectedOutputs[i]);
        }
        return totalError;
    }

    private static float[][] parseTrainingSetFromFile(String s, int linesForTraining) throws FileNotFoundException {
        Scanner reader = new Scanner(new File(s));
        float[][] trainingSet = new float[linesForTraining][3];
        for (int i = 0; i < linesForTraining; i++) {
            String line = reader.nextLine();
            String[] parsedLine = line.trim().split(" +");
            for (int j = 0; j < parsedLine.length; j++) {
                trainingSet[i][j] = Float.parseFloat(parsedLine[j]);
            }
        }
        return trainingSet;
    }

    private static float[] parseExpectedOutputsFromFile(String s, int linesForTraining) throws FileNotFoundException {
        Scanner reader = new Scanner(new File(s));
        float[] expectedSet = new float[linesForTraining];
        for (int i = 0; i < linesForTraining; i++) {
            expectedSet[i] = Float.parseFloat(reader.nextLine().trim());
        }
        return expectedSet;
    }

    private static void printTable(float[][] table) {
        System.out.println("(");
        for (int i = 0; i < table.length; i++) {
            printArray(table[i]);
        }
        System.out.println(")");
    }
}
