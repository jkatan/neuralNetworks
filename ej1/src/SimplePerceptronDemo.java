import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

public class SimplePerceptronDemo {
    public static void main(String[] args) throws IOException {

        // AND test
        float[][] andTrainingSet = {{-1,1,1},{1,-1,1},{-1,-1,1},{1,1,1}};
        float[] andExpectedOutputs = {-1, -1, -1, 1};
        writeTrainingDataToFile("AND-trainingData.csv", andTrainingSet, andExpectedOutputs);
        System.out.println("AND test");
        float[] andWeights = simplePerceptron(andTrainingSet, andExpectedOutputs, 0.01f, 50);
        writeWeightsToFile("AND-weights.csv", andWeights);

        System.out.println();

        // XOR test
        float[][] xorTrainingSet = {{-1,1,1},{1,-1,1},{-1,-1,1},{1,1,1}};
        float[] xorExpectedOutputs = {1, 1, -1, -1};
        writeTrainingDataToFile("XOR-trainingData.csv", xorTrainingSet, xorExpectedOutputs);
        System.out.println("XOR test");
        float[] xorWeights = simplePerceptron(xorTrainingSet, xorExpectedOutputs, 0.01f, 50);
         writeWeightsToFile("XOR-weights.csv", xorWeights);
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

    private static float[] simplePerceptron(float[][] trainingSet, float[] expectedOutputs, float eta, int maxIterations) {
        int i = 0;
        float[] weights = new float[3];
        float[] deltaWeights;
        float error = 1.0f;

        while (error > 0 && i < maxIterations) {
            Random random = new Random();
            int randomIndex = random.nextInt(trainingSet.length);
            float[] randomTrainingSample = trainingSet[randomIndex];
            float excitacion = dotProduct(randomTrainingSample, weights);
            float activacion = signFunction(excitacion);
            deltaWeights = multiplyVectorByScalar(eta * (expectedOutputs[randomIndex] - activacion), randomTrainingSample);
            weights = sumVectors(weights, deltaWeights);
            error = calculateError(trainingSet, expectedOutputs, weights);
            i += 1;
        }

        System.out.println("Iterations: " + i);
        System.out.println("Total error: " + error);
        System.out.println("Weights: ");
        printArray(weights);

        return weights;
    }

    private static float calculateError(float[][] trainingSet, float[] expectedOutputs, float[] weights) {
        float totalError = 0.0f;
        for (int i = 0; i < expectedOutputs.length; i++) {
            float actualOutput = signFunction(dotProduct(trainingSet[i], weights));
            totalError += Math.abs(actualOutput - expectedOutputs[i]);
        }
        return totalError;
    }

    private static float[] sumVectors(float[] aVector, float[] anotherVector) {
        float[] newVector = new float[aVector.length];
        for (int i = 0; i < newVector.length; i++) {
            newVector[i] += aVector[i] + anotherVector[i];
        }
        return newVector;
    }

    private static float[] multiplyVectorByScalar(float scalar, float[] vector) {
        float[] result = new float[vector.length];
        for (int i = 0; i < result.length; i++) {
            result[i] = vector[i] * scalar;
        }
        return result;
    }

    private static float signFunction(float parameter) {
        if (parameter > 0) return 1;
        if (parameter < 0) return -1;
        return 0;
    }

    private static float dotProduct(float[] anArray, float[] otherArray) {
        float result = 0.0f;
        float currentProduct = 0.0f;
        for (int i = 0; i < anArray.length; i++) {
            currentProduct = anArray[i] * otherArray[i];
            result += currentProduct;
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
}
