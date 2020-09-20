import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

public class SimplePerceptronDemo {
    public static void main(String[] args) throws IOException {

        // AND test
        float[][] andTrainingSet = {{-1,1},{1,-1},{-1,-1},{1,1}};
        float[] andExpectedOutputs = {-1, -1, -1, 1};
        writeTrainingDataToFile("AND-trainingData.csv", andTrainingSet, andExpectedOutputs);
        System.out.println("AND test");
        float[] andWeights = simplePerceptron(andTrainingSet, andExpectedOutputs, 0.01f, 50);
        writeWeightsToFile("AND-weights.csv", andWeights);

        System.out.println();

        // XOR test
        float[][] xorTrainingSet = {{-1,1},{1,-1},{-1,-1},{1,1}};
        float[] xorExpectedOutputs = {1, 1, -1, -1};
        writeTrainingDataToFile("XOR-trainingData.csv", xorTrainingSet, xorExpectedOutputs);
        System.out.println("XOR test");
        float[] xorWeights = simplePerceptron(xorTrainingSet, xorExpectedOutputs, 0.01f, 50);
         writeWeightsToFile("XOR-weights.csv", xorWeights);
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
            float excitement = calculatePerceptronExcitement(randomTrainingSample, weights);
            float activation = signFunction(excitement);
            float errorProportion = eta * (expectedOutputs[randomIndex] - activation);
            deltaWeights = calculateDeltaWeights(errorProportion, randomTrainingSample);
            weights = updateWeights(weights, deltaWeights, errorProportion);
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
            float actualOutput = signFunction(calculatePerceptronExcitement(trainingSet[i], weights));
            totalError += Math.abs(actualOutput - expectedOutputs[i]);
        }
        return totalError;
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

    private static float signFunction(float parameter) {
        if (parameter > 0) return 1;
        if (parameter < 0) return -1;
        return 0;
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
}
