import java.io.*;
import java.util.*;

public class SimplePerceptronLinearityDemo {
    public static void main(String[] args) throws IOException {
        String ej2ConfigPath = "src/config.properties";
        final Properties properties = new Properties();
        properties.load(new FileInputStream(ej2ConfigPath));

        int linesToParse = Integer.parseInt(properties.getProperty("totalLinesToParseFromTrainingFile"));
        int trainingAmount = Integer.parseInt(properties.getProperty("linesUsedToTrain"));
        int testingAmount = Integer.parseInt(properties.getProperty("linesUsedToTest"));
        float[][] inputs = parseTrainingSetFromFile(properties.getProperty("trainingInputs"), linesToParse);
        float[] expectedOutputs = parseExpectedOutputsFromFile(properties.getProperty("expectedOutputs"), linesToParse);

        generateRandomTrainingAndTestingFiles(inputs, expectedOutputs, trainingAmount);

        float[][] trainingSet = parseTrainingSetFromFile(properties.getProperty("randomTrainingInputs"), trainingAmount);
        float[] trainingOutputs = parseExpectedOutputsFromFile(properties.getProperty("respectiveRandomTrainingOutputs"), trainingAmount);

        float[][] testingSet = parseTrainingSetFromFile(properties.getProperty("randomTestingInputs"), testingAmount);
        float[] testingOutputs = parseExpectedOutputsFromFile(properties.getProperty("respectiveRandomTestingOutputs"), testingAmount);

        float[] normalizedTrainingOutputs = normalizeOutputs(trainingOutputs);
        float[] normalizedTestingOutputs = normalizeOutputs(testingOutputs);

        System.out.println("Linear test:");
        float linearEta = Float.parseFloat(properties.getProperty("linearEta"));
        float linearEpsilon = Float.parseFloat(properties.getProperty("linearEpsilon"));
        float momentumAlpha = Float.parseFloat(properties.getProperty("momentumAlpha"));
        float[] linearWeights = simpleLinearPerceptron(trainingSet, normalizedTrainingOutputs, linearEta, linearEpsilon, momentumAlpha);
        System.out.println("Weights obtained:");
        printArray(linearWeights);
        System.out.println();
        System.out.println("Linear model evaluation, using only training set");
        evaluateLinearModel(linearWeights, trainingSet, normalizedTrainingOutputs);

        System.out.println();

        System.out.println("Non linear test:");
        float nonLinearEta = Float.parseFloat(properties.getProperty("nonLinearEta"));
        int epochsAmount = Integer.parseInt(properties.getProperty("epochsAmount"));
        float nonLinearEpsilon = Float.parseFloat(properties.getProperty("nonLinearEpsilon"));
        float[] nonLinearWeights = simpleNonLinearPerceptron(trainingSet, normalizedTrainingOutputs,  nonLinearEta, epochsAmount, nonLinearEpsilon, testingSet, normalizedTestingOutputs);
        System.out.println("Weights obtained:");
        printArray(nonLinearWeights);
        System.out.println();
        System.out.println("Non linear model evaluation, using only testing set");
        evaluateNonLinearModel(nonLinearWeights, trainingSet, normalizedTrainingOutputs);
    }

    public static float[] simpleLinearPerceptron(float[][] trainingSet, float[] expectedOutputs, float eta, float epsilon, float momentumAlpha) {
        int i = 0;
        float[] weights = new float[4];
        for (int k = 0; k < weights.length; k++) {
            Random rand = new Random();
            weights[k] = rand.nextFloat();
        }
        float error = epsilon+1;

        float lastDeltaWeight1 = 0.0f;
        float lastDeltaWeight2 = 0.0f;
        float lastDeltaWeight3 = 0.0f;
        float lastDeltaWeight4 = 0.0f;
        while (error > epsilon) {

            Random random = new Random();
            int randomIndex = random.nextInt(trainingSet.length);
            float[] randomTrainingSample = trainingSet[randomIndex];
            float output = (randomTrainingSample[0] * weights[0]) + (randomTrainingSample[1] * weights[1]) + (randomTrainingSample[2] * weights[2]) + weights[3];
            float localError = expectedOutputs[randomIndex] - output;

            weights[0] += eta * localError * randomTrainingSample[0] + momentumAlpha * lastDeltaWeight1;
            weights[1] += eta * localError * randomTrainingSample[1] + momentumAlpha * lastDeltaWeight2;
            weights[2] += eta * localError * randomTrainingSample[2] + momentumAlpha * lastDeltaWeight3;
            weights[3] += eta * localError + momentumAlpha * lastDeltaWeight4;

            lastDeltaWeight1 = eta * localError * randomTrainingSample[0];
            lastDeltaWeight2 = eta * localError * randomTrainingSample[1];
            lastDeltaWeight3 = eta * localError * randomTrainingSample[2];
            lastDeltaWeight4 = eta * localError;

            error = 0.0f;
            for (int j = 0; j < expectedOutputs.length; j++) {
                float newOutput = (trainingSet[j][0] * weights[0]) + (trainingSet[j][1] * weights[1]) + (trainingSet[j][2] * weights[2]) + weights[3];
                error += Math.pow(newOutput - expectedOutputs[j], 2);
            }
            error *= 0.5f;
            i++;
        }
        return weights;
    }

    public static float[] simpleNonLinearPerceptron(float[][] trainingSet, float[] expectedOutputs, float eta, int epochsAmount, float epsilon, float[][] testingSet, float[] testingOutputs) throws IOException {
        int epochs = 1;
        float[] weights = new float[4];
        float[] deltaWeights;
        float trainingSetError = epsilon+1.0f;
        float testingSetError = epsilon+1.0f;
        // We map each epoch to the error in that epoch
        Map<Integer, Float> trainingErrorsPerEpoch = new HashMap<>();
        Map<Integer, Float> testingErrorsPerEpoch = new HashMap<>();

        for (int k = 0; k < weights.length; k++) {
            Random rand = new Random();
            weights[k] = rand.nextFloat();
        }

        while (trainingSetError > epsilon || epochs < epochsAmount) {

            for (int i = 0; i < expectedOutputs.length; i++) {
                Random random = new Random();
                int randomIndex = random.nextInt(trainingSet.length);
                float[] randomTrainingSample = trainingSet[randomIndex];
                // in simple perceptron linear, excitement is equal to activation
                double excitement = calculatePerceptronExcitement(randomTrainingSample, weights);
                float activation = (float) (1.0f/(1.0f+Math.exp(-excitement)));
                float errorProportion = (eta * (expectedOutputs[randomIndex] - activation) * (activation*(1-activation)));
                // which indirectly impacts on the cost function
                deltaWeights = calculateDeltaWeights(errorProportion, randomTrainingSample);
                weights = updateWeights(weights, deltaWeights, errorProportion);
            }

            trainingSetError = calculateNonLinearError(trainingSet, expectedOutputs, weights);

            float epochTrainingError = calculateAccumulatedRelativeError(trainingSet, expectedOutputs, weights);
            trainingErrorsPerEpoch.put(epochs, epochTrainingError);

            float epochTestingError = calculateAccumulatedRelativeError(testingSet, testingOutputs, weights);
            testingErrorsPerEpoch.put(epochs, epochTestingError);

            epochs++;
        }

        writeEpochErrorsMappingToFile("training-errors-per-epoch.csv", trainingErrorsPerEpoch);
        writeEpochErrorsMappingToFile("testing-errors-per-epoch.csv", testingErrorsPerEpoch);

        return weights;
    }

    private static void writeEpochErrorsMappingToFile(String fileName, Map<Integer, Float> epochErrors) throws IOException {
        FileWriter epochErrorsFile = new FileWriter(fileName);
        epochErrors.forEach((epoch, error) -> {
            try {
                epochErrorsFile.write(epoch + ", " + error + "\n");
            } catch (IOException e) {
                e.printStackTrace();
            }
        });

        epochErrorsFile.close();
    }

    private static float calculateAccumulatedRelativeError(float[][] trainingSet, float[] expectedOutputs, float[] weights) {
        float totalError = 0.0f;
        for (int i = 0; i < expectedOutputs.length; i++) {
            float actualOutput = (float) (1.0f / (1.0 + Math.exp(-calculatePerceptronExcitement(trainingSet[i], weights))));
            totalError += Math.abs(expectedOutputs[i] - actualOutput);
        }
        return totalError/expectedOutputs.length;
    }

    private static void evaluateLinearModel(float[] weights, float[][] inputs, float[] expectedOutputs) {
        int length = expectedOutputs.length;
        float error = 0.0f;
        float errorSum = 0.0f;
        for (int i = 0; i < length; i++) {
            float actualOutput = calculatePerceptronExcitement(inputs[i], weights);
            error += Math.pow(expectedOutputs[i] - actualOutput, 2);
            errorSum += Math.abs(expectedOutputs[i] - actualOutput);
        }
        error *= 0.5f;
        System.out.println("(1/2) * Squared sum error = " +  error);
        System.out.println("Average error = " + (errorSum/expectedOutputs.length));
        System.out.println("Total accumulated error: " + errorSum);
    }

    private static void evaluateNonLinearModel(float[] weights, float[][] inputs, float[] expectedOutputs) {
        int length = expectedOutputs.length;
        float error = 0.0f;
        float errorSum = 0.0f;
        for (int i = 0; i < length; i++) {
            float actualOutput = (float) (1.0f / (1.0 + Math.exp(-calculatePerceptronExcitement(inputs[i], weights))));
            error += Math.pow(expectedOutputs[i] - actualOutput, 2);
            errorSum += Math.abs(expectedOutputs[i] - actualOutput);
        }
        error *= 0.5f;
        System.out.println("(1/2) * Squared sum error = " +  error);
        System.out.println("Average error = " + (errorSum/expectedOutputs.length));
        System.out.println("Total accumulated error: " + errorSum);
    }

    private static float[] normalizeOutputs(float[] outputsToNormalize) {
        float[] normalizedOutputs = new float[outputsToNormalize.length];
        float max = findMax(outputsToNormalize);
        float min = findMin(outputsToNormalize);
        float range = max - min;
        for (int i = 0; i < normalizedOutputs.length; i++) {
            normalizedOutputs[i] = (outputsToNormalize[i] - min)/range;
        }
        return normalizedOutputs;
    }

    private static float findMin(float[] array) {
        float min = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] < min) {
                min = array[i];
            }
        }
        return min;
    }

    private static float findMax(float[] array) {
        float max = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > max) {
                max = array[i];
            }
        }
        return max;
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
            float output = trainingSet[i][0] * weights[0] + trainingSet[i][1] * weights[1] + trainingSet[i][2] * weights[2] + weights[3];
            totalError +=  Math.pow(expectedOutputs[i] - output, 2);
        }
        return totalError * 0.5f;
    }

    private static float calculateNonLinearError(float[][] trainingSet, float[] expectedOutputs, float[] weights) {
        float totalError = 0.0f;
        for (int i = 0; i < expectedOutputs.length; i++) {
            float actualOutput = (float) (1.0f / (1.0 + Math.exp(-calculatePerceptronExcitement(trainingSet[i], weights))));
            totalError += Math.pow(expectedOutputs[i] - actualOutput, 2);
        }
        return totalError*0.5f;
    }

    private static float[][] parseTrainingSetFromFile(String s, int linesToParse) throws FileNotFoundException {
        Scanner reader = new Scanner(new File(s));
        float[][] trainingSet = new float[linesToParse][3];
        for (int i = 0; i < linesToParse; i++) {
            String line = reader.nextLine();
            String[] parsedLine = line.trim().split(" +");
            for (int j = 0; j < parsedLine.length; j++) {
                trainingSet[i][j] = Float.parseFloat(parsedLine[j]);
            }
        }
        return trainingSet;
    }

    private static float[] parseExpectedOutputsFromFile(String s, int linesToParse) throws FileNotFoundException {
        Scanner reader = new Scanner(new File(s));
        float[] expectedSet = new float[linesToParse];
        for (int i = 0; i < linesToParse; i++) {
            expectedSet[i] = Float.parseFloat(reader.nextLine().trim());
        }
        return expectedSet;
    }

    private static void printTable(float[][] table) {
        System.out.println("(");
        for (float[] floats : table) {
            printArray(floats);
        }
        System.out.println(")");
    }

    private static void generateRandomTrainingAndTestingFiles(float[][] trainingInputs, float[] expectedOutputs, int trainingSamplesAmount) throws IOException {
        List<TrainingSample> samples = new ArrayList<>();
        for (int i = 0; i < expectedOutputs.length; i++) {
            samples.add(new TrainingSample(trainingInputs[i], expectedOutputs[i]));
        }
        Collections.shuffle(samples);

        List<TrainingSample> samplesForTraining = new ArrayList<>();
        for (int j = 0; j < trainingSamplesAmount; j++) {
            samplesForTraining.add(samples.remove(0));
        }
        generateSampleFiles(samplesForTraining, "training");

        List<TrainingSample> samplesForTesting = new ArrayList<>(samples);
        generateSampleFiles(samplesForTesting, "testing");
    }

    private static void generateSampleFiles(List<TrainingSample> trainingSamples, String filename) throws IOException {

        FileWriter inputs = new FileWriter(filename + "-inputs.csv");
        FileWriter outputs = new FileWriter(filename + "-outputs.csv");
        for (TrainingSample sample : trainingSamples) {
            float[] trainingSample = sample.getInput();
            float outputSample = sample.getExpectedOutput();

            inputs.write(String.valueOf(trainingSample[0]));
            inputs.write("  ");
            inputs.write(String.valueOf(trainingSample[1]));
            inputs.write("  ");
            inputs.write(String.valueOf(trainingSample[2]));
            inputs.write('\n');

            outputs.write("  " + String.valueOf(outputSample));
            outputs.write('\n');
        }
        inputs.close();
        outputs.close();
    }
}
