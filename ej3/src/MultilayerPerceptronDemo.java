import java.util.*;
import java.util.stream.Collectors;

public class MultilayerPerceptronDemo {

    public static void main(String[] args) {
        List<List<Neuron>> neuralNetwork = createANeuralNetwork(2, 4, 2);
        List<Double> entryInputs = new ArrayList<>();
        entryInputs.add(1.0);
        entryInputs.add(0.0);
        List<Double> outputs = forwardPropagate(neuralNetwork, entryInputs);

        for (List<Neuron> layer : neuralNetwork) {
            System.out.println(layer);
        }

        System.out.println("Outputs: ");
        System.out.println(outputs);
    }

    private static List<List<Neuron>> createANeuralNetwork(int entryLayerInputs, int hiddenLayerNeurons, int exitLayerNeurons) {
        List<List<Neuron>> neuralNetwork = new ArrayList<>();

        List<Neuron> hiddenLayer = createALayer(hiddenLayerNeurons, entryLayerInputs);
        neuralNetwork.add(hiddenLayer);

        List<Neuron> outputLayer = createALayer(exitLayerNeurons, hiddenLayer.size());
        neuralNetwork.add(outputLayer);

        return neuralNetwork;
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
}
