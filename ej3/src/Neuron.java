import java.util.List;

public class Neuron {

    private final List<Double> weights;
    private Double output;

    public Neuron(List<Double> weights) {
        this.weights = weights;
        output = 0.0;
    }

    // Here we use the sigmoid activation function
    public Double activate(List<Double> inputs) {
        Double neuronExcitement = excite(inputs);
        output = 1.0 / (1.0 + Math.exp(-neuronExcitement));
        return output;
    }

    private Double excite(List<Double> inputs) {
        // We assume that the bias is the weight in the last position of the list
        Double neuronExcitement = this.weights.get(this.weights.size()-1);
        for (int i = 0; i < inputs.size(); i++) {
            neuronExcitement += this.weights.get(i) * inputs.get(i);
        }

        return neuronExcitement;
    }

    public List<Double> getWeights() {
        return weights;
    }

    public Double getOutput() {
        return output;
    }

    public void updateWeights(List<Double> deltaWeights) {

    }

    @Override
    public String toString() {
        return "Neuron{" +
                "weights=" + weights +
                ", output=" + output +
                '}';
    }
}
