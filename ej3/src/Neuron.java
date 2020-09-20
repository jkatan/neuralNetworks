import java.util.List;

public class Neuron {

    private final List<Double> weights;
    private Double output;
    private Double delta;

    public Neuron(List<Double> weights) {
        this.weights = weights;
        this.output = 0.0;
        this.delta = 0.0;
    }

    // Here we use the sigmoid activation function
    public Double activate(List<Double> inputs) {
        Double neuronExcitement = excite(inputs);
        output = 1.0 / (1.0 + Math.exp(-neuronExcitement));
        return output;
    }

    // This is the derivative of the sigmoid function
    public Double activationDerivative() {
        return this.output * (1.0 - this.output);
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

    public Double getDelta() {
        return delta;
    }

    public void setDelta(Double delta) {
        this.delta = delta;
    }

    @Override
    public String toString() {
        return "Neuron{" +
                "weights=" + weights +
                ", output=" + output +
                ", delta=" + delta +
                '}';
    }
}
