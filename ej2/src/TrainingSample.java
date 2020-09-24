public class TrainingSample {
    private final float[] input;
    private final float expectedOutput;

    public TrainingSample(float[] input, float expectedOutput) {
        this.input = input;
        this.expectedOutput = expectedOutput;
    }

    public float[] getInput() {
        return input;
    }

    public float getExpectedOutput() {
        return expectedOutput;
    }
}
