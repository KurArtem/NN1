import java.util.Arrays;

import static java.lang.Math.*;
import static java.lang.System.*;

public class Main {
    static double[] startWeights = { 1, 3 };
    static double[] biasWeights = {2, 4, -5};
    static double[] firstWeights = startWeights.clone();
    static double[] secondWeights = startWeights.clone();
    static double[] finalWeights = startWeights.clone();

    static double[][] xTrain = {
            {0, 0},
            {1, 0},
            {0, 1},
            {1, 1}
    };
    static double[] yTrainFirst = {0, 1, 1, 1};
    static double[] yTrainSecond = {0, 0, 1, 1};
    static double[] yTrainThird = {0, 0, 0, 1};

    public static void main(String[] args) {
        fitNetwork();
    }

    private static double output(double[] x, double[] weights, int neuronId) {
        double result = 1;
        for (int i = 0; i < x.length; i++) {
            result += x[i] * weights[i];
        }
        result += biasWeights[neuronId];

        return ((exp(result) - exp(-result)) / (exp(result) + exp(-result))) >= 0
                ? 1
                : 0;
    }

    private static void fitNetwork() {
        int age = 1;
        boolean firstFitted = false;
        boolean secondFitted = false;
        boolean finalFitted = false;
        double[] epochPredicts = new double[4];

        while (!(firstFitted && secondFitted && finalFitted)) {
            out.println("age: " + age);
            double[] fPr = firstWeights;
            double[] sPr = secondWeights;
            double[] fnlPr = finalWeights;
            for (int i = 0; i < yTrainFirst.length; i++) {
                double[] xCurr = xTrain[i];
                double[] xFinal = new double[] {
                        output(xCurr, Main.firstWeights, 0),
                        output(xCurr, Main.secondWeights, 1)
                };

                epochPredicts[i] = output(xFinal, finalWeights, 2);
                fitNeuron(xCurr, firstWeights, yTrainFirst[i], 0);
                fitNeuron(xCurr, secondWeights, yTrainSecond[i], 1);
                fitNeuron(xFinal, Main.finalWeights, yTrainThird[i], 2);
            }

            double lossSum = 0;
            for (int j = 0; j < epochPredicts.length; j++) {
                lossSum += pow(yTrainThird[j] - epochPredicts[j], 2);
            }
            out.println("loss: " + lossSum / yTrainThird.length);

            firstFitted = fPr == firstWeights;
            if (firstFitted) { out.println("Первый нейрон обучен за " + age + " эпох"); }
            secondFitted = sPr == secondWeights;
            if (secondFitted) { out.println("Второй нейрон обучен за " + age + " эпох"); }
            finalFitted = fnlPr == finalWeights;
            if (finalFitted) { out.println("Третий нейрон обучен за " + age + " эпох"); }
            age ++;
        }
    }

    private static void fitNeuron(double[] x, double[] weights, double y, int neuronId) {
        double yPredicted = output(x, weights, neuronId);
        if (yPredicted != y) {
            if (yPredicted == 0) {
                for (int i = 0; i < weights.length; i++) {
                    if (weights[i] == 1) {
                        weights[i] += 1;
                    }
                }
                biasWeights[neuronId] = biasWeights[neuronId] == 1
                        ? biasWeights[neuronId] + 1
                        : biasWeights[neuronId];
            }
            else {
                for (int i = 0; i < weights.length; i++) {
                    if (weights[i] == 1) {
                        weights[i] -= 1;
                    }
                }
                biasWeights[neuronId] = biasWeights[neuronId] == 1
                        ? biasWeights[neuronId] - 1
                        : biasWeights[neuronId];
            }
        }
        out.println("[ " + weights[0] + ", " + weights[1] + ", " + Main.biasWeights[neuronId] + " ]");
    }
}
