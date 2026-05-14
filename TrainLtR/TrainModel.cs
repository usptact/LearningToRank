using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Models;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace TrainLtR
{
    public class TrainModel
    {
        Variable<int> numExamples;
        VariableArray<int> exampleSize;

        Range example;
        Range item;

        VariableArray<VariableArray<double>, double[][]> scores;
        VariableArray<VariableArray<Vector>, Vector[][]> features;
        VariableArray<int> winner;

        Variable<Vector> w;
        Variable<double> scoresNoise;

        public InferenceEngine engine;

        public TrainModel(int dimFeatures)
        {
            numExamples = Variable.Observed(default(int)).Named(nameof(numExamples));
            example = new Range(numExamples).Named(nameof(example));

            exampleSize = Variable.Observed(default(int[]), example).Named(nameof(exampleSize));
            item = new Range(exampleSize[example]).Named(nameof(item));

            features = Variable.Observed(default(Vector[][]), example, item).Named(nameof(features));
            winner = Variable.Observed(default(int[]), example).Named(nameof(winner));

            scores = Variable.Array(Variable.Array<double>(item), example).Named(nameof(scores));

            w = Variable.Random(new VectorGaussian(
                Vector.Zero(dimFeatures),
                PositiveDefiniteMatrix.Identity(dimFeatures))).Named(nameof(w));
            scoresNoise = Variable.GammaFromShapeAndScale(1.0, 3.0).Named(nameof(scoresNoise));

            using (Variable.ForEach(example))
            {
                using (Variable.ForEach(item))
                {
                    scores[example][item] = Variable.GaussianFromMeanAndPrecision(
                        Variable.InnerProduct(w, features[example][item]), scoresNoise);
                }

                // Plackett-Luce top-1 observation: the best-ranked item wins a softmax draw
                // over all items in the query. This replaces the pairwise boolean observations
                // of the Thurstonian model and uses VMP (softmax factor has no EP messages).
                var probs = Variable.Softmax(scores[example]);
                winner[example].SetTo(Variable.Discrete(probs));
            }

            engine = new InferenceEngine(new VariationalMessagePassing());
            engine.ShowProgress = false;
            engine.Compiler.WriteSourceFiles = false;
            engine.NumberOfIterations = 100;
            engine.Compiler.UseParallelForLoops = true;

            engine.GetCompiledInferenceAlgorithm(w, scoresNoise);
        }

        public void Learn(int[] itemSizesData, Vector[][] featuresData, int[] winnerData,
                          out VectorGaussian wPosterior, out Gamma scoresNoisePosterior)
        {
            numExamples.ObservedValue = itemSizesData.Length;
            exampleSize.ObservedValue = itemSizesData;
            features.ObservedValue = featuresData;
            winner.ObservedValue = winnerData;

            wPosterior = engine.Infer<VectorGaussian>(w);
            scoresNoisePosterior = engine.Infer<Gamma>(scoresNoise);
        }
    }
}
