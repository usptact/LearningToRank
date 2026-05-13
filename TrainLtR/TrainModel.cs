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
        VariableArray<int> rankSize;

        Range example;
        Range item;
        Range pair;

        VariableArray<VariableArray<double>, double[][]> scores;
        VariableArray<VariableArray<Vector>, Vector[][]> features;
        VariableArray<VariableArray<bool>, bool[][]> ranks;

        Variable<Vector> w;
        Variable<double> scoresNoise;

        public InferenceEngine engine;

        public TrainModel(int dimFeatures)
        {
            // Dataset size — declared as observed (data shape is known before inference).
            numExamples = Variable.Observed(default(int)).Named(nameof(numExamples));
            example = new Range(numExamples).Named(nameof(example));

            exampleSize = Variable.Observed(default(int[]), example).Named(nameof(exampleSize));
            item = new Range(exampleSize[example]).Named(nameof(item));

            rankSize = Variable.Observed(default(int[]), example).Named(nameof(rankSize));
            pair = new Range(rankSize[example]).Named(nameof(pair));

            // Observed data: declared as Variable.Observed so the compiler knows they are data, not latent.
            features = Variable.Observed(default(Vector[][]), example, item).Named(nameof(features));
            ranks = Variable.Observed(default(bool[][]), example, pair).Named(nameof(ranks));

            // Latent variable (inferred).
            scores = Variable.Array(Variable.Array<double>(item), example).Named(nameof(scores));

            // Model parameters with priors.
            w = Variable.Random(new VectorGaussian(
                Vector.Zero(dimFeatures),
                PositiveDefiniteMatrix.Identity(dimFeatures))).Named(nameof(w));
            scoresNoise = Variable.GammaFromShapeAndScale(1.0, 3.0).Named(nameof(scoresNoise));

            // Generative model.
            using (Variable.ForEach(example))
            {
                using (Variable.ForEach(item))
                {
                    scores[example][item] = Variable.GaussianFromMeanAndPrecision(
                        Variable.InnerProduct(w, features[example][item]), scoresNoise);
                }

                using (ForEachBlock pairBlock = Variable.ForEach(pair))
                {
                    var idx = pairBlock.Index;
                    var diff = scores[example][idx + 1] - scores[example][idx];
                    ranks[example][pair] = (diff > 0);
                }
            }

            engine = new InferenceEngine();
            engine.ShowProgress = false;
            engine.Compiler.WriteSourceFiles = false;
            engine.NumberOfIterations = 50;
            engine.Compiler.UseParallelForLoops = true;

            // Pre-compile to avoid compilation cost on the first Learn() call.
            engine.GetCompiledInferenceAlgorithm(w);
        }

        public void Learn(int[] itemSizesData, int[] pairwiseSizesData,
                          Vector[][] featuresData, bool[][] pairwiseData,
                          out VectorGaussian wPosterior,
                          out Gamma scoresNoisePosterior)
        {
            numExamples.ObservedValue = itemSizesData.Length;
            exampleSize.ObservedValue = itemSizesData;
            rankSize.ObservedValue = pairwiseSizesData;
            features.ObservedValue = featuresData;
            ranks.ObservedValue = pairwiseData;

            wPosterior = engine.Infer<VectorGaussian>(w);
            scoresNoisePosterior = engine.Infer<Gamma>(scoresNoise);
        }
    }
}
