using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Models;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace PredictLtR
{
    public class PredictModel
    {
        Variable<int> numExamples;
        VariableArray<int> exampleSize;
        VariableArray<int> rankSize;

        Range example;
        Range item;
        Range rank;

        VariableArray<VariableArray<double>, double[][]> scores;
        VariableArray<VariableArray<Vector>, Vector[][]> features;
        VariableArray<VariableArray<bool>, bool[][]> ranks;

        Variable<VectorGaussian> wPrior;
        Variable<Vector> w;

        Variable<Gamma> scoresNoisePrior;
        Variable<double> scoresNoise;

        public InferenceEngine engine;

        public PredictModel()
        {
            numExamples = Variable.Observed(default(int)).Named(nameof(numExamples));
            example = new Range(numExamples).Named(nameof(example));

            exampleSize = Variable.Observed(default(int[]), example).Named(nameof(exampleSize));
            item = new Range(exampleSize[example]).Named(nameof(item));

            rankSize = Variable.Observed(default(int[]), example).Named(nameof(rankSize));
            rank = new Range(rankSize[example]).Named(nameof(rank));

            features = Variable.Observed(default(Vector[][]), example, item).Named(nameof(features));
            scores = Variable.Array(Variable.Array<double>(item), example).Named(nameof(scores));
            ranks = Variable.Array(Variable.Array<bool>(rank), example).Named(nameof(ranks));

            wPrior = Variable.New<VectorGaussian>().Named(nameof(wPrior));
            w = Variable<Vector>.Random(wPrior).Named(nameof(w));

            scoresNoisePrior = Variable.New<Gamma>().Named(nameof(scoresNoisePrior));
            scoresNoise = Variable<double>.Random(scoresNoisePrior).Named(nameof(scoresNoise));

            using (Variable.ForEach(example))
            {
                using (Variable.ForEach(item))
                {
                    scores[example][item] = Variable.GaussianFromMeanAndPrecision(
                        Variable.InnerProduct(w, features[example][item]), scoresNoise);
                }

                using (ForEachBlock pairBlock = Variable.ForEach(rank))
                {
                    var idx = pairBlock.Index;
                    var diff = scores[example][idx + 1] - scores[example][idx];
                    var positiveDiff = (diff > 0).Named("positiveDiff");

                    // Soft constraints improve EP convergence vs. a hard bool assignment.
                    using (Variable.If(positiveDiff))
                        ranks[example][rank].SetTo(Variable.Bernoulli(0.999));
                    using (Variable.IfNot(positiveDiff))
                        ranks[example][rank].SetTo(Variable.Bernoulli(0.001));
                }
            }

            engine = new InferenceEngine();
            engine.ShowProgress = false;
            engine.Compiler.WriteSourceFiles = false;
            engine.NumberOfIterations = 5;
            engine.Compiler.UseParallelForLoops = false;

            engine.GetCompiledInferenceAlgorithm(ranks);
        }

        public void SetPriors(VectorGaussian wPriorDist, Gamma scoresNoisePriorDist)
        {
            wPrior.ObservedValue = wPriorDist;
            scoresNoisePrior.ObservedValue = scoresNoisePriorDist;
        }

        public void Predict(int[] itemSizesData, int[] pairwiseSizesData,
                            Vector[][] featuresData,
                            out Bernoulli[][] ranksPosterior)
        {
            numExamples.ObservedValue = itemSizesData.Length;
            exampleSize.ObservedValue = itemSizesData;
            rankSize.ObservedValue = pairwiseSizesData;
            features.ObservedValue = featuresData;

            ranksPosterior = engine.Infer<Bernoulli[][]>(ranks);
        }
    }
}
