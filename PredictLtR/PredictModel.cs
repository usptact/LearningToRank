using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Models;

namespace Predict
{
    public class PredictModel
    {
        Variable<int> numExamples;
        VariableArray<int> exampleSize;
        VariableArray<int> rankSize;

        Range example;  // over examples in a dataset
        Range item;     // over items in an examples
        Range rank;     // over item pairs

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
            //
            // Dataset size
            //

            numExamples = Variable.New<int>();
            example = new Range(numExamples);

            //
            // Jagged arrays for (items, features)
            //

            exampleSize = Variable.Array<int>(example);
            item = new Range(exampleSize[example]);

            scores = Variable.Array(Variable.Array<double>(item), example);
            features = Variable.Array(Variable.Array<Vector>(item), example);

            //
            // Jagged array for item pair ranks
            //

            rankSize = Variable.Array<int>(example);
            rank = new Range(rankSize[example]);

            ranks = Variable.Array(Variable.Array<bool>(rank), example);

            //
            // Model parameters
            //

            wPrior = Variable.New<VectorGaussian>();
            w = Variable.Random<Vector, VectorGaussian>(wPrior);

            //
            // Model
            //

            scoresNoisePrior = Variable.New<Gamma>();
            scoresNoise = Variable.Random<double, Gamma>(scoresNoisePrior);

            using (Variable.ForEach(example))
            {
                using (Variable.ForEach(item))
                {
                    var mean = Variable.InnerProduct(w, features[example][item]);
                    scores[example][item] = Variable.GaussianFromMeanAndPrecision(mean, scoresNoise);
                }

                using (ForEachBlock pairBlock = Variable.ForEach(rank))
                {
                    var idx = pairBlock.Index;
                    var diff = scores[example][idx + 1] - scores[example][idx];

                    var positiveDiff = diff > 0;

                    using (Variable.If(positiveDiff))
                        ranks[example][rank].SetTo(Variable.Bernoulli(0.999));
                    using (Variable.IfNot(positiveDiff))
                        ranks[example][rank].SetTo(Variable.Bernoulli(0.001));
                }
            }

            //
            // Inference engine
            //

            engine = new InferenceEngine();
            engine.NumberOfIterations = 5;
            engine.Compiler.UseParallelForLoops = false;
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
