using System;
using System.Linq;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Models;

namespace Train
{
    public class TrainModel
    {
        Variable<int> numExamples;
        VariableArray<int> exampleSize;
        VariableArray<int> rankSize;

        Range example;  // over N examples in a dataset
        Range item;     // over M items in an example
        Range pair;     // over M-1 item pairs in an example

        VariableArray<VariableArray<double>, double[][]> scores;
        VariableArray<VariableArray<Vector>, Vector[][]> features;

        VariableArray<VariableArray<bool>, bool[][]> ranks;

        Variable<Vector> w;

        Variable<double> scoresNoise;

        public InferenceEngine engine;

        public TrainModel(int dimFeatures)
        {
            //
            // Dataset size
            //

            numExamples = Variable.New<int>();
            example = new Range(numExamples);

            //
            // Jagged 1-D arrays of arrays for item scores and features
            //

            exampleSize = Variable.Array<int>(example);
            item = new Range(exampleSize[example]);

            scores = Variable.Array(Variable.Array<double>(item), example);
            features = Variable.Array(Variable.Array<Vector>(item), example);

            //
            // Jagged 1-D arrays for pairwise item ranks
            //

            rankSize = Variable.Array<int>(example);
            pair = new Range(rankSize[example]);

            ranks = Variable.Array(Variable.Array<bool>(pair), example);

            //
            // Model parameters
            //

            w = Variable.VectorGaussianFromMeanAndVariance(Vector.Zero(dimFeatures),
                                                           PositiveDefiniteMatrix.Identity(dimFeatures)).Named("w");

            //
            // Model
            //

            scoresNoise = Variable.GammaFromShapeAndScale(1.0, 3.0);

            using (Variable.ForEach(example))
            {
                using (Variable.ForEach(item))
                {
                    var mean = Variable.InnerProduct(w, features[example][item]);
                    scores[example][item] = Variable.GaussianFromMeanAndPrecision(mean, scoresNoise);
                }

                using (ForEachBlock pairBlock = Variable.ForEach(pair))
                {
                    var idx = pairBlock.Index;

                    var diff = scores[example][idx + 1] - scores[example][idx];

                    using (Variable.If(diff > 0))
                        ranks[example][pair] = true;
                    using (Variable.IfNot(diff > 0))
                        ranks[example][pair] = false;
                }
            }

            //
            // Inference engine
            //

            engine = new InferenceEngine();
            engine.NumberOfIterations = 50;
            engine.Compiler.UseParallelForLoops = true;
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

            // main model parameter inference
            wPosterior = engine.Infer<VectorGaussian>(w);

            // aux model parameter inference
            scoresNoisePosterior = engine.Infer<Gamma>(scoresNoise);
        }
    }
}
