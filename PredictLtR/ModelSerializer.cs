using System;
using System.IO;
using System.Text.Json;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;

namespace PredictLtR
{
    public static class ModelSerializer
    {
        public static (VectorGaussian wPosterior, Gamma scoresNoisePosterior) DeserializeModel(string fileName)
        {
            string json = File.ReadAllText(fileName);
            var modelData = JsonSerializer.Deserialize<ModelData>(json);

            var wMean = Vector.FromArray(modelData!.WPosteriorMean);
            
            // Reconstruct the 2D variance matrix from flattened array
            var varianceMatrix = new double[modelData.WPosteriorVarianceRows, modelData.WPosteriorVarianceCols];
            int index = 0;
            for (int i = 0; i < modelData.WPosteriorVarianceRows; i++)
            {
                for (int j = 0; j < modelData.WPosteriorVarianceCols; j++)
                {
                    varianceMatrix[i, j] = modelData.WPosteriorVariance[index++];
                }
            }
            
            var wVariance = new PositiveDefiniteMatrix(varianceMatrix);
            var wPosterior = VectorGaussian.FromMeanAndVariance(wMean, wVariance);

            var scoresNoisePosterior = Gamma.FromShapeAndRate(modelData.ScoresNoiseShape, modelData.ScoresNoiseRate);

            return (wPosterior, scoresNoisePosterior);
        }

        private class ModelData
        {
            public double[] WPosteriorMean { get; set; } = Array.Empty<double>();
            public double[] WPosteriorVariance { get; set; } = Array.Empty<double>();
            public int WPosteriorVarianceRows { get; set; }
            public int WPosteriorVarianceCols { get; set; }
            public double ScoresNoiseShape { get; set; }
            public double ScoresNoiseRate { get; set; }
        }
    }
}
