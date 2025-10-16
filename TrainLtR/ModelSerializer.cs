using System;
using System.IO;
using System.Text.Json;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;

namespace TrainLtR
{
    public static class ModelSerializer
    {
        public static void SerializeModel(string fileName, VectorGaussian wPosterior, Gamma scoresNoisePosterior)
        {
            var wMean = wPosterior.GetMean();
            var wVariance = wPosterior.GetVariance();
            
            var modelData = new ModelData
            {
                WPosteriorMean = new double[wMean.Count],
                WPosteriorVariance = new double[wVariance.Rows * wVariance.Cols],
                WPosteriorVarianceRows = wVariance.Rows,
                WPosteriorVarianceCols = wVariance.Cols,
                ScoresNoiseShape = scoresNoisePosterior.Shape,
                ScoresNoiseRate = scoresNoisePosterior.Rate
            };

            // Copy mean vector
            for (int i = 0; i < wMean.Count; i++)
            {
                modelData.WPosteriorMean[i] = wMean[i];
            }

            // Copy variance matrix (flattened)
            int index = 0;
            for (int i = 0; i < wVariance.Rows; i++)
            {
                for (int j = 0; j < wVariance.Cols; j++)
                {
                    modelData.WPosteriorVariance[index++] = wVariance[i, j];
                }
            }

            var options = new JsonSerializerOptions
            {
                WriteIndented = true
            };

            string json = JsonSerializer.Serialize(modelData, options);
            File.WriteAllText(fileName, json);
        }

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
