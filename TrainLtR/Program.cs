using System;
using System.IO;
using Microsoft.ML.Probabilistic.Distributions;

namespace TrainLtR
{
    class Program
    {
        public static void Main(string[] args)
        {
            //
            // Read data
            //

            if (args.Length < 2)
            {
                Console.WriteLine("Usage: Train.exe <train.ltr> <model.json>");
                return;
            }

            string trainFileName = args[0];
            string modelFileName = args[1];

            if (!File.Exists(trainFileName))
            {
                Console.WriteLine("Input file does not exist!");
                return;
            }

            Reader trainReader = new Reader(trainFileName);
            Data trainData = trainReader.Read();

            int dimFeatures = Reader.dimFeatures + 1;

            //
            // Learn model parameters from data
            //

            TrainModel trainModel = new TrainModel(dimFeatures);

            trainModel.Learn(trainData.sizes,
                             trainData.pairwiseSizes,
                             trainData.features,
                             trainData.pairwise,
                             out VectorGaussian wPosteriorDist,
                             out Gamma scoresNoisePosteriorDist);

            var wParam = wPosteriorDist.GetMean();

            //
            // Inference
            //

            Console.WriteLine("w mean posterior: {0}", wParam);
            Console.WriteLine("\nscores noise posterior: {0}", scoresNoisePosteriorDist);

            //
            // Save posterior distributions
            //

            ModelSerializer.SerializeModel(modelFileName, wPosteriorDist, scoresNoisePosteriorDist);
        }
    }
}
