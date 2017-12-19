using System;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;

namespace Predict
{
    class Program
    {
        static void Main(string[] args)
        {
            if (args.Length < 2)
            {
                Console.WriteLine("Usage: Predict.exe <model.bin> <predict.ltr>");
                return;
            }

            string modelFileName = args[0];
            string predictFileName = args[1];

            if (!File.Exists(modelFileName))
            {
                Console.WriteLine("Model file cannot be found!");
                return;
            }

            if (!File.Exists(predictFileName))
            {
                Console.WriteLine("Data file cannot be found!");
                return;
            }

            //
            // Read model
            //

            BinaryFormatter serializer = new BinaryFormatter();

            VectorGaussian wPosteriorDist;
            Gamma scoresNoisePosteriorDist;

            using (FileStream stream = new FileStream(modelFileName, FileMode.Open))
            {
                wPosteriorDist = (VectorGaussian)serializer.Deserialize(stream);
                scoresNoisePosteriorDist = (Gamma)serializer.Deserialize(stream);
            }

            //
            // Prediction
            //

            Reader predictReader = new Reader(predictFileName);
            Data predictData = predictReader.Read();

            int numPredict = predictData.features.Length;

            PredictDiscriminative pDisc = new PredictDiscriminative(wPosteriorDist.GetMean(),
                                                                    scoresNoisePosteriorDist.GetMean());

            for (int i = 0; i < numPredict; i++)
            {
                Vector[] features = predictData.features[i];

                double[][] rankDists = pDisc.GetRankDistributions(features);

                PredictDiscriminative.PrintRankDistributions(rankDists);

                Console.WriteLine("\n");
            }

        }
    }
}
