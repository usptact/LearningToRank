using System;
using System.IO;
using System.Text;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;

namespace PredictLtR
{
    class Program
    {
        static void Main(string[] args)
        {
            if (args.Length < 2)
            {
                Console.WriteLine("Usage: Predict.exe <model.json> <predict.ltr> [output.csv]");
                Console.WriteLine("If output.csv is not specified, results will be written to predictions.csv");
                return;
            }

            string modelFileName = args[0];
            string predictFileName = args[1];
            string outputFileName = args.Length > 2 ? args[2] : "predictions.csv";

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

            Console.WriteLine("Loading model...");
            var (wPosteriorDist, scoresNoisePosteriorDist) = ModelSerializer.DeserializeModel(modelFileName);

            //
            // Prediction
            //

            Console.WriteLine("Reading prediction data...");
            Reader predictReader = new Reader(predictFileName);
            Data predictData = predictReader.Read();

            int numPredict = predictData.features.Length;
            Console.WriteLine($"Processing {numPredict} queries...");

            PredictDiscriminative pDisc = new PredictDiscriminative(wPosteriorDist.GetMean(),
                                                                    scoresNoisePosteriorDist.GetMean());

            using (var writer = new StreamWriter(outputFileName, false, Encoding.UTF8))
            {
                // Write CSV header
                writer.WriteLine("QueryIndex,ItemIndex,Rank0,Rank1,Rank2,Rank3,Rank4,Rank5,Rank6,Rank7,Rank8,Rank9");

                for (int i = 0; i < numPredict; i++)
                {
                    Vector[] features = predictData.features[i];
                    double[][] rankDists = pDisc.GetRankDistributions(features);

                    // Write rank distributions for each item in this query
                    for (int itemIndex = 0; itemIndex < rankDists.Length; itemIndex++)
                    {
                        var rankDist = rankDists[itemIndex];
                        var csvLine = new StringBuilder();
                        csvLine.Append($"{i},{itemIndex}");
                        
                        // Write up to 10 rank probabilities (pad with 0 if fewer ranks)
                        for (int rank = 0; rank < 10; rank++)
                        {
                            double prob = rank < rankDist.Length ? rankDist[rank] : 0.0;
                            csvLine.Append($",{prob:F6}");
                        }
                        
                        writer.WriteLine(csvLine.ToString());
                    }

                    // Progress reporting every 100 queries
                    if ((i + 1) % 100 == 0 || i == numPredict - 1)
                    {
                        Console.WriteLine($"Processed {i + 1}/{numPredict} queries ({(double)(i + 1) / numPredict * 100:F1}%)");
                    }
                }
            }

            Console.WriteLine($"Prediction complete! Results written to {outputFileName}");
        }
    }
}
