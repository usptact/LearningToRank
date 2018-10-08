using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Distributions;

namespace Predict
{
    class PredictDiscriminative
    {
        Vector w;
        double noise;

        Gaussian XPrior;
        Gaussian YPrior;
        Variable<Gaussian> XParam;
        Variable<Gaussian> YParam;
        Variable<double> X;
        Variable<double> Y;
        Variable<bool> XBeatsY;

        InferenceEngine engine;

        public PredictDiscriminative(Vector wParam, double scoreNoise)
        {
            w = wParam;
            noise = scoreNoise;

            engine = new InferenceEngine();
            engine.ShowProgress = false;

            XPrior = new Gaussian(1.0, 1.0);
            XParam = Variable.Observed(XPrior);
            X = Variable.Random<double, Gaussian>(XParam);

            YPrior = new Gaussian(1.0, 1.0);
            YParam = Variable.Observed(YPrior);
            Y = Variable.Random<double, Gaussian>(YParam);

            XBeatsY = X > Y;
        }

        // returns rank distribution for each item (feature vector)
        public double[][] GetRankDistributions(Vector[] features)
        {
            double[][] pairProbabilities = PairProbabilities(features);
            double[][] rankDistributions = RankDistributions(pairProbabilities);
            return rankDistributions;
        }

        // returns raw score for each item (feature vector)
        public double[] GetItemScores(Vector[] features)
        {
            double[] scores = new double[features.Length];
            for (int i = 0; i < features.Length; i++)
                scores[i] = Vector.InnerProduct(w, features[i]);
            return scores;
        }

        //
        // Private helper functions
        //

        double[][] PairProbabilities(Vector[] features)
        {
            int numItems = features.Length;
            int numPairs = numItems - 1;

            double[][] pairProbabilities = new double[numItems][];

            for (int i = 0; i < numItems; i++)
            {
                pairProbabilities[i] = new double[numPairs];
                int ptr = 0;

                double scoreA = Vector.InnerProduct(w, features[i]);

                for (int j = 0; j < numItems; j++)
                {
                    if (i == j)
                        continue;

                    double scoreB = Vector.InnerProduct(w, features[j]);

                    pairProbabilities[i][ptr] = ComputePairProbability(scoreA, scoreB);
                    ptr++;
                }
            }

            return pairProbabilities;
        }

        double[][] RankDistributions(double[][] pairProbs)
        {
            int numItems = pairProbs.Length;
            int numRanks = numItems;

            double[][] rankDists = new double[numItems][];

            for (int i = 0; i < numItems; i++)
            {
                rankDists[i] = new double[numRanks];

                for (int r = 0; r < numItems; r++)
                    rankDists[i][r] = RankFunc(r, numItems - 2, pairProbs[i]);
            }

            return rankDists;
        }

        double ComputePairProbability(double scoreA, double scoreB)
        {
            XPrior = new Gaussian(scoreA, noise);
            YPrior = new Gaussian(scoreB, noise);

            XParam.ObservedValue = XPrior;
            YParam.ObservedValue = YPrior;

            Bernoulli probXBeatsY = engine.Infer<Bernoulli>(XBeatsY);

            return probXBeatsY.GetProbTrue();
        }

        //
        // static helper functions
        //

        public static void PrintRankDistributions(double[][] rankDists)
        {
            for (int i = 0; i < rankDists.Length; i++)
            {
                for (int j = 0; j < rankDists[i].Length; j++ )
                {
                    string line = string.Format("{0:0.000}", rankDists[i][j]);
                    System.Console.Write(line + " ");
                }
                System.Console.WriteLine();
            }
        }

        public static double RankFunc(int r, int i, double[] p)
        {
            if (r < 0)
                return 0.0;
            if (i == -1)
            {
                if (r == 0)
                    return 1.0;
                else
                    return 0.0;
            }
            else
                return RankFunc(r - 1, i - 1, p) * (1 - p[i]) + RankFunc(r, i - 1, p) * p[i];
        }

    }
}
