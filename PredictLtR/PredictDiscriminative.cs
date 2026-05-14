using System;
using Microsoft.ML.Probabilistic.Math;

namespace PredictLtR
{
    class PredictDiscriminative
    {
        readonly Vector w;

        public PredictDiscriminative(Vector wParam)
        {
            w = wParam;
        }

        // Returns rank distribution for each item given its feature vector.
        public double[][] GetRankDistributions(Vector[] features)
        {
            int n = features.Length;
            double[] scores = GetItemScores(features);

            double[][] rankDists = new double[n][];
            for (int i = 0; i < n; i++)
            {
                double[] pairProbs = new double[n - 1];
                int k = 0;
                for (int j = 0; j < n; j++)
                {
                    if (j == i) continue;
                    // P(item i beats item j) = σ(score_i − score_j): logistic link,
                    // consistent with the Plackett-Luce generative model (Gumbel noise).
                    pairProbs[k++] = 1.0 / (1.0 + Math.Exp(scores[j] - scores[i]));
                }
                rankDists[i] = ComputeRankDistribution(pairProbs);
            }
            return rankDists;
        }

        public double[] GetItemScores(Vector[] features)
        {
            double[] scores = new double[features.Length];
            for (int i = 0; i < features.Length; i++)
                scores[i] = Vector.InnerProduct(w, features[i]);
            return scores;
        }

        // O(n²) DP: replaces the original O(2^n) RankFunc recursion.
        // pairProbs[k] = P(item beats its k-th opponent).
        // Returns rankDist where rankDist[r] = P(item ends up at rank r).
        static double[] ComputeRankDistribution(double[] pairProbs)
        {
            int numRanks = pairProbs.Length + 1;
            double[] dp = new double[numRanks];
            dp[0] = 1.0;

            for (int i = 0; i < pairProbs.Length; i++)
            {
                double[] next = new double[numRanks];
                double p = pairProbs[i];
                for (int r = 0; r < numRanks; r++)
                {
                    next[r] += dp[r] * p;                       // beats opponent: rank stays r
                    if (r > 0) next[r] += dp[r - 1] * (1 - p); // loses: rank was r-1
                }
                dp = next;
            }
            return dp;
        }

    }
}
