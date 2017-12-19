using MicrosoftResearch.Infer.Maths;
using System;
using System.Collections.Generic;
using System.IO;

namespace Train
{
    public class Reader
    {
        public string fName;

        public double[][][] xdata;
        public int[][] ydata;

        public static int dimFeatures;

        public Reader(string fileName)
        {
            fName = fileName;
        }

        // main entry point
        public Data Read()
        {
            List<List<Item>> dataset = ParseDataset();
            ExtractDataset(dataset);

            Vector[][] features = GetVectorData();
            int[] sizes = GetSizesArray();
            bool[][] pairwise = GetPairwiseRankingData();
            int[] pairwiseSizes = GetPairwiseSizesArray();

            return new Data(features, sizes, pairwise, pairwiseSizes);
        }

        // parse text file into a dataset
        List<List<Item>> ParseDataset()
        {
            string line;

            List<List<Item>> dataset = new List<List<Item>>();
            List<Item> example = new List<Item>();

            //
            // Read lines, parse information and group into examples
            //

            int prev_qid = -1;

            StreamReader file = new StreamReader(fName);

            while ((line = file.ReadLine()) != null)
            {
                ParseLine(line, out int rank, out int qid, out int[] features, out double[] featureValues);

                if (prev_qid == -1)
                    prev_qid = qid;     // first line

                if (prev_qid == qid)
                {
                    // qid for prev and curr lines are the same
                    // continue appending to the current example
                    example.Add(new Item(rank, qid, features, featureValues));
                }
                else
                {
                    // qid changed
                    prev_qid = qid;

                    // append example to the dataset and start a new one
                    dataset.Add(new List<Item>(example));
                    example.Clear();
                    example.Add(new Item(rank, qid, features, featureValues));
                }
            }

            // add last example
            dataset.Add(example);

            file.Close();

            return dataset;
        }

        // extracts parsed dataset into .NET arrays
        void ExtractDataset(List<List<Item>> dataset)
        {
            int numExamples = dataset.Count;

            xdata = new double[numExamples][][];
            ydata = new int[numExamples][];

            for (int i = 0; i < numExamples; i++)
            {
                List<Item> example = dataset[i];

                int numItems = example.Count;

                xdata[i] = new double[numItems][];
                ydata[i] = new int[numItems];

                for (int j = 0; j < numItems; j++)
                {
                    Item item = example[j];

                    ydata[i][j] = item.rank;
                    xdata[i][j] = new double[dimFeatures];

                    int numFeatures = item.features.Length;
                    for (int k = 0; k < numFeatures; k++)
                        xdata[i][j][item.features[k]] = item.feature_values[k];
                }
            }
        }

        // returns feature vectors (includes bias term "1"!)
        Vector[][] GetVectorData()
        {
            int numExamples = xdata.Length;

            Vector[][] data = new Vector[numExamples][];

            for (int i = 0; i < numExamples; i++)
            {
                int numItems = xdata[i].Length;

                data[i] = new Vector[numItems];

                for (int j = 0; j < numItems; j++)
                {
                    double[] featureVector = new double[dimFeatures + 1];                   // features + value 1.0 (bias term)
                    Array.Copy(xdata[i][j], 0, featureVector, 0, xdata[i][j].Length);   // copy features into destination features vector
                    featureVector[dimFeatures] = 1.0;                               // bias term is the last in features vector
                    data[i][j] = Vector.FromArray(featureVector);
                }
            }
            return data;
        }

        // returns pairwise ranks (compares rank values)
        bool[][] GetPairwiseRankingData()
        {
            int numExamples = ydata.Length;

            bool[][] ranks = new bool[numExamples][];

            for (int i = 0; i < numExamples; i++)
            {
                int numItems = ydata[i].Length;

                if (numItems < 2)
                    throw new System.InvalidOperationException("Item size must be at least 2!");

                ranks[i] = new bool[numItems - 1];

                for (int j = 0; j < numItems - 1; j++)
                {
                    int left = ydata[i][j + 1];
                    int right = ydata[i][j];
                    if (left > right)
                        ranks[i][j] = true;
                    else
                        ranks[i][j] = false;
                }
            }
            return ranks;
        }

        // return example sizes
        int[] GetSizesArray()
        {
            int numExamples = xdata.Length;

            int[] sizes = new int[numExamples];

            for (int i = 0; i < numExamples; i++)
            {
                int numItems = xdata[i].Length;
                sizes[i] = numItems;
            }
            return sizes;
        }

        // returns example pair sizes (example size - 1)
        int[] GetPairwiseSizesArray()
        {
            int numExamples = xdata.Length;

            int[] sizes = new int[numExamples];

            for (int i = 0; i < numExamples; i++)
            {
                int numItems = xdata[i].Length;
                sizes[i] = numItems - 1;
            }
            return sizes;
        }

        // parses one string
        void ParseLine(string line, out int rank, out int qid, out int[] features, out double[] featureValues)
        {
            string[] tokens = line.Split(' ');

            // get rank field (int)
            rank = Int32.Parse(tokens[0]);

            // get qid (int)
            string[] qid_parts = tokens[1].Split(':');
            if (qid_parts[0] != "qid")
                throw new System.IndexOutOfRangeException("Invalid data format: Examples must have qid fields!");
            qid = Int32.Parse(qid_parts[1]);

            //
            // get feature-value pairs
            //

            features = new int[tokens.Length - 2];
            featureValues = new double[tokens.Length - 2];

            for (int i = 2; i < tokens.Length; i++)
            {
                string[] parts = tokens[i].Split(':');

                features[i - 2] = Int32.Parse(parts[0]) - 1;
                featureValues[i - 2] = Double.Parse(parts[1]);

                dimFeatures = Math.Max(dimFeatures, features[i - 2] + 1);   // update feature dimensionality as we parse the dataset
            }
        }
    }

    public struct Item
    {
        public int rank;
        public int qid;
        public int[] features;
        public double[] feature_values;

        public Item(int r, int q, int[] f, double[] fv)
        {
            rank = r;
            qid = q;
            features = f;
            feature_values = fv;
        }
    }

    public struct Data
    {
        public Vector[][] features;
        public int[] sizes;
        public bool[][] pairwise;
        public int[] pairwiseSizes;

        public Data(Vector[][] f, int[] s, bool[][] p, int[] ps)
        {
            features = f;
            sizes = s;
            pairwise = p;
            pairwiseSizes = ps;
        }
    }
}
