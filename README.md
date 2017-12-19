# Learning To Rank (LtR)
Learning to Rank (LtR) implementation using Infer.NET

The model implements pairwise preferences using a linear model. The ties are not supported.

The package is composed of two command-line applications:
- Training
- Prediction

The training tool accepts a SVM-Light formatted file (input) and a model filename (output). Some diagnostic information is printed during learning.

The prediction tool accepts a model file (input) and a SVM-Light formatted file (input). The predictions are rank distributions and are printed to the console.

Note: This project uses the Infer.NET framework which has a different license!
