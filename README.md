# Learning To Rank (LtR)
Learning to Rank (LtR) implementation using Infer.NET

> ⚠️ **SCALABILITY WARNING**: This implementation has severe performance limitations for large queries. The prediction algorithm uses exponential-time recursive computation (O(2^n)) that becomes intractable for queries with more than 10-15 items. For queries with 40+ items (common in real datasets), prediction may stall or take hours to complete. Consider using this only for small-scale experiments or queries with very few items per query.

The model implements pairwise preferences using a linear model. The ties are not supported.

The package is composed of two command-line applications:
- Training
- Prediction

The training tool accepts a SVM-Light formatted file (input) and a model filename (output). Some diagnostic information is printed during learning.

The prediction tool accepts a model file (input) and a SVM-Light formatted file (input). The predictions are rank distributions and are printed to the console.

## Prerequisites
- .NET 8.0 SDK
- [Infer.NET](https://github.com/dotnet/infer) framework (now open source under MIT license)

This solution has been modernized to use .NET 8 and the latest version of Infer.NET (0.4.2504.701).

Cross-platform support: Works on Windows, Linux, and macOS.

## Graphical Model
The graphical model for LtR is the modified TrueSkill/Thurstonian type model. The difference is that the observed feature vectors are introduced.

There are `K` training examples. Each training example has a variable number of items (`n` in figure). There are `m=n-1` pairwise preference observations per example.

Slightly abusing factor graph notation, the random variables `w` and `noise` are inside the plate but with different notation. In the model, there is a single random variable for each that is inferred.
![TrueSkill/Thurstonian model for LtR](https://github.com/usptact/LearningToRank/blob/master/img/LtR%20Graphical%20Model.png)

## Building
This solution uses modern .NET SDK-style projects. To build:

```bash
dotnet build
```

Or open in Visual Studio 2022 or later, or VS Code with C# extension.

## Data Format
Training and prediction data is expected to be in the SVM-Light format.

```
line ::= <rank> qid:<qid> <fid>:<fval> ... <fid>:<fval>
rank ::= <int>
qid ::= <int>
fid ::= <int>
fval ::= <float>
```

Feature ids are expected to start from 1!

## Usage
Prepare training dataset in SVM-Light format. Make sure that the items within an example are shuffled (for 2 ranks only!). This enables the model to better learn pairwise preferences and do prediction on unseen test data (order of items in examples of the dataset are of course unknown apriori). If there are more than 2 ranks, the items in an example must be sorted so that the model can learn the preferences correctly.

Run the training application:
```bash
dotnet run --project TrainLtR -- <train.ltr> <model.json>
```

Run the prediction application:
```bash
dotnet run --project PredictLtR -- <model.json> <predict.ltr>
```

Note: Models are now saved in JSON format instead of binary format for better portability and debugging.

The current application outputs probability ranks for full picture. If only order information is required, the prediction `Program.cs` can be modified to output raw scores. Sorting in descending order will provide ranking.

## Sample Data
Check `/data` folder for data samples. All but smallest data examples are from the LETOR MQ2008 dataset.

## Modernization Notes
This solution has been modernized from the original .NET Framework 4.6.1 version to .NET 8 with the following changes:

- **Framework**: Upgraded to .NET 8
- **Infer.NET**: Updated to latest version (0.4.2504.701)
- **Project Format**: Converted to SDK-style projects
- **Serialization**: Replaced BinaryFormatter with JSON serialization for better security and portability
- **Namespaces**: Updated to use consistent namespace structure
- **Dependencies**: Removed legacy package.config files in favor of PackageReference

The core Bayesian learning to rank algorithm remains unchanged, maintaining compatibility with the original implementation.

