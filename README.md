# Learning To Rank (LtR)
Learning to Rank (LtR) implementation using Infer.NET

The model implements pairwise preferences using a linear model. The ties are not supported.

The package is composed of two command-line applications:
- Training
- Prediction

The training tool accepts a SVM-Light formatted file (input) and a model filename (output). Some diagnostic information is printed during learning.

The prediction tool accepts a model file (input) and a SVM-Light formatted file (input). The predictions are rank distributions and are printed to the console.

## Prerequisites
The only prerequisite to compile and run the model is the [Infer.NET](http://infernet.azurewebsites.net/) framework. Please note that it has a different and very restrictive license!

Runs just fine with .NET 4.6 under Windows. Can be run under Linux/MacOS with Mono (less tested).

If you would like to modify the model, some familiarity with graphical models (factor graphs in particular) is required.

## Graphical Model
The graphical model for LtR is the modified TrueSkill/Thurstonian type model. The difference is that the observed feature vectors are introduced.

There are `K` training examples. Each training example has a variable number of items (`n` in figure). There are `m=n-1` pairwise preference observations per example.

Slightly abusing factor graph notation, the random variables `w` and `noise` are inside the plate but with different notation. In the model, there is a single random variable for each that is inferred.
![TrueSkill/Thurstonian model for LtR](https://github.com/usptact/LearningToRank/blob/master/img/LtR%20Graphical%20Model.png)

## Building
Open the solution and run build: "Build" -> "Build Solution" or Ctrl+Shift+B. Tested with Visual Studio 2017 Community Edition.

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

Run `Train.exe <train.ltr> <model.bin>` to train a model. Parallel training is enabled by default and may use large amounts of RAM on large datasets!

Run `Predict.exe <model.bin> <predict.ltr>` to predict ranks.

The current application outputs probability ranks for full picture. If only order information is required, the prediction `Program.cs` can be modified to output raw scores. Sorting in descending order will provide ranking.

## Sample Data
Check `/data` folder for data samples. All but smallest data examples are from the LETOR MQ2008 dataset.

## Notes
Updated to use open source Infer.NET.

