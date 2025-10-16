# Learning to Rank (LtR) with Infer.NET

A Bayesian learning to rank implementation using Microsoft's Infer.NET probabilistic programming framework. This project implements pairwise preference learning using a linear model based on the TrueSkill/Thurstonian ranking approach.

> ⚠️ **SCALABILITY WARNING**: This implementation has severe performance limitations for large queries. The prediction algorithm uses exponential-time recursive computation (O(2^n)) that becomes intractable for queries with more than 10-15 items. For queries with 40+ items (common in real datasets), prediction may stall or take hours to complete. Consider using this only for small-scale experiments or queries with very few items per query.

## Overview

This solution provides two command-line applications for learning to rank:

- **TrainLtR**: Trains a Bayesian ranking model from pairwise preference data
- **PredictLtR**: Generates rank distribution predictions for new queries

The model learns pairwise preferences using a linear model where ties are not supported. It's particularly suitable for scenarios where you have explicit pairwise comparisons between items within queries.

## Prerequisites

- **.NET 8.0 SDK** or later
- **Infer.NET** framework (v0.4.2504.701) - [GitHub](https://github.com/dotnet/infer) (open source, MIT license)

### Platform Support
✅ **Cross-platform**: Windows, Linux, and macOS  
✅ **Modern tooling**: Visual Studio 2022+, VS Code, or JetBrains Rider

> **Note**: This solution has been modernized from .NET Framework 4.6.1 to .NET 8 with the latest Infer.NET framework.

## Algorithm

### Graphical Model
The implementation uses a modified TrueSkill/Thurstonian ranking model with observed feature vectors. The model learns:

- **Feature weights** (`w`): Linear combination weights for ranking features
- **Noise parameters**: Uncertainty in pairwise comparisons

For `K` training examples, each with `n` items, the model generates `m = n-1` pairwise preference observations.

![TrueSkill/Thurstonian model for LtR](https://github.com/usptact/LearningToRank/blob/master/img/LtR%20Graphical%20Model.png)

### Learning Process
1. **Feature extraction**: Convert items to feature vectors
2. **Pairwise comparison**: Generate all possible item pairs within each query
3. **Bayesian inference**: Learn feature weights using variational message passing
4. **Ranking prediction**: Compute probability distributions over possible rankings

## Quick Start

### Building
```bash
git clone <repository-url>
cd LearningToRank
dotnet build
```

### Running
```bash
# Train a model
dotnet run --project TrainLtR -- data/train.small.ltr model.json

# Generate predictions
dotnet run --project PredictLtR -- model.json data/predict.ltr predictions.csv
```

### IDE Support
- **Visual Studio 2022+**: Open `LearningToRank.sln`
- **VS Code**: Install C# extension and open the folder
- **JetBrains Rider**: Open the solution file

## Data Format

### Input Format (SVM-Light)
Training and prediction data must be in SVM-Light format:

```
<rank> qid:<query_id> <feature_id>:<feature_value> ... <feature_id>:<feature_value>
```

**Example:**
```
1 qid:1 1:0.5 2:1.0 3:0.2
2 qid:1 1:0.3 2:0.8 3:0.9
1 qid:2 1:0.7 2:0.1 3:0.4
```

**Requirements:**
- `rank`: Integer ranking (lower = better)
- `qid`: Query identifier (groups items for comparison)
- `feature_id`: Must start from 1 (not 0)
- `feature_value`: Floating-point feature value

### Output Format (CSV)
Predictions are written to CSV with rank distribution probabilities:

```csv
QueryIndex,ItemIndex,Rank0,Rank1,Rank2,Rank3,Rank4,Rank5,Rank6,Rank7,Rank8,Rank9
0,0,0.500222,0.499778,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
0,1,0.499778,0.500222,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
```

## Usage

### Training
```bash
dotnet run --project TrainLtR -- <training_data.ltr> <output_model.json>
```

**Example:**
```bash
dotnet run --project TrainLtR -- data/train.small.ltr model.json
```

### Prediction
```bash
dotnet run --project PredictLtR -- <model.json> <prediction_data.ltr> [output.csv]
```

**Examples:**
```bash
# Use default output filename (predictions.csv)
dotnet run --project PredictLtR -- model.json data/predict.ltr

# Specify custom output filename
dotnet run --project PredictLtR -- model.json data/predict.ltr my_results.csv
```

### Data Preparation Tips
- **For 2-rank queries**: Shuffle items within each query for better pairwise learning
- **For multi-rank queries**: Sort items by rank within each query
- **Feature IDs**: Must start from 1 (not 0)
- **Query grouping**: Items with the same `qid` are compared pairwise

### Output Interpretation
- **CSV format**: Each row represents one item's rank distribution
- **Probabilities**: Sum to 1.0 for each item across all possible ranks
- **Ranking**: Lower rank numbers indicate better positions (rank 0 = best)

## Sample Data

The `/data` folder contains example datasets:
- `train.small.ltr` - Small training set for testing
- `predict.ltr` - Prediction dataset (LETOR MQ2008)
- `test.small.ltr` - Small test set
- `test.sorted.ltr` - Sorted test data

> **Note**: Most datasets are from the LETOR MQ2008 benchmark collection.

## Performance Considerations

### Scalability Limitations
- **Query size**: Optimal for 2-10 items per query
- **Maximum recommended**: 15 items per query
- **Avoid**: Queries with 40+ items (exponential slowdown)

### Optimization Tips
- Use smaller query sizes when possible
- Consider data preprocessing to reduce query complexity
- Monitor memory usage for large datasets
- Use progress reporting to track long-running predictions

## Technical Details

### Modernization (v2.0)
This solution has been modernized from .NET Framework 4.6.1:

| Component | Old Version | New Version |
|-----------|-------------|-------------|
| Framework | .NET Framework 4.6.1 | .NET 8 |
| Infer.NET | 0.3.1810.501 | 0.4.2504.701 |
| Project Format | Legacy .csproj | SDK-style |
| Serialization | BinaryFormatter | JSON |
| Dependencies | packages.config | PackageReference |

### Breaking Changes
- **Model format**: Now uses JSON instead of binary
- **Command line**: Updated for .NET CLI
- **Requirements**: .NET 8 SDK required

The core Bayesian learning to rank algorithm remains unchanged, ensuring compatibility with the original implementation.

