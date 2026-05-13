# Learning to Rank (LtR) with Infer.NET

A Bayesian learning-to-rank implementation using Microsoft's [Infer.NET](https://github.com/dotnet/infer) probabilistic programming framework. Implements pairwise preference learning with a linear TrueSkill/Thurstonian model.

> ⚠️ **Scalability**: Prediction is O(n²) per query where n is the number of items. Optimal for 2–10 items per query; avoid queries with 40+ items.

## Prerequisites

- .NET 10.0 SDK
- Infer.NET v0.4.2504.701 (restored automatically via NuGet)

## Quick Start

```bash
git clone <repository-url>
cd LearningToRank
dotnet build

# Train
dotnet run --project TrainLtR -- data/train.small.ltr model.json

# Predict
dotnet run --project PredictLtR -- model.json data/predict.ltr predictions.csv
```

## Algorithm

The model uses a modified TrueSkill/Thurstonian graphical model with observed feature vectors. It learns:

- **`w`** — feature weight vector (VectorGaussian posterior)
- **noise** — pairwise comparison uncertainty (Gamma posterior)

For each query with `n` items, `n-1` pairwise preference observations are generated. Training uses variational message passing (50 iterations). Prediction computes pairwise win probabilities via `Φ((sᵢ - sⱼ) / √(2·noise))` and converts them to rank distributions via O(n²) DP.

![TrueSkill/Thurstonian model for LtR](https://github.com/usptact/LearningToRank/blob/master/img/LtR%20Graphical%20Model.png)

## Data Format

**Input** — SVM-Light format (feature IDs start at 1):
```
<rank> qid:<query_id> <feature_id>:<value> ...
```
```
1 qid:1 1:0.5 2:1.0 3:0.2
2 qid:1 1:0.3 2:0.8 3:0.9
```

- Lower rank = better position
- Items sharing a `qid` are compared pairwise
- For 2-rank queries: shuffle items within each query; for multi-rank: sort by rank

**Output** — CSV with per-item rank probability distributions (rank 0 = best):
```
QueryIndex,ItemIndex,Rank0,Rank1,...,Rank9
0,0,0.500222,0.499778,0.000000,...
```

## Usage

```bash
# Training
dotnet run --project TrainLtR -- <train.ltr> <model.json>

# Prediction (output defaults to predictions.csv)
dotnet run --project PredictLtR -- <model.json> <predict.ltr> [output.csv]
```

If the prediction file has more features than the training file, extra features are automatically ignored to match the model dimension.

## Sample Data

The `data/` folder contains LETOR MQ2008 benchmark datasets:

| File | Description |
|------|-------------|
| `train.small.ltr` | Small training set |
| `train.ltr` | Full training set |
| `predict.ltr` | Prediction set |
| `test.small.ltr` / `test.sorted.ltr` | Test sets |
