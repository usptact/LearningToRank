# Learning to Rank (LtR) with Infer.NET

A Bayesian learning-to-rank implementation using Microsoft's [Infer.NET](https://github.com/dotnet/infer) probabilistic programming framework. Implements a Plackett-Luce top-1 listwise model with a linear Gaussian score model.

> **Scalability**: Prediction is O(n²) per query where n is the number of items.

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

The model uses a **Plackett-Luce top-1 listwise** graphical model with observed feature vectors. It learns:

- **`w`** — feature weight vector (VectorGaussian posterior)
- **noise** — score precision (Gamma posterior)

For each query with `n` items, a single top-1 observation is generated: the best-ranked item is observed to win a softmax draw over all items' latent scores (`winner ~ Discrete(Softmax(s₁, …, sₙ))`). Training uses variational message passing (VMP, 100 iterations) — required because the softmax factor in Infer.NET supports VMP only. Prediction computes pairwise win probabilities via the logistic link `σ(sᵢ − sⱼ)` and converts them to rank distributions via O(n²) DP.

## Data Format

**Input** — SVM-Light format (feature IDs start at 1):
```
<rank> qid:<query_id> <feature_id>:<value> ...
```
```
1 qid:1 1:0.5 2:1.0 3:0.2
2 qid:1 1:0.3 2:0.8 3:0.9
```

- Lower rank number = better position (rank 1 = best)
- Items sharing a `qid` belong to the same query; the item with the lowest rank label is the observed winner

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
