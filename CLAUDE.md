# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Build both projects
dotnet build

# Train a model
dotnet run --project TrainLtR -- data/train.small.ltr model.json

# Generate predictions
dotnet run --project PredictLtR -- model.json data/predict.ltr predictions.csv

# Build a single project
dotnet build TrainLtR/TrainLtR.csproj
dotnet build PredictLtR/PredictLtR.csproj
```

There are no automated tests in this project.

## Architecture

Two independent .NET 10 console apps sharing no library code:

**TrainLtR** — trains the model:
- `Reader.cs` parses SVM-Light `.ltr` files into jagged `Vector[][]` arrays; it sets the instance `dimFeatures` field as a side effect of parsing, so it must be called before constructing `TrainModel`
- `TrainModel.cs` defines the Infer.NET graphical model (Plackett-Luce top-1 listwise) and runs **VMP** (100 iterations) to infer the weight vector `w` (VectorGaussian) and noise (Gamma). The softmax factor in Infer.NET only supports VMP, not EP.
- `ModelSerializer.cs` flattens the posteriors to JSON (mean vector + variance matrix as flat array + Gamma shape/rate)

**PredictLtR** — generates rank distributions for new queries:
- `Reader.cs` (identical logic to TrainLtR, copy-pasted, with `targetDimFeatures` capping for dimension mismatch)
- `PredictModel.cs` — Infer.NET model that accepts trained priors; **unused by `Program.cs`**, kept as an alternative generative prediction path
- `PredictDiscriminative.cs` — the active prediction path; takes the posterior mean of `w`, computes pairwise win probabilities via the **logistic link** `σ(score_i − score_j)`, then converts to rank distributions via O(n²) DP

## Key Design Notes

- The bias term is appended to every feature vector inside `Reader.GetVectorData()`, making the weight vector one dimension larger than the raw feature count (`dimFeatures + 1`).
- Training uses the **Plackett-Luce top-1** observation per query: `winner[example] ~ Discrete(Softmax(scores[example]))` where `winner` is the index of the item with the lowest rank label (rank 1 = best in SVM-Light format). This replaces the old pairwise bool observations.
- Prediction uses the **logistic/sigmoid** link `P(i beats j) = 1/(1+exp(score_j − score_i))`, consistent with the Plackett-Luce generative model (Gumbel noise → logistic pairwise probability).
- The model serialization format is JSON; old `.bin` files in `data/` are not compatible.
- `Reader.dimFeatures` is an instance field; access it via the reader instance (`trainReader.dimFeatures`), not as a static.
- Observed data arrays in `TrainModel` are declared with `Variable.Observed(default(T[][]), outerRange, innerRange)` (from Infer.NET's ChessAnalysis pattern). This is required for pre-compilation via `engine.GetCompiledInferenceAlgorithm()` to work.
- `PredictDiscriminative` does not use the `scoresNoise` posterior at prediction time — only the posterior mean of `w` is needed for the logistic link.
- `PredictModel` exists but is not used by `PredictLtR/Program.cs` (which uses `PredictDiscriminative` instead).
