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
- `TrainModel.cs` defines the Infer.NET graphical model (TrueSkill/Thurstonian) and runs variational message passing (50 iterations) to infer the weight vector `w` (VectorGaussian) and noise (Gamma)
- `ModelSerializer.cs` flattens the posteriors to JSON (mean vector + variance matrix as flat array + Gamma shape/rate)

**PredictLtR** — generates rank distributions for new queries:
- `Reader.cs` (identical logic to TrainLtR, copy-pasted)
- `PredictModel.cs` — Infer.NET model that accepts trained priors and infers pairwise rank booleans; unused by `Program.cs` but kept as an alternative generative prediction path
- `PredictDiscriminative.cs` — the active prediction path; takes the posterior mean of `w` and noise, computes pairwise win probabilities via `MMath.NormalCdf`, then converts to rank distributions via O(n²) DP

## Key Design Notes

- The bias term is appended to every feature vector inside `Reader.GetVectorData()`, making the weight vector one dimension larger than the raw feature count (`dimFeatures + 1`).
- Pairwise observations compare adjacent items by index, not by rank value ordering — so training data for 2-rank queries should be shuffled, while multi-rank data should be sorted by rank before feeding to the model.
- The model serialization format changed from binary (old .NET Framework version) to JSON in the current version; old `.bin` files in `data/` are not compatible with the current loader.
- `Reader.dimFeatures` is an instance field; access it via the reader instance (`trainReader.dimFeatures`), not as a static.
- Observed data arrays in `TrainModel` are declared with `Variable.Observed(default(T[][]), outerRange, innerRange)` (from Infer.NET's ChessAnalysis pattern). This is required for pre-compilation via `engine.GetCompiledInferenceAlgorithm()` to work — declaring with `Variable.Array` leaves the variable undefined and causes a build-time error.
- `PredictDiscriminative` uses `MMath.NormalCdf` directly instead of Infer.NET inference for pairwise win probabilities; rank distributions are computed via O(n²) DP rather than O(2^n) recursion.
- `PredictModel` exists but is not used by `PredictLtR/Program.cs` (which uses `PredictDiscriminative` instead). `PredictModel` uses the full posterior over `w`; `PredictDiscriminative` uses only the posterior mean.
