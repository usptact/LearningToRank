# Plackett-Luce Listwise Model

This document describes the Bayesian Plackett-Luce learning-to-rank model implemented in this repository, replacing the original pairwise Thurstonian model.

## Background

The original model was a **pairwise Thurstonian** model: for each query with `n` items it generated `n-1` adjacent pairwise boolean observations (`rank[pair] = score[i+1] > score[i]`) using the Gaussian probit link via Infer.NET's `IsPositive` factor (EP inference).

The **Plackett-Luce** model provides a proper listwise generative story based on Luce's choice axiom: at each stage, the next item is chosen from the remaining set with probability proportional to its exponentiated latent score. This better captures inter-item rank correlations and is the standard Bayesian listwise LtR approach (Guiver & Snelson, ICML 2009).

## Generative Model

For each query with items `1, ..., n`:

```
w ~ VectorGaussian(0, I)               # weight vector (shared across all queries)
scoresNoise ~ Gamma(1, 3)              # score precision

for each item i:
    s_i ~ N(w · x_i, 1/scoresNoise)   # latent score

winner ~ Discrete(Softmax(s_1, ..., s_n))  # top-1 listwise observation
```

The **full Plackett-Luce likelihood** for a complete ranking permutation `π` decomposes as:

```
P(π | s) = ∏_{k=1}^{n-1}  exp(s_{π_k}) / ∑_{j ≥ k} exp(s_{π_j})
```

## Top-1 Simplification

The full decomposition requires `n-1` softmax factors of decreasing size per query (over the "remaining" items at each stage). This is incompatible with Infer.NET's static model graph, where factor structure must be fixed at compile time.

This implementation uses the **top-1 observation only** — the first stage of the Plackett-Luce chain:

```
P(item π_1 is best | s) = exp(s_{π_1}) / ∑_j exp(s_j) = Softmax(s)[winner_idx]
```

One `Discrete(Softmax(s))` factor per query. The winner index is the position of the item with the lowest rank label in the SVM-Light file (rank 1 = best position).

Trade-off: positions 2..n carry no gradient signal during training. In practice this is partially compensated by the number of queries and the shared `w` prior.

## Comparison with Thurstonian Model

| Property | Thurstonian (old) | Plackett-Luce (new) |
|---|---|---|
| Observation per query | `n-1` bool pairs | 1 winner index |
| Observation factor | `IsPositive` (probit) | `Discrete(Softmax(...))` |
| Link function | Gaussian CDF `Φ` | Softmax / logistic `σ` |
| Noise model | Gaussian score noise | Gaussian score noise + implicit Gumbel |
| Inference | EP | VMP |
| Iterations | 50 | 100 |
| Prediction link | `Φ(Δ / √(2·noise))` | `σ(Δ) = 1/(1+exp(−Δ))` |

## Why VMP

The `Softmax` factor in Infer.NET (`SoftmaxOp_Bouchard`, `SoftmaxOp_Bouchard_Sparse`) only implements `AverageLogarithm` message methods — VMP only. EP requires `AverageConditional` messages for the softmax factor, which are not available in the runtime. The engine is therefore configured as:

```csharp
engine = new InferenceEngine(new VariationalMessagePassing());
```

## Prediction

At prediction time, the posterior mean `w̄` produces item scores `μ_i = w̄ · x_i`. Pairwise win probabilities use the **logistic link**, consistent with the Plackett-Luce generative model (Gumbel noise → logistic pairwise CDF):

```
P(item i beats item j) = σ(μ_i − μ_j) = 1 / (1 + exp(μ_j − μ_i))
```

Full rank distributions are then computed via the same O(n²) DP as before (see `PredictDiscriminative.ComputeRankDistribution`). The `scoresNoise` posterior is stored in `model.json` but is not used at prediction time.

## Fallback: Variable-Size Softmax

`Variable.Softmax` is called with a `VariableArray<double>` whose range size varies per query (different item counts). If the Infer.NET compiler cannot handle this variable-size range, pad all queries to `maxItems` length with sentinel items: all-zero feature vectors (bias=1 only). The model learns to assign negligible scores to these items, and their softmax weight is effectively zero.

## References

- Guiver, J. & Snelson, E. (2009). *Bayesian inference for Plackett-Luce ranking models*. ICML.
- Luce, R.D. (1959). *Individual Choice Behavior*. Wiley.
- Minka, T. et al. *Infer.NET*. Microsoft Research.
