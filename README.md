# Learning to Rank (LtR) with Infer.NET

A Bayesian learning-to-rank implementation of the Plackett-Luce top-1 listwise model using [Infer.NET](https://github.com/dotnet/infer).

Plackett-Luce is a listwise model: it scores all items in a query jointly rather than comparing pairs or scoring items in isolation. This makes it more label-efficient than pairwise approaches (one observation per query instead of O(n²) pairs) and avoids the imbalance problems of pointwise regression. Crucially, observing only the top-ranked item is a sufficient statistic under the Plackett-Luce distribution, so no full ranking is needed during training — a weak supervision signal that is common in real retrieval logs. The probabilistic formulation also enables a full Bayesian treatment: rather than a point estimate of the weight vector, we obtain a posterior distribution that captures uncertainty and can be updated incrementally as new data arrives.

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

### Generative model

**Priors:**

$$\mathbf{w} \sim \mathcal{N}(\mathbf{0},\, \mathbf{I})$$

$$\tau \sim \text{Gamma}(1,\, 3)$$

**For each query** $q$ with $n_q$ items and feature vectors $\mathbf{x}_{q,1}, \ldots, \mathbf{x}_{q,n_q}$:

$$s_{q,i} \mid \mathbf{w}, \tau \sim \mathcal{N}(\mathbf{w}^{\!\top}\mathbf{x}_{q,i},\ \tau^{-1}), \quad i = 1, \ldots, n_q$$

$$\text{winner}_q \mid \mathbf{s}_q \sim \text{Discrete}\left(\text{Softmax}(s_{q,1}, \ldots, s_{q,n_q})\right)$$

The observed winner for each query is the item with the lowest rank label (rank 1 = best in SVM-Light format). Each query contributes a single top-1 observation under the Plackett-Luce model.

### Inference

Posterior inference over $\mathbf{w}$ (VectorGaussian) and $\tau$ (Gamma) uses **variational message passing** (VMP, 100 iterations). VMP is required because the softmax factor in Infer.NET has no EP messages.

### Prediction

Item scores are computed as $s_i = \hat{\mathbf{w}}^{\!\top}\mathbf{x}_i$ using the posterior mean $\hat{\mathbf{w}}$. Pairwise win probabilities follow the logistic link, consistent with Plackett-Luce (Gumbel noise implies logistic pairwise):

$$P(i \succ j) = \sigma(s_i - s_j) = \frac{1}{1 + e^{s_j - s_i}}$$

An $O(n^2)$ dynamic program converts these pairwise probabilities into a full rank distribution for each item.

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

## Further Reading

- T.-Y. Liu, "Learning to Rank for Information Retrieval," *Foundations and Trends in Information Retrieval*, vol. 3, no. 3, pp. 225–331, 2009. A comprehensive survey covering pointwise, pairwise, and listwise approaches. [https://doi.org/10.1561/1500000016](https://doi.org/10.1561/1500000016)

- Z. Cao, T. Qin, T.-Y. Liu, M.-F. Tsai, and H. Li, "Learning to Rank: From Pairwise Approach to Listwise Approach," *ICML*, 2007. Foundational paper introducing the listwise learning paradigm. [https://dl.acm.org/doi/10.1145/1273496.1273513](https://dl.acm.org/doi/10.1145/1273496.1273513)
