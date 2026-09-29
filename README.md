# Vendi Clustering

Reference implementation of Vendi Clustering, a topic reduction method that merges
topics so as to preserve the diversity of the topic set, measured by the Vendi Score.

Given a fitted topic model with more topics than you want, the usual choice is
agglomerative merging on topic similarity. Vendi Clustering instead picks, at each
step, the merge that best preserves the Vendi Score of order *q* over the remaining
topics.

The core depends only on numpy, scipy and scikit-learn. BERTopic is optional and is
reached solely through its public API.

## Install

```bash
pip install vendi-clustering              # core only
pip install vendi-clustering[bertopic]    # with the BERTopic integration
pip install vendi-clustering[metrics]     # with the evaluation metrics
```

## Usage

### With BERTopic

```python
from bertopic import BERTopic
from vendi_clustering.bertopic_adapter import cluster_topics

topic_model = BERTopic().fit(docs)
cluster_topics(topic_model, docs, nr_topics=25)   # merges in place
```

`use_ctfidf=True` clusters on c-TF-IDF vectors rather than semantic embeddings.
`q=` selects the Vendi Score order (default 2.0). The outlier topic (-1) is held
out and left untouched.

### Standalone

The core takes topic embeddings and sizes and returns a mapping from each input
topic to the topic that absorbed it. It never needs to know where they came from.

```python
from vendi_clustering import VendiClustering

mapping = VendiClustering().cluster(
    embeddings=topic_embeddings,   # (n_topics, dim), row i belongs to topic_ids[i]
    topic_ids=topic_ids,
    topic_sizes=topic_sizes,       # {topic_id: document count}
    target_k=25,
)
```

`VendiClustering` implements q=2 using a Frobenius-norm shortcut that scores every
candidate merge in closed form. `GeneralVendiClustering(q=...)` supports any q > 0,
including q=1 and q=inf, via eigendecomposition — considerably slower, and used for
the q-sensitivity analysis in our paper.

Each merged group's surviving topic is the lowest ID in that group.

## Evaluation

`vendi_clustering.metrics` scores any topic model, not just BERTopic, by going
through a plain `TopicModelOutput` (topic words, document assignments, optional
topic vectors). This is what lets the same metric code compare against native topic
models.

```python
from sentence_transformers import SentenceTransformer

from vendi_clustering.metrics.adapters import bertopic_analyzer, from_bertopic
from vendi_clustering.metrics.scores import evaluate

coh = SentenceTransformer("paraphrase-MiniLM-L6-v2")

output = from_bertopic(topic_model, method="vendi")
scores = evaluate(
    output,
    docs,
    analyzer=bertopic_analyzer(topic_model),
    embed_words=coh.encode,
)
```

The analyzer is passed explicitly so every method being compared is scored against
one tokenization of the reference corpus. `embed_words` is optional; embedding
coherence (`coh`) is computed only when it is given.

## Reproducing the paper

The experiments live in `experiments/` and are not part of the installed package.
Protocols are driven by YAML configs:

```bash
pip install -r requirements-repro.txt
PYTHONPATH=src python experiments/vendi_experiments.py --config experiments/configs/p1_20NG.yaml
```

`requirements-repro.txt` pins the dependencies but not this package, so `src` goes on
`PYTHONPATH` rather than installing over the pinned environment.

### Baseline comparison

`experiments/baselines/` compares Vendi-reduced BERTopic against LDA, CombinedTM,
ETM and FASTopic fitted natively at the same *k*. Those models come from TopMost,
which brings its own torch stack, so they run in a **separate environment** and
communicate only through a plain `TopicModelOutput` JSON:

```
topmost env (LDA, CombinedTM, ETM, FASTopic) ──> *.json ──┐
                                                          ├──> dev env: evaluate()
BERTopic + Vendi Clustering ──────────────────────────────┘
```

Scoring happens once, in the dev environment, so every method shares one analyzer,
one reference corpus and one COH encoder.

```bash
# baselines environment -- imports nothing from this repo
pip install -r requirements-baselines.txt
python experiments/baselines/run_baseline.py --method fastopic --k 50 --seed 42

# dev environment
PYTHONPATH=src:. python experiments/baselines/run_vendi.py --k 50 --seed 42
PYTHONPATH=src:. python experiments/baselines/score_all.py   # a whole (k, seed) grid
```

`run_baseline.py` departs from TopMost in two places, both explained in the source:
FASTopic is driven through the `fastopic` package directly rather than TopMost's
trainer, and CombinedTM's bag-of-words concatenation — commented out upstream — is
re-enabled.

## Citation

```bibtex
@article{kimvendi,
  title={Vendi Clustering for Diversity-Preserving Topic Reduction},
  author={Kim, Sean and Dieng, Adji Bousso}
}
```

## License

MIT. See [LICENSE](LICENSE).
