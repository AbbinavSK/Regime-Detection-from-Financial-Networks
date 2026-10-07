import hashlib
import zlib

import numpy as np
import gensim.models.doc2vec as gensim_doc2vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from regime_networks import config


# Weisfeiler-Lehman feature bag of one graph: node degrees, then wl_iterations rounds of MD5-hashed labels of each node with its sorted neighbours' labels.
def _wl_graph_features(G, wl_iterations):
    labels = {node: str(G.degree(node)) for node in G.nodes()}
    all_features = list(labels.values())

    for _ in range(wl_iterations):
        labels = {node: hashlib.md5((labels[node] + "_" + "_".join(sorted(labels[n] for n in G.neighbors(node)))).encode()).hexdigest() for node in G.nodes()}
        all_features += list(labels.values())

    return all_features


# Fraction of a graph's WL features that are absent from the trained vocabulary.
def _oov_rate(features, vocab):
    return sum(1 for f in features if f not in vocab) / len(features) if features else 0.0


# Hash for seeding gensim's inference start vector that, unlike Python's per-process salted str hash, is identical in every process.
def _stable_seed_hash(seed_string):
    return zlib.crc32(seed_string.encode())


# Self-contained Graph2Vec: WL graph features fed into a gensim Doc2Vec (PV-DBOW, negative sampling, no word-vector training) model, one document per graph; hyperparameters default to config.
class Graph2Vec:
    # Store the hyperparameters; the Doc2Vec model itself is built by fit().
    def __init__(self, wl_iterations=config.G2V_WL_ITERATIONS, dimensions=config.G2V_DIMENSIONS, epochs=config.G2V_EPOCHS, learning_rate=config.G2V_LEARNING_RATE,
                 min_count=config.G2V_MIN_COUNT, down_sampling=config.G2V_DOWN_SAMPLING, workers=config.G2V_WORKERS, seed=config.SEED,
                 train_min_alpha=config.G2V_TRAIN_MIN_ALPHA, infer_min_alpha=config.G2V_INFER_MIN_ALPHA, negative=config.G2V_NEGATIVE, ns_exponent=config.G2V_NS_EXPONENT):
        self.wl_iterations = wl_iterations
        self.dimensions = dimensions
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.min_count = min_count
        self.down_sampling = down_sampling
        self.workers = workers
        self.seed = seed
        self.train_min_alpha = train_min_alpha
        self.infer_min_alpha = infer_min_alpha
        self.negative = negative
        self.ns_exponent = ns_exponent

    # Train Doc2Vec on the graphs' WL feature bags, then embed the training graphs with the same inference map later applied to test graphs (the jointly trained document vectors live on a different scale); keeps and returns those vectors and each graph's OOV rate.
    def fit(self, graphs):
        features = [_wl_graph_features(G, self.wl_iterations) for G in graphs]
        self.model = Doc2Vec([TaggedDocument(words=f, tags=[str(i)]) for i, f in enumerate(features)], vector_size=self.dimensions, window=0, min_count=self.min_count,
                             dm=0, dbow_words=0, hs=0, negative=self.negative, ns_exponent=self.ns_exponent, sample=self.down_sampling, workers=self.workers,
                             epochs=self.epochs, alpha=self.learning_rate, min_alpha=self.train_min_alpha, seed=self.seed)
        self.train_vectors, self.train_oov_rate = self.infer(graphs)
        return self.train_vectors, self.train_oov_rate

    # Infer one vector per graph deterministically (fresh seeded sampler and stable-hash start vector per graph, so a vector depends only on the model and its graph), plus each graph's OOV rate.
    def infer(self, graphs):
        features = [_wl_graph_features(G, self.wl_iterations) for G in graphs]
        original_start_vector = gensim_doc2vec.pseudorandom_weak_vector
        gensim_doc2vec.pseudorandom_weak_vector = lambda size, seed_string=None: original_start_vector(size, seed_string=seed_string, hashfxn=_stable_seed_hash)

        try:
            vectors = []

            for f in features:
                self.model.random = np.random.RandomState(self.seed)
                vectors.append(self.model.infer_vector(f, alpha=self.learning_rate, min_alpha=self.infer_min_alpha, epochs=self.epochs))
        finally:
            gensim_doc2vec.pseudorandom_weak_vector = original_start_vector

        vocab = self.model.wv.key_to_index
        return np.array(vectors), [_oov_rate(f, vocab) for f in features]
