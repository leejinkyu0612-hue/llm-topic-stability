# -*- coding: utf-8 -*-
"""Traditional baselines under the same K: LDA (bag of words), NMF (TF-IDF) and BERTopic (embeddings + UMAP + k-means + c-TF-IDF).
Each baseline is run with several random seeds so that its seed-to-seed stability can be compared with LLM run-to-run stability."""
import json, os, time, argparse, numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans
from common import load_corpus, doc_text, ROOT

RUNDIR = os.path.join(ROOT, "runs")
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def texts_of(rows):
    return [doc_text(r, max_words=120) for r in rows]


def top_words(components, vocab, n=10):
    return [" ".join(vocab[i] for i in np.argsort(-c)[:n]) for c in components]


def embeddings(corpus, texts):
    p = os.path.join(ROOT, "data", f"{corpus}_emb_minilm.npy")
    if os.path.exists(p) and np.load(p).shape[0] == len(texts):
        return np.load(p)
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(EMB_MODEL, device="cpu")
    e = m.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    np.save(p, e)
    return e


def run_lda(texts, K, seed):
    cv = CountVectorizer(stop_words="english", min_df=2, max_df=0.5, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b")
    X = cv.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=K, learning_method="batch", max_iter=100, random_state=seed, doc_topic_prior=0.1, topic_word_prior=0.01)
    th = lda.fit_transform(X)
    return th.argmax(1).tolist(), top_words(lda.components_, cv.get_feature_names_out())


def run_nmf(texts, K, seed):
    tv = TfidfVectorizer(stop_words="english", min_df=2, max_df=0.5, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b", sublinear_tf=True)
    X = tv.fit_transform(texts)
    nmf = NMF(n_components=K, init="random", random_state=seed, max_iter=500)
    W = nmf.fit_transform(X)
    return W.argmax(1).tolist(), top_words(nmf.components_, tv.get_feature_names_out())


def run_bertopic(texts, emb, K, seed):
    from bertopic import BERTopic
    from umap import UMAP
    um = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=seed)
    km = KMeans(n_clusters=K, random_state=seed, n_init=10)
    vec = CountVectorizer(stop_words="english", min_df=2, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b")
    tm = BERTopic(umap_model=um, hdbscan_model=km, vectorizer_model=vec, calculate_probabilities=False, verbose=False)
    topics, _ = tm.fit_transform(texts, emb)
    labels = [" ".join(w for w, _ in tm.get_topic(t)[:10]) for t in range(K)]
    return [int(t) for t in topics], labels


def main(corpus, seeds, n_docs=None, tag="main", methods=("LDA", "NMF", "BERTopic")):
    rows = load_corpus(corpus)
    if n_docs:
        rows = rows[::max(1, len(rows) // n_docs)][:n_docs]
    K = len(set(r["primary"] for r in rows))
    texts = texts_of(rows)
    emb = embeddings(corpus + ("" if not n_docs else f"_n{n_docs}"), texts) if "BERTopic" in methods else None
    os.makedirs(os.path.join(RUNDIR, tag), exist_ok=True)
    for meth in methods:
        for s in seeds:
            run_id = f"{corpus}__{meth}__base__r{s}"
            path = os.path.join(RUNDIR, tag, run_id + ".json")
            if os.path.exists(path):
                continue
            t0 = time.time()
            if meth == "LDA":
                lab, tw = run_lda(texts, K, s)
            elif meth == "NMF":
                lab, tw = run_nmf(texts, K, s)
            else:
                lab, tw = run_bertopic(texts, emb, K, s)
            wall = time.time() - t0
            out = dict(run_id=run_id, corpus=corpus, model=meth, cond="base", rep=s, K=K, seed=s,
                       topics=[{"id": i, "label": w, "description": ""} for i, w in enumerate(tw)], n_topics_returned=K,
                       labels=lab, doc_ids=[r["id"] for r in rows], wall_s=round(wall, 3), missing_docs=0, invalid_topic=0,
                       parse_fail_batches=0, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
            json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False)
            print(run_id, "wall", round(wall, 2), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus"); ap.add_argument("--seeds", default="1,2,3,4,5,6,7,8,9,10"); ap.add_argument("--n_docs", type=int, default=None)
    ap.add_argument("--tag", default="main")
    a = ap.parse_args()
    main(a.corpus, [int(x) for x in a.seeds.split(",")], a.n_docs, a.tag)

