"""
---
title: "ANN - 961742: Deep Learning Lab 1"
title-block-banner: true
author: "Htet Aung 680632035"
engine: marimo
format:
  pdf:
    echo: false
    keep-tex: false
---
"""

import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Load Modules
    """)
    return


@app.cell
def _():
    # import marimo as mo
    from icecream import ic
    import polars as pl
    import numpy as np
    import json
    import matplotlib.pyplot as plt
    from gensim.models import Word2Vec
    from nltk.tokenize import (
        word_tokenize,
        sent_tokenize,
    )
    from plotnine import (
        ggplot,
        aes,
        geom_histogram,
        theme,
        geom_point,
    )
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        brier_score_loss,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import (
        CountVectorizer,  # Bag of words
        TfidfVectorizer,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import FunctionTransformer
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    # Run the following two lines if `punkt` hasn't been downloaded yet
    # import nltk
    # nltk.download("punkt")
    return (
        CountVectorizer,
        FunctionTransformer,
        KMeans,
        LogisticRegression,
        PCA,
        Pipeline,
        TfidfVectorizer,
        Word2Vec,
        accuracy_score,
        aes,
        brier_score_loss,
        f1_score,
        geom_histogram,
        geom_point,
        ggplot,
        ic,
        json,
        np,
        pl,
        plt,
        precision_score,
        recall_score,
        roc_auc_score,
        sent_tokenize,
        theme,
        word_tokenize,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Example
    """)
    return


@app.cell
def _(ic, word_tokenize):
    text_data = [
        "Natural language processing is a field of artificial intelligence.",
        "Word embeddings capture semantic meaning of words.",
        "Machine learning and deep learning power many AI applications.",
        "Neural networks improve the performance of NLP models.",
        "Text data can be represented using embeddings.",
    ]
    ic(len(text_data))
    tokenized_text = [word_tokenize(sentence.lower()) for sentence in text_data]
    return (tokenized_text,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Train Word2Vec
    """)
    return


@app.cell
def _(PCA, Word2Vec, tokenized_text):
    word2vec_model = Word2Vec(
        sentences=tokenized_text, vector_size=100, window=5, min_count=1, workers=4
    )

    # Retrieve word vectors
    word_vectors = word2vec_model.wv
    vocab = list(word_vectors.index_to_key)

    # Reduce dimensions using PCA
    pca = PCA(n_components=2)
    word_vecs_2d = pca.fit_transform([word_vectors[word] for word in vocab])
    return vocab, word_vecs_2d


@app.cell
def _(mo):
    mo.md(r"""
    ## Visualize
    """)
    return


@app.cell
def _(plt, vocab, word_vecs_2d):
    # Visualize
    plt.figure(figsize=(10, 6))
    plt.scatter(word_vecs_2d[:, 0], word_vecs_2d[:, 1], marker="o")
    for i, word in enumerate(vocab):
        plt.annotate(word, xy=(word_vecs_2d[i, 0], word_vecs_2d[i, 1]))
    plt.title("Word Embeddings Visualization (PCA)")
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Project
    """)
    return


@app.cell
def _(ic, pl):
    df = pl.read_csv("../data/AmazonProductReviewsDataset/7817_1.csv")
    ic(df.shape);
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Drop rows with `reviews.rating==null`
    """)
    return


@app.cell
def _(df, ic):
    # Check the column description before dropping nulls
    ic(df["reviews.rating"].null_count());
    return


@app.cell
def _(df, ic):
    df_nonnull = df.drop_nulls("reviews.rating")
    # Column after dropping nulls
    ic(df_nonnull["reviews.rating"].describe());
    return (df_nonnull,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Check `reviews.rating` distribution
    """)
    return


@app.cell
def _(aes, df_nonnull, geom_histogram, ggplot, theme):
    (
        ggplot(df_nonnull, aes(x="reviews.rating"))
        + geom_histogram(bins=5)
        + theme(figure_size=(3, 2))
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Define `reviews.rating > 3` as positive, otherwise negative
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### Define Y
    """)
    return


@app.cell
def _(df_nonnull):
    y = (df_nonnull["reviews.rating"] > 3).to_numpy()
    return (y,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Check `reviews.text`
    """)
    return


@app.cell
def _(df_nonnull):
    assert 0 == df_nonnull["reviews.text"].null_count(), "There are nulls in `reviews.text`"
    assert 0 == sum(df_nonnull["reviews.text"].str.strip_chars().str.len_chars() == 0), (
        "There are empty strings."
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### Define X
    """)
    return


@app.cell
def _(df_nonnull):
    X = df_nonnull["reviews.text"].to_numpy()
    return (X,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Train-Test-Split 80-20%
    """)
    return


@app.cell
def _(X, ic, y):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    ic(X_train.shape)
    ic(X_test.shape);
    return X_test, X_train, y_test, y_train


@app.cell
def _(mo):
    mo.md(r"""
    ## Building Models

    Comparison is across feature extraction methods: `Bag of Words`, `TF-IDF` and `Word2Vec`. Simple Logistic Regression is used.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Vectorizer Parameters
    """)
    return


@app.cell
def _(ic):
    vectorizer_parameters = {
        "analyzer": "word",
        "lowercase": True,
        "stop_words": "english",
    }
    ic(vectorizer_parameters);
    return (vectorizer_parameters,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Baseline: BoW
    """)
    return


@app.cell
def _(
    CountVectorizer,
    LogisticRegression,
    Pipeline,
    X_train,
    vectorizer_parameters,
    y_train,
):
    mdl_bow = Pipeline(
        [
            ("bow", CountVectorizer(**vectorizer_parameters)),
            ("lr", LogisticRegression()),
        ]
    )
    mdl_bow.fit(X_train, y_train);
    return (mdl_bow,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Baseline: TF-IDF
    """)
    return


@app.cell
def _(
    LogisticRegression,
    Pipeline,
    TfidfVectorizer,
    X_train,
    vectorizer_parameters,
    y_train,
):
    mdl_tfidf = Pipeline(
        [
            ("tfidf", TfidfVectorizer(**vectorizer_parameters)),
            ("lr", LogisticRegression()),
        ]
    )
    mdl_tfidf.fit(X_train, y_train);
    return (mdl_tfidf,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Evaluation Function
    """)
    return


@app.cell
def _(
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
):
    def evaluate_binary(y_true, y_pred, y_proba):
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_proba)),
            "brier": float(brier_score_loss(y_true, y_proba)),
        }


    def evaluate_binary_skmodel(mdl, X, y):
        y_true = y
        y_pred = mdl.predict(X)
        y_proba = mdl.predict_proba(X)[:, 1]
        return evaluate_binary(y_true, y_pred, y_proba)

    return (evaluate_binary_skmodel,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Word2Vec

    [Word2Vec mechanism](https://code.google.com/archive/p/word2vec/)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### String Preprocessor

    Possible refinement: Some texts contain URLs. They are not explicitly handled in this version.
    """)
    return


@app.cell
def _(sent_tokenize, word_tokenize):
    def preprocess_text(X):
        # Some texts contains multiple sentences but no whitespace after "."
        # which causes word tokenizer to identify false words such as "this.but"
        # This increase w2v based logistic regression classifier ROC
        X = [word_tokenize(". ".join(sent_tokenize(text.lower().strip()))) for text in X]
        return X


    def preprocess_text_direct_word_tokens(X):
        X = [word_tokenize(text.lower().strip()) for text in X]
        return X

    return preprocess_text, preprocess_text_direct_word_tokens


@app.cell
def _(mo):
    mo.md(r"""
    ### Scikit-Learn compatible Word2VecTransformer

    Build a custom feature transformer compatible with Scikit-Learn interface so that it can be in used in `Pipeline`.
    """)
    return


@app.cell
def _(Word2Vec, np):
    from sklearn.base import BaseEstimator, TransformerMixin


    class Sent2VecTransformer(BaseEstimator, TransformerMixin):
        def __init__(self, vector_size=100, window=5, min_count=1, workers=4):
            self.vector_size = vector_size
            self.window = window
            self.min_count = min_count
            self.workers = workers
            self.model = None

        def fit(self, X, y=None):
            self.model = Word2Vec(
                sentences=X,
                vector_size=self.vector_size,
                window=self.window,
                min_count=self.min_count,
                workers=self.workers,
            )
            return self

        def transform(self, X):
            vectors = []
            for doc in X:
                word_vecs = [
                    self.model.wv[word] for word in doc if word in self.model.wv
                ]  # OOV words are ignored
                if word_vecs:  # Take average of words to from the sentence vector
                    vectors.append(np.mean(word_vecs, axis=0))
                else:  # If every word is OOV, sentence vector is all zeros
                    vectors.append(np.zeros(self.vector_size))
            return np.array(vectors)

    return (Sent2VecTransformer,)


@app.cell
def _(
    FunctionTransformer,
    LogisticRegression,
    Pipeline,
    Sent2VecTransformer,
    X_train,
    preprocess_text,
    preprocess_text_direct_word_tokens,
    y_train,
):
    mdl_s2v = Pipeline(
        [
            ("text_preprocess", FunctionTransformer(preprocess_text)),
            ("s2v", Sent2VecTransformer()),
            ("lr", LogisticRegression()),
        ]
    )
    mdl_s2v.fit(X_train, y_train)
    mdl_s2v_direct_word_tokens = Pipeline(
        [
            (
                "text_preprocess",
                FunctionTransformer(preprocess_text_direct_word_tokens),
            ),
            ("s2v", Sent2VecTransformer()),
            ("lr", LogisticRegression()),
        ]
    )
    mdl_s2v_direct_word_tokens.fit(X_train, y_train);
    return mdl_s2v, mdl_s2v_direct_word_tokens


@app.cell
def _(mo):
    mo.md(r"""
    ### Calculate performances and save to json
    """)
    return


@app.cell
def _(
    X_test,
    evaluate_binary_skmodel,
    ic,
    json,
    mdl_bow,
    mdl_s2v,
    mdl_s2v_direct_word_tokens,
    mdl_tfidf,
    y_test,
):
    performances = {
        "bow": evaluate_binary_skmodel(mdl_bow, X_test, y_test),
        "tfidf": evaluate_binary_skmodel(mdl_tfidf, X_test, y_test),
        "s2v_direct_word_tokens": evaluate_binary_skmodel(
            mdl_s2v_direct_word_tokens, X_test, y_test
        ),
        "s2v": evaluate_binary_skmodel(mdl_s2v, X_test, y_test),
    }

    with open("../outputs/performances.json", "w") as f:
        json.dump(performances, f)
    ic(performances);
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Embedding Visualization of Reviews
    """)
    return


@app.cell
def _(PCA, Word2Vec, X, plt, preprocess_text):
    def visualize_w2v_2dpca(tokenized_text):
        word2vec_model = Word2Vec(
            sentences=tokenized_text,
            vector_size=100,
            window=5,
            min_count=10,
            workers=16,
        )

        # Retrieve word vectors
        word_vectors = word2vec_model.wv
        vocab = list(word_vectors.index_to_key)

        # Reduce dimensions using PCA
        pca = PCA(n_components=2)
        word_vecs_2d = pca.fit_transform([word_vectors[word] for word in vocab])
        # Visualize
        plt.figure(figsize=(20, 20))
        plt.scatter(word_vecs_2d[:, 0], word_vecs_2d[:, 1], marker="o")
        for i, word in enumerate(vocab):
            plt.annotate(word, xy=(word_vecs_2d[i, 0], word_vecs_2d[i, 1]))
        plt.title("Word Embeddings Visualization (PCA)")
        plt.show()


    visualize_w2v_2dpca(preprocess_text(X));
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Review Clustering
    """)
    return


@app.cell
def _(
    FunctionTransformer,
    KMeans,
    PCA,
    Pipeline,
    Sent2VecTransformer,
    X_train,
    preprocess_text,
):
    cluster_s2v = Pipeline(
        [
            ("text_preprocess", FunctionTransformer(preprocess_text)),
            ("s2v", Sent2VecTransformer()),
            ("km", KMeans()),
        ]
    )
    cluster_s2v.fit(X_train)
    pca2d_s2v = Pipeline(
        [
            ("text_preprocess", FunctionTransformer(preprocess_text)),
            ("s2v", Sent2VecTransformer()),
            ("pca2d", PCA(n_components=2)),
        ]
    )
    pca2d_s2v.fit(X_train)

    pca_vector_train = pca2d_s2v.transform(X_train)
    clusters_train = cluster_s2v.predict(X_train)
    dim1, dim2, labels = pca_vector_train[:, 0], pca_vector_train[:, 1], clusters_train
    return dim1, dim2, labels


@app.cell
def _(aes, dim1, dim2, geom_point, ggplot, labels, pl):
    frame = pl.DataFrame({"dim1": dim1, "dim2": dim2, "label": labels})
    ggplot(frame, aes("dim1", "dim2", color="label")) + geom_point()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    END
    """)
    return


if __name__ == "__main__":
    app.run()
