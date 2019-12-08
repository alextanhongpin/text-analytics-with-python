import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


def bow_extractor(corpus, ngram_range=(1, 1)):
    vectorizer = CountVectorizer(min_df=1, ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features


def tfidf_transformer(bow_matrix):
    transformer = TfidfTransformer(norm='l2', smooth_idf=True, use_idf=True)
    tfidf_matrix = transformer.fit_transform(bow_matrix)
    return transformer, tfidf_matrix


def tfidf_extractor(corpus, ngram_range=(1, 1)):
    vectorizer = TfidfVectorizer(
        min_df=1,
        norm='l2',
        smooth_idf=True,
        use_idf=True,
        ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features


# Define function to average word vectors for a text document.
def average_word_vectors(words, model, vocabulary, num_features):
    feature_vector = np.zeros((num_features, ), dtype='float64')
    nwords = 0.

    for word in words:
        if word in vocabulary:
            nwords = nwords + 1.
            feature_vector = np.add(feature_vector, model.wv[word])

    if nwords:
        feature_vector = np.divide(feature_vector, nwords)

    return feature_vector


# Generalize above function for a corpus of documents.
def averaged_word_vectorizer(corpus, model, num_features):
    vocabulary = set(model.wv.index2word)
    features = [
        average_word_vectors(tokenized_sentence, model, vocabulary,
                             num_features) for tokenized_sentence in corpus
    ]
    return np.array(features)


# Define function to compute tfidf weighted averaged word vector for a document.
def tfidf_wtd_avg_word_vectors(words, tfidf_vector, tfidf_vocabulary, model,
                               num_features):
    word_tfidfs = [
        tfidf_vector[0, tfidf_vocabulary.get(word)]
        if tfidf_vocabulary.get(word) else 0 for word in words
    ]
    word_tfidf_map = {
        word: tfidf_val
        for word, tfidf_val in zip(words, word_tfidfs)
    }
    feature_vector = np.zeros((num_features, ), dtype='float64')

    vocabulary = set(model.wv.index2word)
    wts = 0.
    for word in words:
        if word in vocabulary:
            word_vector = model.wv[word]
            weighted_word_vector = word_tfidf_map[word] * word_vector
            wts = wts + word_tfidf_map[word]
            feature_vector = np.add(feature_vector, weighted_word_vector)

    if wts:
        feature_vector = np.divide(feature_vector, wts)

    return feature_vector


# Generalize above function for corpus of documents.
def tfidf_weighted_averaged_word_vectorizer(
        corpus, tfidf_vectors, tfidf_vocabulary, model, num_features):
    docs_tfidfs = [(doc, doc_tfidf)
                   for doc, doc_tfidf in zip(corpus, tfidf_vectors)]
    features = [
        tfidf_wtd_avg_word_vectors(tokenized_sentence, tfidf, tfidf_vocabulary,
                                   model, num_features)
        for tokenized_sentence, tfidf in docs_tfidfs
    ]

    return np.array(features)
