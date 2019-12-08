import html
import re

from module.contractions import expand_contractions
from module.lemmatize import lemmatize_text
from module.tokenize import tokenize_text
from module.normalization import remove_special_characters, remove_stopwords


def keep_text_characters(text):
    filtered_tokens = []
    tokens = tokenize_text(text)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def normalize_corpus(corpus,
                     lemmatize=True,
                     tokenize=False,
                     only_text_chars=False):
    normalized_corpus = []
    for text in corpus:
        text = html.unescape(text)
        text = expand_contractions(text)
        if lemmatize:
            text = lemmatize_text(text)
        else:
            text = text.lower()
        text = remove_special_characters(text)
        text = remove_stopwords(text)

        if only_text_chars:
            text = keep_text_characters(text)

        if tokenize:
            text = tokenize_text(text)
            normalized_corpus.append(text)
        else:
            normalized_corpus.append(text)

    return normalized_corpus


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def build_feature_matrix(documents,
                         feature_type='frequency',
                         min_df=1,
                         max_df=100,
                         ngram_range=(1, 1)):
    feature_type = feature_type.lower().strip()

    if feature_type == 'binary':
        vectorizer = CountVectorizer(
            binary=True, min_df=min_df, max_df=max_df, ngram_range=ngram_range)
    elif feature_type == 'frequency':
        vectorizer = CountVectorizer(
            binary=False,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range)
    elif feature_type == 'tfidf':
        vectorizer = TfidfVectorizer(
            min_df=min_df, max_df=max_df, ngram_range=ngram_range)
    else:
        raise Exception(
            'Wrong feature type entered. Possible values: "binary", "frequency", "tfidf"'
        )

    feature_matrix = vectorizer.fit_transform(documents).astype(float)
    return vectorizer, feature_matrix


from sklearn import metrics
import numpy as np
import pandas as pd


def display_evaluation_metrics(true_labels, predicted_labels,
                               positive_class=1):
    print('Accuracy:',
          np.round(metrics.accuracy_score(true_labels, predicted_labels), 2))
    print(
        'Precision:',
        np.round(
            metrics.precision_score(
                true_labels,
                predicted_labels,
                pos_label=positive_class,
                average='binary'), 2))
    print(
        'Recall:',
        np.round(
            metrics.recall_score(
                true_labels,
                predicted_labels,
                pos_label=positive_class,
                average='binary'), 2))
    print(
        'F1 Score:',
        np.round(
            metrics.f1_score(
                true_labels,
                predicted_labels,
                pos_label=positive_class,
                average='binary'), 2))


def display_confusion_matrix(true_labels, predicted_labels, classes=[1, 0]):
    cm = metrics.confusion_matrix(
        y_true=true_labels, y_pred=predicted_labels, labels=classes)
    cm_frame = pd.DataFrame(
        data=cm,
        columns=pd.MultiIndex(
            levels=[['Predicted:'], classes], labels=[[0, 0], [0, 1]]),
        index=pd.MultiIndex(
            levels=[['Actual:'], classes], labels=[[0, 0], [0, 1]]))

    print(cm_frame)


def display_classification_report(true_labels,
                                  predicted_labels,
                                  classes=[1, 0]):
    report = metrics.classification_report(
        y_true=true_labels, y_pred=predicted_labels, labels=classes)
    print(report)
