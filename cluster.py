#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.utils.extmath import randomized_svd
import warnings
warnings.filterwarnings("ignore")
import re
import time
import umap
import sys
import csv
from datetime import date
from gensim.models import LsiModel
from gensim import corpora
import nltk
from nltk.corpus import stopwords

try:
    nltk.data.find('tokenizers/punkt')
    nltk.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# MODELING / CLUSTERING DATA
class Clustering:
  def modeling_cluster(self, df):
    # vectorize
    text_raw = df["text_stemmed_stopword_without_number"].to_numpy()
    # print(text_raw.shape[0])
    n_dim_sample = text_raw.shape[0]
    if n_dim_sample >= 10:
      vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=False, ngram_range=(1,3))
      data_vector = vectorizer.fit_transform(text_raw)
      # truncate vector
      svd = TruncatedSVD(n_components=2)
      X = svd.fit_transform(data_vector)
      # print(X.shape)
      # determine the number of cluster
      km = KMeans(random_state=100)
      visualizer = KElbowVisualizer(km, timings=True)
      visualizer.fit(X)
      # clustering the data
      k = visualizer.elbow_value_
      # print("number of cluster:", k)
      kmeans = KMeans(n_clusters = k, random_state=100, algorithm='auto', max_iter=300, n_init=10)
      kmeans.fit(X)
      df["kluster"] = kmeans.labels_
      clusters = kmeans.labels_.tolist()

      # evaluating the cluster results
      ss = silhouette_score(X, kmeans.labels_)
      sc = calinski_harabasz_score(X, kmeans.labels_)
      sd = davies_bouldin_score(X, kmeans.labels_)
      # print(f"Silhoutte score: {ss}")
      # print(f"Calonski Harabasz Score: {sc}")
      # print(f"Davies Bouldin Score: {sd}")

      return X, clusters, k, ss, sc, sd, df
    else:
      print("sample less than 10 dimensions")
      return 0, 0, 0, 0, 0, 0, 0

  # VISUALIZATION THE CLUSTER
  def visualizing_cluster(self, x_vector, clusters, n_cluster, name):
    # def get_cmap(n, name='hsv'):
    #   return plt.cm.get_cmap(name, n)

    U, S, V = randomized_svd(x_vector, n_components = n_cluster)
    X_topics=U*S
    # cmap = get_cmap(len(n_cluster))
    embedding = umap.UMAP(n_neighbors=100, min_dist=0.5, random_state=12).fit_transform(X_topics)
    plt.figure(figsize=(7,5))
    plt.scatter(embedding[:, 0], embedding[:, 1],
                c = clusters,
                s = 10, # size
                cmap = "tab20b"
                )
    # plt.show()
    plt.savefig(f'result/cluster_{name}.png')

# IDENTIFYING TOPIC
class Topic_identifying:
  def identifying(self, df, k):
    doc_clean = df["text_stemmed_stopword_without_number"]
    doc_clean_set = [d.split() for d in doc_clean]
    dictionary = corpora.Dictionary(doc_clean_set)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean_set]
    words = 10
    lsamodel = LsiModel(doc_term_matrix, num_topics=k, id2word = dictionary)
    list_topic = lsamodel.print_topics(num_topics=k, num_words=words)
    
    return list_topic
