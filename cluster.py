import pandas as pd
import numpy as np
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

# SETTING DATA
df_raw = pd.read_csv("data/Final Dataset Fix.csv")

# check how many data including the null values
# print(len(df_raw))
# print(df_raw["divisions"].isnull().sum())
# print(100*(df_raw["divisions"].isnull().sum()/len(df_raw)))
# delete the null values based on divisions
df_raw = df_raw.dropna(subset=['divisions'])
total_data = len(df_raw)
fac = ['FKIP', 'FMIPA', 'FEB', 'FT', 'FP', 'FISIP', 'FK', 'FH']
# print(df_raw.isnull().sum())
# df_raw.info()
# print(df_raw.head())
# sys.exit()

# MODELING / CLUSTERING DATA
def modeling_cluster(df):
  text_raw = df["text_stemmed_stopword_without_number"].to_numpy()
  # vectorize
  vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=False, ngram_range=(1,3))
  # vectorizer = TfidfVectorizer()
  data_vector = vectorizer.fit_transform(text_raw)
  # truncate vector
  svd = TruncatedSVD(n_components=2)
  X = svd.fit_transform(data_vector)
  # determine the number of cluster
  km = KMeans(random_state=100)
  visualizer = KElbowVisualizer(km, k=(1,100), timings=True)
  visualizer.fit(X)
  # clustering the data
  k = visualizer.elbow_value_
  # print("number of cluster:", k)
  kmeans = KMeans(n_clusters = k, random_state=100)
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

  return clusters, k, ss, sc, sd, df

# VISUALIZATION THE CLUSTER
def visualizing_cluster(X, clusters):
  U, S, V = randomized_svd(X, n_components=7)
  X_topics=U*S
  color = ['lightcoral', 'darkorange', 'olive', 'teal', 'violet', 'skyblue', 'red']
  embedding = umap.UMAP(n_neighbors=100, min_dist=0.5, random_state=12).fit_transform(X_topics)
  plt.figure(figsize=(7,5))
  plt.scatter(embedding[:, 0], embedding[:, 1],
              c = clusters,
              s = 10, # size
              edgecolor=color
              )
  plt.show()

# IDENTIFYING TOPIC
def identifying_topic(df, k):
  doc_clean = df["text_stemmed_stopword_without_number"]
  doc_clean_set = [d.split() for d in doc_clean]
  dictionary = corpora.Dictionary(doc_clean_set)
  doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean_set]
  words = 10
  lsamodel = LsiModel(doc_term_matrix, num_topics=k, id2word = dictionary)
  list_topic = lsamodel.print_topics(num_topics=k, num_words=words)
  
  return list_topic

# EXPORT TO CSV
csv_file = open('result/cluster_information_per_division.csv', 'w', newline="", encoding='utf-8') 
writer = csv.writer(csv_file, sys.stdout, lineterminator='\n')
writer.writerow(["faculty", "faculty_unique_relation", "total_data", "percentage", "cluster_number", "silhoutte_score", "calonski_harabasz_score", "davies_bouldin_score", "list_topic"])
# create new dataframe
dict_new = {}
for f in fac:
  df = df_raw[df_raw["divisions"].str.contains(f)]
  clusters, k, ss, sc, sd, df_cluster = modeling_cluster(df)
  list_topic = identifying_topic(df_cluster, k)
  print(f)
  dict_new["faculty"] = f
  dict_new["faculty_unique_relation"] = str(df["divisions"].unique().tolist()).replace('"', '')
  dict_new["total_data"] = int(len(df))
  dict_new["percentage"] = round(100*(len(df)/total_data), 2)
  dict_new["cluster_number"] = int(k)
  dict_new["silhoutte_score"] = float(ss)
  dict_new["calonski_harabasz_score"] = float(sc)
  dict_new["davies_bouldin_score"] = float(sd)
  dict_new["list_topic"] = list_topic
  writer.writerow(dict_new.values())