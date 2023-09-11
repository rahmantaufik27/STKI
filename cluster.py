import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from google.colab import drive
from yellowbrick.cluster import KElbowVisualizer
# from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
# from sklearn.utils.extmath import randomized_svd
import warnings
warnings.filterwarnings("ignore")
# import re
# import time
# import umap
import sys
from gensim.models import LsiModel
from gensim import corpora
import nltk
# import tokenize
from nltk.tokenize import RegexpTokenizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.corpus import stopwords

try:
    nltk.data.find('tokenizers/punkt')
    nltk.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# SETTING DATA (FINAL DATASET FROM PREPROCESSING)
df_raw = pd.read_csv("Final Dataset.csv")
# print(df_raw["divisions"].unique())
# df_raw.info()

df_raw = df_raw.dropna(subset=['divisions'])
df_fkip = df_raw[df_raw["divisions"].str.contains('FKIP')]
print("FKIP")
print(df_fkip["divisions"].unique())

df_raw = df_raw.dropna(subset=['divisions'])
df_mipa = df_raw[df_raw["divisions"].str.contains('FMIPA')]
print("FMIPA")
print(df_mipa["divisions"].unique())

df_raw = df_raw.dropna(subset=['divisions'])
df_feb = df_raw[df_raw["divisions"].str.contains('FEB')]
print("FEB")
print(df_feb["divisions"].unique())

df_raw = df_raw.dropna(subset=['divisions'])
df_ft = df_raw[df_raw["divisions"].str.contains('FT')]
print("FT")
print(df_ft["divisions"].unique())

df_raw = df_raw.dropna(subset=['divisions'])
df_fp = df_raw[df_raw["divisions"].str.contains('FP')]
print("FP")
print(df_fp["divisions"].unique())

df_raw = df_raw.dropna(subset=['divisions'])
df_fisip = df_raw[df_raw["divisions"].str.contains('FISIP')]
print("FISIP")
print(df_fisip["divisions"].unique())

df_raw = df_raw.dropna(subset=['divisions'])
df_fk = df_raw[df_raw["divisions"].str.contains('FK')]
print("FK")
print(df_fk["divisions"].unique())

df_raw = df_raw.dropna(subset=['divisions'])
df_fh = df_raw[df_raw["divisions"].str.contains('FH')]
print("FH")
print(df_fh["divisions"].unique())

sys.exit()

def clean_text(texts):
  # text lower case
  texts = str(texts)
  text_clean = texts.lower()
  # get only alphabet text
  # text_clean = re.sub("[^0-9a-z]+", " ", text_clean)
  text_clean = re.sub("[^a-z]+", " ", text_clean)
  return text_clean

def remove_stopword(text):
  factory = StopWordRemoverFactory()
  custom = ['00',
            'aa',
            'aaa',
            'rf']
  stopwords = factory.get_stop_words()
  stopwords.extend(custom)

  result = []
  for x in  text.split():
    if x not in stopwords:
      result.append(x)

  return " ".join(result)

def remove_stopword_english(text):
  stop_words = set(stopwords.words('english'))
  result = []
  for x in  text.split():
    if x not in stop_words:
      result.append(x)

  return " ".join(result)

df_raw['text_stemmed_stopword'] = df_raw['text_stemmed_stopword'].apply(lambda x : remove_stopword(x))
df_raw['text_stemmed_stopword_eng'] = df_raw['text_stemmed_stopword'].apply(lambda x : remove_stopword_english(x))
df_raw['text_stemmed_stopword_without_number'] = df_raw['text_stemmed_stopword_eng'].apply(lambda x : clean_text(x))
df_raw = df_raw.drop(['Unnamed: 0.1', 'Unnamed: 0'], axis=1)

# MODELING / CLUSTERING DATA
# select the data
# df["abstract"] = df["abstract"].replace(np.nan, '', regex=True)
# df["text"] = df['title'] + df["abstract"]
# df["text"] = df["text"].apply(lambda x:clean_text(x))
df = df_raw.copy()
text_raw = df["text_stemmed_stopword_without_number"].to_numpy()
# text_raw
print("Text shape", text_raw.shape)

# vectorize
vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=False, ngram_range=(1,3))
# vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(text_raw)
print("Vector shape", X.shape)
terms = vectorizer.get_feature_names_out()
print(terms)

# truncate vector
svd = TruncatedSVD(n_components=2)
X2 = svd.fit_transform(X)

# determine the number of cluster
km = KMeans(random_state=100)
visualizer = KElbowVisualizer(km, k=(1,100), timings=True)
visualizer.fit(X2)
visualizer.show()

# clustering the data
k = visualizer.elbow_value_
# k = 7
kmeans = KMeans(n_clusters = k, random_state=100)
kmeans.fit(X2)
df["kluster"] = kmeans.labels_
clusters = kmeans.labels_.tolist()

ss = silhouette_score(X2, kmeans.labels_)
print(f"Silhoutte score: {ss}")
sc = calinski_harabasz_score(X2, kmeans.labels_)
print(f"Calonski Harabasz Score: {sc}")
sd = davies_bouldin_score(X2, kmeans.labels_)
print(f"Davies Bouldin Score: {sd}")

# # VISUALIZATION THE CLUSTER
# U, S, V = randomized_svd(X2, n_components=7)
# X_topics=U*S
# color = ['lightcoral', 'darkorange', 'olive', 'teal', 'violet', 'skyblue', 'red']
# embedding = umap.UMAP(n_neighbors=100, min_dist=0.5, random_state=12).fit_transform(X_topics)
# plt.figure(figsize=(7,5))
# plt.scatter(embedding[:, 0], embedding[:, 1],
#             c = clusters,
#             s = 10, # size
#             edgecolor=color
#             )
# plt.show()

# IDENTIFYING TOPIC
doc_clean = df["text_stemmed_stopword_without_number"]
doc_clean_set = [d.split() for d in doc_clean]

# doc_term_matrix = X2
# text_raw = df["text"]
# df["token"] = df["text"].apply(lambda x:tokenize_text(x))
dictionary = corpora.Dictionary(doc_clean_set)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean_set]
words = 10
lsamodel = LsiModel(doc_term_matrix, num_topics=k, id2word = dictionary)
list_topic = lsamodel.print_topics(num_topics=k, num_words=words)