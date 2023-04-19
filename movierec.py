import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('white', { 'axes.spines.right': False, 'axes.spines.top': False})
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords

# the IMDB movies data is available on Kaggle.com
# https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset

# in case you have placed the files outside of your working directory, you need to specify the path
path = 'data/movie_recommendations/' 

# load the movie metadata
df_meta=pd.read_csv(path + 'movies_metadata.csv', low_memory=False, encoding='UTF-8') 

# some records have invalid ids, which is why we remove them
df_meta = df_meta.drop([19730, 29503, 35587])

# convert the id to type int and set id as index
df_meta = df_meta.set_index(df_meta['id'].str.strip().replace(',','').astype(int))
pd.set_option('display.max_colwidth', 20)
df_meta.head(2)