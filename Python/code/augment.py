import pandas as pd
import numpy as np
import seaborn as sns
import re
import csv
import spacy
import nltk
import textstat
from nltk.corpus import stopwords
from collections import Counter

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS


# DATASET 1 (twitter-hate-speech.csv)
dat1 = pd.read_csv('data/twitter-hate-speech.csv', header=0, index_col=0, encoding='latin-1')
dat1 = dat1.rename(columns={'does_this_tweet_contain_hate_speech': 'label',
                            'does_this_tweet_contain_hate_speech:confidence': 'confidence'})
mapping1 = {'The tweet is not offensive': 'neither',
           'The tweet uses offensive language but not hate speech': 'offensive_language',
           'The tweet contains hate speech': 'hate_speech'}
dat1['label'] = dat1['label'].map(lambda x: mapping1[x])
dat1['original_label'] = dat1['label']
dat1 = dat1[['tweet_text', 'original_label', 'label', 'confidence']]
dat1 = dat1.reset_index(drop=True)

# DATASET 2 (twitter-hate-speech2.csv)
dat2 = pd.read_csv('data/twitter-hate-speech2.csv', index_col=0, header=0, encoding='latin-1')
conditions = [dat2['class'] == 0, dat2['class'] == 1, dat2['class'] == 2]
conf_choices = [dat2['hate_speech']/dat2['count'], dat2['offensive_language']/dat2['count'],
                dat2['neither']/dat2['count']]
lab_choices = ['hate_speech', 'offensive_language', 'neither']
dat2['confidence'] = np.select(conditions, conf_choices)
dat2['label'] = np.select(conditions, lab_choices)
dat2['original_label'] = dat2['label']
dat2 = dat2[['tweet', 'original_label', 'label', 'confidence']]
dat2 = dat2.rename(columns={'tweet': 'tweet_text'})
dat2 = dat2.reset_index(drop=True)

# Combine
frames = [dat1, dat2]
df = pd.concat(frames)

# Pre-processing functions
def preprocess(string):
    space_regex = '\s+'
    url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    emoji_regex = '&#\d*;'
    amp_regex = '&amp;'

    text = string.replace('#', '')
    text = re.sub(space_regex, ' ', text)
    text = re.sub(url_regex, 'URLHERE', text)
    text = re.sub(mention_regex, 'MENTIONHERE', text)
    text = re.sub(emoji_regex, 'EMOJIHERE', text)
    text = re.sub(amp_regex, '&', text)

    return text


def basic_tokenize(tweet):
    result = []
    for token in simple_preprocess(tweet, deacc=True, min_len=1):
        result.append(token)
    return result


stop_words_sc = stopwords.words('english')  # stopwords from spacy
stop_words_wn = [x for x in STOPWORDS]  # stopwords from wordnet
twitter_stop_words = ['urlhere', 'mentionhere', 'hastaghere', 'emojihere']
final_stopwords = list(dict.fromkeys(stop_words_sc + stop_words_wn + twitter_stop_words))  # concatenate lists and remove duplicates


def remove_stopwords(tweets):
    return [[word for word in simple_preprocess(str(doc)) if word not in final_stopwords] for doc in tweets]


# Explore readability of these tweets
df_ex = df
df_ex['preprocessed'] = df_ex['tweet_text'].map(preprocess)
df_ex['flesch_reading_idx'] = df_ex['preprocessed'].map(textstat.flesch_reading_ease)
df_ex['flesch_grade'] = df['preprocessed'].map(textstat.flesch_kincaid_grade)

df.query('label == "hate_speech"').filter(['flesch_reading_idx']).describe()
df.query('label == "offensive_language"').filter(['flesch_reading_idx']).describe()
df.query('label == "neither"').filter(['flesch_reading_idx']).describe()

df.query('label == "hate_speech"').filter(['flesch_grade']).describe()
df.query('label == "offensive_language"').filter(['flesch_grade']).describe()
df.query('label == "neither"').filter(['flesch_grade']).describe()


# DATASET 3 (NLP+CSS_2016.csv)
orig3 = pd.read_csv('data/NLP+CSS_2016.csv', sep='\t', header=0, index_col=False)

dat3_conf = pd.read_csv('data/final_NLP+CSS_2016.csv')
conditions = [dat3_conf[['sexism', 'racism', 'neither']].max(axis=1) == dat3_conf['sexism'],
              dat3_conf[['sexism', 'racism', 'neither']].max(axis=1) == dat3_conf['racism'],
              dat3_conf[['sexism', 'racism', 'neither']].max(axis=1) == dat3_conf['neither']]
lab_choices = ['sexism', 'racism', 'neither']
dat3_conf['label'] = np.select(conditions, lab_choices)
dat3_conf['count'] = dat3_conf['sexism'] + dat3_conf['racism'] + dat3_conf['neither']
conf_choices = [dat3_conf['sexism']/dat3_conf['count'], dat3_conf['racism']/dat3_conf['count'],
                dat3_conf['neither']/dat3_conf['count']]
dat3_conf['confidence'] = np.select(conditions, conf_choices)
dat3_conf = dat3_conf[['id', 'tweets', 'label', 'confidence']]
dat3_conf = dat3_conf[dat3_conf['tweets'].notnull()]
dat3_conf = dat3_conf.rename(columns={'id': 'tweet_id', 'tweets': 'tweet_text', 'label': 'original_label',
                                      'confidence': 'confidence'})

dat3_lab = pd.read_csv('data/final_labels.csv', header=0)[['tweet_text', 'speech_label']]
dat3_lab = dat3_lab.rename(columns={'tweet_text': 'tweet_text', 'speech_label': 'label'})
mapping2 = {'hate_speech': 'hate_speech',
           'offensive': 'offensive_language',
           'not_offensive': 'neither'}
dat3_lab['label'] = dat3_lab['label'].map(lambda x: mapping2[x])

dat3_final = pd.merge(dat3_conf, dat3_lab, on='tweet_text', how='inner')[['tweet_text', 'original_label', 'label', 'confidence']]

# Combine
frames = [dat1, dat2, dat3_final]
final_df = pd.concat(frames)
final_df = final_df.reset_index(drop=True)
final_df.to_csv('./final_augmented_ds.csv')

res = {x: final_df['label'].tolist().count(x) for x in final_df['label'].tolist()}
res