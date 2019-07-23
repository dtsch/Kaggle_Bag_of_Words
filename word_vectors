import pandas as pd
import os
import re
from nltk.corpus import stopwords
import nltk.data
import logging
import numpy as np
from bs4 import BeautifulSoup
from gensim.models import word2vec
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)
unlabeled_train = pd.read_csv("unlableledTrainData.tsv", header=0, delimiter="\t",
                              quoting=3)
print("Read %d labeled train reviews, %d labeled test reviews, and %d unlabeled "
      "reviews.\n" % (train["review"].size, test["review"].size,
       unlabeled_train["review"].size))


def review_to_wordlist(review, remove_stopwords=False):
    # remove HTML
    review_text = BeautifulSoup(review).get_text()
    # remove non-letters
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    # convert words to lower case and split them
    words = review_text.lower().split()
    # remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if w not in stops]
    # return a list of words
    return words


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def review_to_sentences(review, tokenizer, remove_stopwords=False):
    # split into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    # loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(review_to_wordlist(raw_sentence, remove_stopwords))
    # return list of sentences
    return sentences


sentences = []
print("Parsing sentences from training set...")
for review in train["review"]:
    sentences += review_to_sentences(review, tokenizer)

print("Parsing sentences from unlabeled set...")
for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review, tokenizer)

print("There are %d sentences." % len(sentences))

##################################################################################
# Training the model

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', 
                    level=logging.INFO)
num_features = 300
min_word_count = 40
num_workers = 4
context = 10
downsampling = 1e-3

print("Training model...\n")
model = word2vec.Word2Vec(sentences, workers=num_workers, size=num_features, 
                          min_count=min_word_count, window=context, 
                          sample=downsampling)
model.init_sims(replace=True)
model_name = "300features_40minwords_10context"
model.save(model_name)

##################################################################################
# Model exploration
model.doesnt_match("man woman child kitchen".split())
model.doesnt_match("france england germany berlin".split())
model.doesnt_match("paris berlin london austria".split())

model.most_similar("man")
model.most_similar("queen")
model.most_similar("awful")

