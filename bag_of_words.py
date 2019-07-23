import numpy as np
import pandas as pd
# import os
import re
import time
import logging
import nltk.data
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import word2vec, Word2Vec
# from KaggleWord2VecUtility import KaggleWord2VecUtility

train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
print(train.shape)
print(train.columns.values)

print(train["review"][0])

# example1 = BeautifulSoup(train["review"][0])
# print(example1.get_text)
#
# letters_only = re.sub("[^a-zA-Z]", " ", example1.get_text())
# print(letters_only)
# lower_case = letters_only.lower()
# words = lower_case.split()
#
# #nltk.download()
# print(stopwords.words("english"))
# words = [w for w in words if w not in stopwords.words("english")]
# print(words)


def review_to_words(raw_review):
    # remove HTML
    review_text = BeautifulSoup(raw_review).get_text()
    # remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    # convert to lower case, split individual words
    words = letters_only.lower().split()
    # convert stop words to a set
    stops = set(stopwords.words("english"))
    # remove stop words
    meaningful_words = [w for w in words if w not in stops]
    # join words back into one string separated by space, return result
    return " ".join(meaningful_words)


clean_review = review_to_words(train["review"][0])
print(clean_review)

num_reviews = train["review"].size
clean_train_reviews = []
print("Cleaning and parsing the training set movie reviews... \n")
for i in range(0, num_reviews):
    if (i+1)%1000 == 0:
        print("Review %d of %d\n" % (i+1, num_reviews))
    clean_train_reviews.append(review_to_words(train["review"][i]))

print("Creating the bag of words...\n")
vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None,
                             stop_words=None, max_features=5000)
train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()
print(train_data_features.shape)

vocab = vectorizer.get_feature_names()
print(vocab)

dist = np.sum(train_data_features, axis=0)
for tag, count in zip(vocab, dist):
    print(count, tag)

print("Training the random forest")
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(train_data_features, train["sentiment"])

#################################################################################

test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)
print(test.shape)
num_reviews = len(test["review"])
clean_test_reviews = []

print("Cleaning and parsing the test set movie reviews...\n")
for i in range(0, num_reviews):
    if (i+1) % 1000 == 0:
        print("Review %d of %d\n" % (i+1, num_reviews))
    clean_review = review_to_words(test["review"][i])
    clean_test_reviews.append(clean_review)

test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

result = forest.predict(test_data_features)
output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
output.to_csv("Bag_of_Words_model.csv", index=False, quoting=3)


#################################################################################
ain = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
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

#################################################################################
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

#################################################################################
# Model exploration
model.doesnt_match("man woman child kitchen".split())
model.doesnt_match("france england germany berlin".split())
model.doesnt_match("paris berlin london austria".split())

model.most_similar("man")
model.most_similar("queen")
model.most_similar("awful")

#################################################################################
model = Word2Vec.load("300features_40minwords_10context")
type(model.syn0)
print(model.syn0.shape)
print(model["flower"])


def makeFeatureVec(words, model, num_features):
    # pre-initialize empty array
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords=0
    # convert model's vocab to a set
    index2word_set = set(model.index2word)
    # loop over each word, add it's feature to total if it's in the model
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec, model[word])
    # divide result by number of words to get avg
    featureVec = np.divide(featureVec, nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # initialize counter
    counter = 0
    # preallocate array
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    # loop through reviews
    for review in reviews:
        if counter%1000 == 0:
            print("Review %d of %d." % (counter, len(reviews)))
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        counter = counter + 1
    return reviewFeatureVecs


clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append(review_to_wordlist(review, remove_stopwords=True))
trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)

print("Creating average feature vectors for test reviews")
clean_test_reviews = []
for review in test["review"]:
    clean_train_reviews.append(review_to_wordlist(review, remove_stopwords=True))
testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)

forest = RandomForestClassifier(n_estimators=100)

result = forest.predict(testDataVecs)

output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
output.to_csv("Word2Vec_AverageVectors.csv", index=False, quoting=3)

#################################################################################

start = time.time()

word_vectors = model.syn0
num_clusters = word_vectors.shape[0]/5

kmeans_clustering = KMeans(n_clusters=num_clusters)
idx = kmeans_clustering.fit_predict(word_vectors)

end = time.time()
elapsed = end - start
print("Time taken for K-Means clustering: ", elapsed, " seconds.")

word_centroid_map = dict(zip(model.index2word, idx))

for cluster in range(0,10):
    print("\nCluster %d" % cluster)
    words = []
    for i in range(0, len(word_centroid_map.values())):
        if word_centroid_map.values()[i] == cluster:
            words.append(word_centroid_map.keys()[i])
    print(words)


def create_bag_of_centroids(wordlist, word_centroid_map):
    num_centroids = max(word_centroid_map.values()) + 1
    bag_of_centroids = np.zeros(num_centroids, dtype="float32")

    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1

    return bag_of_centroids


train_centroids = np.zeros((train["review"].size, num_clusters), dtype="float32")

counter = 0
for review in clean_train_reviews:
    train_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)
    counter += 1

test_centroids = np.zeros((test["review"].size, num_clusters), dtype="float32")

counter = 0
for review in clean_test_reviews:
    test_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)
    counter += 1

forest = RandomForestClassifier(n_estimators=100)

print("Fitting a random forest to labeled training data...")
forest = forest.fit(train_centroids, train["sentiment"])
result = forest.predict(test_centroids)

output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
output.to_csv("BagOfCentroids.csv", index=False, quoting=3)
