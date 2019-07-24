import numpy as np
import pandas as pd
# import os
import re
import time
import logging
import nltk.data
nltk.download()
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import word2vec, Word2Vec
# from KaggleWord2VecUtility import KaggleWord2VecUtility

#################################################################################
# training the bag of words model                                               #
#################################################################################
# read in training data, observe data properties
train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
print("Training data has %d columns and %d rows/n" % (train.shape[0], train.shape[1]))
print("Training column values are:/n", train.columns.values)
# printing out example review
print("Raw review example:\n", train["review"][0])


def review_to_words(raw_review):
    # function to clean and parse reviews into individual words
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


# running parsing function on an example review
print("Clean review example:\n")
clean_review = review_to_words(train["review"][0])
print(clean_review)

# obtaining number of reviews
num_reviews = train["review"].size
# initializing empty list
clean_train_reviews = []
# cleaning and parsing all reviews
print("Cleaning and parsing the training set movie reviews...\n")
for i in range(0, num_reviews):
    if (i+1)%1000 == 0: # printing progress to the console every 1000th review
        print("Review %d of %d\n" % (i+1, num_reviews))
    clean_train_reviews.append(review_to_words(train["review"][i]))
print("Cleaning and parsing complete.\n")

# initializing bag of words tool
print("Creating the bag of words...\n")
vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None,
                             stop_words=None, max_features=5000)
# fitting model
train_data_features = vectorizer.fit_transform(clean_train_reviews)
# converting to array
train_data_features = train_data_features.toarray()
print("Training data has %d rows and %d features (vocab words)." % 
      (train_data_features.shape[0], train_data_features.shape[1]))

# observing words in the vocab
vocab = vectorizer.get_feature_names()
print("Vocab words:/n", vocab)

# word counts
dist = np.sum(train_data_features, axis=0)
print("Word counts:/n")
for tag, count in zip(vocab, dist):
    print(count, tag)

print("Training random forest classifier.../n")
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(train_data_features, train["sentiment"])

#################################################################################
# creating Kaggle submission formatted output with the test data                #
#################################################################################
# reading in test data
test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)
print("There are %d rows and %d columns in the test data." % 
      (test.shape[0], test.shape[1]))
num_reviews = len(test["review"])
clean_test_reviews = []

print("Cleaning and parsing the test set movie reviews...\n")
for i in range(0, num_reviews):
    if (i+1) % 1000 == 0:
        print("Review %d of %d\n" % (i+1, num_reviews))
    clean_review = review_to_words(test["review"][i])
    clean_test_reviews.append(clean_review)
print("Cleaning and parsing complete.\n")

# obtaining bag of words and converting to array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()
# creating predictions from the model and outputing to csv
result = forest.predict(test_data_features)
output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
output.to_csv("Bag_of_Words_model.csv", index=False, quoting=3)

#################################################################################
# distributed word vectors                                                      #
#################################################################################
# loading unlabeled training data
unlabeled_train = pd.read_csv("unlableledTrainData.tsv", header=0, 
                              delimiter="\t", quoting=3)
print("Read %d labeled train reviews, %d labeled test reviews, and %d unlabeled "
      "reviews.\n" % (train["review"].size, test["review"].size,
                      unlabeled_train["review"].size))


def review_to_wordlist(review, remove_stopwords=False):
    # new function to clean and parse reviews, with optional stopword removal
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


# loading tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def review_to_sentences(review, tokenizer, remove_stopwords=False):
    # new function to clean and parse reviews into sentences and tokens
    # split into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    # loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(review_to_wordlist(raw_sentence, remove_stopwords))
    # return list of sentences
    return sentences

# initializing empty list
sentences = []
print("Parsing sentences from training set...\n")
for review in train["review"]:
    sentences += review_to_sentences(review, tokenizer)
print("Finished parsing training set sentences.\n")

print("Parsing sentences from unlabeled set...\n")
for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review, tokenizer)
print("Finished parsing unlabeled set sentences.\n")
print("There are %d sentences." % len(sentences))

#################################################################################
# training the word vector model                                                #
#################################################################################
# logging module for clean output messages
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)
# set parameter values
num_features = 300
min_word_count = 40
num_workers = 4
context = 10
downsampling = 1e-3

print("Training word vector model...\n")
model = word2vec.Word2Vec(sentences, workers=num_workers, size=num_features,
                          min_count=min_word_count, window=context,
                          sample=downsampling)
# increase memory efficiency, if not training model more
model.init_sims(replace=True)
model_name = "300features_40minwords_10context"
model.save(model_name)
print("Finished training word vector model.\n")

#################################################################################
# word vector model results exploration                                         #
#################################################################################
print("The mismatched word among 'man, woman, child, kitchen' is ',
      model.doesnt_match("man woman child kitchen".split()))
print("The mismatched word among 'france england germany berlin' is ',
      model.doesnt_match("france england germany berlin".split())
print("The mismatched word among 'paris berlin london austria' is ',
      model.doesnt_match("paris berlin london austria".split())

print("The words most similar to 'man' are:\n",
      model.most_similar("man"))
print("The words most similar to 'queen' are:\n",
      model.most_similar("queen"))
print("The words most similar to 'awful' are:\n",
      model.most_similar("awful"))

#################################################################################
# vector averaging w/ random forest classifier                                  #
#################################################################################
# loading the model
model = Word2Vec.load("300features_40minwords_10context")
# calling the array of word feature vectors
type(model.syn0)
print("The model has %d words and %d features.\n" % 
      (model.syn0.shape[0], model.syn0.shape[1]))
print("Individual word vector example:\n", model["flower"])

      
def makeFeatureVec(words, model, num_features):
    # function to average all the word vectors in a given paragraph
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
      # function to average the feature vectors
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


# calculating average feature vectors for training and test sets
print("Creating average feature vectors for training reviews\n")
clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append(review_to_wordlist(review, remove_stopwords=True))
trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)

print("Creating average feature vectors for test reviews\n")
clean_test_reviews = []
for review in test["review"]:
    clean_train_reviews.append(review_to_wordlist(review, remove_stopwords=True))
testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)

# training the classifier withthe average paragraph vectors
print("Training random forest classifier on labeled training data...\n")
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(trainDataVecs, train["sentiment"])
print("Finished training random forest classifier.\n")

# testing and extracting results
print("Creating predictions and writing results to csv...\n")
result = forest.predict(testDataVecs)
# writing results to csv
output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
output.to_csv("Word2Vec_AverageVectors.csv", index=False, quoting=3)
print("Finished predicting and writing.\n")

#################################################################################
# k-means clusering/vector quantization                                         #
#################################################################################
# for clocking runtime
start = time.time()

# setting k to 1/5th vocab size
word_vectors = model.syn0
num_clusters = word_vectors.shape[0]/5

# initialize k-means and extract centroids
kmeans_clustering = KMeans(n_clusters=num_clusters)
idx = kmeans_clustering.fit_predict(word_vectors)

# calculate and print runtime
end = time.time()
elapsed = end - start
print("Time taken for K-Means clustering: ", elapsed, " seconds.")

# creating index dictionary, mapping vocab words to cluster numbers
word_centroid_map = dict(zip(model.index2word, idx))

# observing the first 10 clusters
for cluster in range(0,10):
    print("\nCluster %d" % cluster) # cluster number
    words = [] # obtaining all words for each cluster
    for i in range(0, len(word_centroid_map.values())):
        if word_centroid_map.values()[i] == cluster:
            words.append(word_centroid_map.keys()[i])
    print(words)


# function to convert reviews to bags of centroids
def create_bag_of_centroids(wordlist, word_centroid_map):
    num_centroids = max(word_centroid_map.values()) + 1
    bag_of_centroids = np.zeros(num_centroids, dtype="float32")
    
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    
    return bag_of_centroids

# creating bags of centroids for training and test sets
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

# fitting random forest classifier with training set and running predictions on test
forest = RandomForestClassifier(n_estimators=100)

print("Fitting a random forest to labeled training data...\n")
forest = forest.fit(train_centroids, train["sentiment"])
print("Finished fitting.\n")
      
print("Creating predictions and writing results to csv...\n") 
result = forest.predict(test_centroids)
output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
output.to_csv("BagOfCentroids.csv", index=False, quoting=3)
print("Finished predicting and writing.\n")
