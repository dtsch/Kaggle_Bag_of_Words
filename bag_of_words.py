import numpy as np
import pandas as pd
#import os
import re
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

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


def review_to_words( raw_review ):
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
