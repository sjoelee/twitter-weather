# Analyzing weather sentiment on Twitter feeds
# v3 involves using the NaiveBayesClassifier within nltk
#
# Authors: Sammy Lee
#
import math
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn import naive_bayes
from nltk.tokenize import RegexpTokenizer

NUM_WEATHER = 5

def tokenize_tweet(data):
    tokenizer = RegexpTokenizer('\w+')
    tokens = tokenizer.tokenize(data)
    return tokens

def sanitize_tweets(train_tweet):
    """
    Process the tweets and return only regular expression items that contain
    characters. This converts pandas's Series data to lists.
    """
    clean_tweets = []
    tokens_set = []

    for t_tweet in train_tweet:
        clean_tweet = tokenize_tweet(t_tweet)
        # append to the end of the list and not make a list within a list 
        # which is what would happen if we did clean_tweets.append(clean_tweet)
        tokens_set[len(tokens_set):] = clean_tweet
        clean_tweets.append(' '.join(clean_tweet))
    return clean_tweets

def get_vocab(clean_tweets):
    """
    Similar to TfidfVectorizer, CountVectorizer removes all the common words
    based on word occurrences in the corpus.
    Returns a sorted vocab list of words.
    """
    cv = CountVectorizer(max_features = 1000,
                         stop_words = 'english',
                         lowercase = True,
                         analyzer = 'word')
    cv._count_vocab(clean_tweets, False)
    words = cv.fit_transform(clean_tweets)
    return (sorted(cv.vocabulary_.items(), key=lambda x:x[1]))

def print_test_point(vocab, y):
    """
    For debugging purpose, use this.
    Pick a test item and check to see what is the tf-idf vector being passed 
    back. This will output a list of vocab words and weights.
    """
    # convert to list for further processing
    ydenselist = list(np.array(y.todense()).reshape(-1,))
    print type(ydenselist)
    print ydenselist
    for idx, val in enumerate(ydenselist):
        if val > 0:
            print (vocab[idx][0], val)
    
def get_class_tweets_and_indices(train_set, class_vector, train_count_d):
    """
    Return a list of indices of the training set that correspond to a class.
    """
    train_idx = np.array([idx for idx, val in enumerate(class_vector) if val])
    train = [train_set[idx] for idx in train_idx]
    train_count = np.array([train_count_d[idx] for idx in train_idx])

    return train, train_idx, train_count

def main():    
    #
    # Note: make sure you remove empty lines at the end of the training set!
    #
    paths = ['./train_short.csv', './test.csv']
    train = pd.read_csv(paths[0])
    test = pd.read_csv(paths[1])
    train_tweet = train['tweet']
    num_train = train_tweet.__len__()
    # Obtain indices for weather sentiment
    # NOTE: if it's unrelated, then why do i care whether it's positive, negative, or 
    # neutural? Perhaps you need keywords to determine if a tweet is unrelated or not.
    #
    # unknown = 's1'
    # negative = 's2'
    # neutral = 's3'
    # positive = 's4'
    # unrelated = 's5'
    labels = [1, 2, 3, 4, 5]
    str_labels = ['s1', 's2', 's3', 's4', 's5']
    train_labels = [0]*num_train
    
    # Convert labels to a vector of classes
    for idx in range(0, 5):
        label_p = train[str_labels[idx]]
        for idx2 in range(0, num_train):
            if label_p[idx2] > 0.5:
                # train_labels[idx2] = labels[idx]
                train_labels[idx2] = labels[idx]
    print train_labels
    print train_labels[0:7]

    lbin = LabelBinarizer()
    train_labels_ml = lbin.fit_transform(train_labels)
    print train_labels_ml


    clean_tweets = sanitize_tweets(train_tweet)
    train_set = clean_tweets[0:8]
    print train_set.__len__()
    vocab = get_vocab(train_set)

    #
    # NAIVE BAYES IMPLEMENTATION
    #
    # Obtain vectors which count how often a word has occurred in the list of
    # possible vocab words (formed by looking at all samples)
    cv = CountVectorizer(max_features = 1000,
                         stop_words = 'english',
                         lowercase = True,
                         analyzer = 'word')
    train_count = cv.fit_transform(train_set)
    train_count_d = np.array(train_count.toarray())

    trainSet = train_count_d[0:7,:]
    testSet = train_count_d[7,:]
    print train_count_d[7,:]
    print train_count_d.shape
    print train_labels.__len__()
    print train_count_d[7]

    mnb = naive_bayes.MultinomialNB()
    mnb.fit(train_count_d, train_labels_ml[0:8])
    mnb.score(testSet, train_labels_ml[8])


if __name__=="__main__":
    main()
    
