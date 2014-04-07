# Analyzing weather sentiment on Twitter feeds
# v3 involves using the NaiveBayesClassifier within nltk
#
# Authors: Sammy Lee
#
import math
import numpy as np
import pandas as pd
import nltk as nltk

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk import NaiveBayesClassifier
from nltk.tokenize import RegexpTokenizer
from nltk.probability import ProbDistI, DictionaryProbDist

NUM_WEATHER = 5
class LabelProbD(ProbDistI):
    def __init__(self, prob_dict, labels):
        self._prob_dict = prob_dict
        self._labels = labels

    def prob(self, sample):
        return self._prob_dist[sample]

    def max(self):
        if not hasattr(self, '_max'):
            self._max = max((p,v) for (v,p) in self._prob_dist.items())[1]
        return self._max

    def samples(self):
        return self._prob_dict.keys()

class FeatureProbD(ProbDistI):
    def __init__(self, prob_dist, feature_dict):
        self._prob_dist = prob_dist
        self._feature_dict = feature_dict.copy()

    def prob(self, sample):
        if sample in self._feature_dict.keys():
            idx = self._feature_dict[sample]
            return self.prob_dist(idx)
        return -1

    def logprob(self, sample):
        # Default definition, in terms of prob()
        p = self.prob(sample)
        if p == 0:
            # Use some approximation to infinity.  What this does
            # depends on your system's float implementation.
            return _NINF
        else:
            return math.log(p, 2)

    def max(self):
        max_idx = np.argmax(self.prob_dist)
        return [feature for feature, idx in vocab if idx==max_idx]

    def samples(self):
        return feature_dict.keys() #hack, returns the whole dictionary

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
    # Obtain indices for weather sentiment
    # NOTE: if it's unrelated, then why do i care whether it's positive, negative, or 
    # neutural? Perhaps you need keywords to determine if a tweet is unrelated or not.
    #
    # unknown = 's1'
    # negative = 's2'
    # neutral = 's3'
    # positive = 's4'
    # unrelated = 's5'
    labels = ['s1', 's2', 's3', 's4', 's5']
    s1 = train['s1']
    s2 = train['s2']
    s3 = train['s3']
    s4 = train['s4']
    s5 = train['s5']

    classes_prob = np.array([sum(s1>0), sum(s2>0), sum(s3>0), sum(s4>0), sum(s5>0)])/float(train.shape[0])
    classes_prob_dict = {}
    for idx, label in enumerate(labels):
        classes_prob_dict[label] = classes_prob[idx]
    classes_prob = DictionaryProbDist(classes_prob_dict)

    clean_tweets = sanitize_tweets(train_tweet)
    train_set = clean_tweets[0:9]
#    test_set = clean_tweets[8]
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

    # From train count, pull only the vectors corresponding to their classes.
    # Then get assign the classes to different lists.
    (train_s1_idx, train_s1, train_count_s1) =\
        get_class_tweets_and_indices(train_set, s1, train_count_d)
    (train_s2_idx, train_s2, train_count_s2) =\
        get_class_tweets_and_indices(train_set, s2, train_count_d)
    (train_s3_idx, train_s3, train_count_s3) =\
        get_class_tweets_and_indices(train_set, s3, train_count_d)
    (train_s4_idx, train_s4, train_count_s4) =\
        get_class_tweets_and_indices(train_set, s4, train_count_d)
    (train_s5_idx, train_s5, train_count_s5) =\
        get_class_tweets_and_indices(train_set, s5, train_count_d)

    # Grab the size of each class
    classes_size = np.array([len(train_s1), len(train_s2), len(train_s3),\
                              len(train_s4), len(train_s5)])

    # Sum all the word counts for each class. This counts the number of times a
    # feature has occurred in the respective classes
    # NOTE: This seems incredibly inefficient. Why create a 2D matrix that can
    # be potentially HUGE?
    train_count_class = np.zeros(shape=(NUM_WEATHER, len(vocab)))
    train_count_class[0][:] = np.sum(train_count_s1, axis=0)
    train_count_class[1][:] = np.sum(train_count_s2, axis=0)
    train_count_class[2][:] = np.sum(train_count_s3, axis=0)
    train_count_class[3][:] = np.sum(train_count_s4, axis=0)
    train_count_class[4][:] = np.sum(train_count_s5, axis=0)

    # Create a dictionary that captures the conditional probability matrix
    # P(X_i=x | Y=k), where each row represents a class (Y=k) and each column
    # represents a feature instance X_i=x.
    # The keys are (label, fname). From our count_vocab, we'll need to extract
    # the fname and its corresponding index.
    # feat_prob_dist = {}
    # test_dict = {}
    # for class_val in range(0,NUM_WEATHER):
    #     for feature, idx in vocab:
    #         feat_prob_dist[class_val, feature] =\
    #             train_count_class[class_val][idx]/float(classes_size[class_val])
    feature_dict = {}
    for feature, idx in vocab:
        feature_dict[feature] = idx

    feat_prob_dist = np.zeros(shape=(NUM_WEATHER, len(vocab)))
    for class_val in range(0, NUM_WEATHER):
        for feature, idx in vocab:
            feat_prob_dist[class_val][idx] = train_count_class[class_val][idx]/\
                float(classes_size[class_val])

    # For class 0
    classProbDist = []
    classProbDist.append(FeatureProbD(feat_prob_dist[0][:], feature_dict))
    classProbDist.append(FeatureProbD(feat_prob_dist[1][:], feature_dict))

    print classProbDist[0].prob('sun')
              
    # for class_val in range(0, NUM_WEATHER):
    #     for word in clean_tweets[7].lower().split(' '):
    #         if (class_val, word) in feat_prob_dist.keys():
    #             test_dict[class_val, word] =\
    #                 feat_prob_dist[class_val, word]
    #         else: 
    #             test_dict[class_val, word] = 0

    # Create a NaiveBayesClassifier object based off of our probability
    # distributions
    nb = NaiveBayesClassifier(classes_prob, feat_prob_dist)
    # nb.classify(test_dict)

    # Self implementation of NB

    test_prob = np.zeros((5,1))
    
    print test_prob
    # Form the conditional probability of each feature/word given a certain class
    # Transform count to a 2-D matrix of conditional probabilities for each class. 
    # print vocab
    # print "indices: ", X.indices
    # print "indptr: ", X.indptr
    # print "data: ", X.data
    
    
    # 1. Create a vector space model of all the words in the training set
    # 2. Then for each tweet, weight each word's occurence within the tweet.
    # All the weights should sum to 1.
    # tfidf = TfidfVectorizer(max_features = 10000,
    #                         stop_words = 'english',
    #                         lowercase = True,
    #                         analyzer = 'word',
    #                         smooth_idf = True,
    #                         use_idf = True)
    # X = tfidf.fit_transform(train_set)
    # test_point = tfidf.transform([test_set])
    # print_test_point(vocab, test_point)

if __name__=="__main__":
    main()
    
