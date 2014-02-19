# Analyzing weather sentiment on Twitter feeds
#
# Authors: Sammy Lee
#

import numpy as np
import pandas as pd
import nltk as nltk

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.tokenize import RegexpTokenizer

def tokenize_tweet(data):
    tokenizer = RegexpTokenizer('\w+')
    tokens = tokenizer.tokenize(data)
    return tokens

def get_clean_tweets(train_tweet):
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
    cv = CountVectorizer(max_features = 1000,
                         stop_words = 'english',
                         lowercase = True,
                         analyzer = 'word')
    cv._count_vocab()
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
    
def main():    
    #
    # Note: make sure you remove empty lines at the end of hte training set!
    #
    paths = ['./train_short.csv', './test.csv']
    train = pd.read_csv(paths[0])
    test = pd.read_csv(paths[1])
    train_tweet = train['tweet']
    # Obtain indices for weather sentiment
    # unknown = 's1'
    # negative = 's2'
    # neutral = 's3'
    # positive = 's4'
    # unrelated = 's5'
    s1 = train['s1']
    s2 = train['s2']
    s3 = train['s3']
    s4 = train['s4']
    s5 = train['s5']
    
    clean_tweets = get_clean_tweets(train_tweet)
    train_set = clean_tweets[0:9]
#    test_set = clean_tweets[8]
#    vocab = get_vocab(train_set)

    # Obtain vectors which count how often a word has occurred in the list of
    # possible vocab words (formed by looking at all samples)
    cv = CountVectorizer(max_features = 1000,
                         stop_words = 'english',
                         lowercase = True,
                         analyzer = 'word')
    train_count = cv.fit_transform(train_set)
    train_count_d = train_count.toarray()

    # From train count, pull only the vectors corresponding to their classes.
    # Then get assign the classes to different lists.
    train_s1_idx = np.array([idx for idx, val in enumerate(s1) if val])
    train_s2_idx = np.array([idx for idx, val in enumerate(s2) if val])
    train_s3_idx = np.array([idx for idx, val in enumerate(s3) if val])
    train_s4_idx = np.array([idx for idx, val in enumerate(s4) if val])
    train_s5_idx = np.array([idx for idx, val in enumerate(s5) if val])
    train_s1 = [train_set[idx] for idx in train_s1_idx]
    train_s2 = [train_set[idx] for idx in train_s2_idx]
    train_s3 = [train_set[idx] for idx in train_s3_idx]
    train_s4 = [train_set[idx] for idx in train_s4_idx]
    train_s5 = [train_set[idx] for idx in train_s5_idx]
    
    # Form the conditional probability of each feature/word given a certain class
    # Transform count to a 2-D matrix of conditional probabilities for each class. 
    # print vocab
    # print "indices: ", X.indices
    # print "indptr: ", X.indptr
    # print "data: ", X.data
    print train_count.toarray()
    
    # 1. Create a vector space model of all the words in the training set
    # 2. Then for each tweet, weight each word's occurence within the tweet.
    # All the weights should sum to 1.
    tfidf = TfidfVectorizer(max_features = 10000,
                            stop_words = 'english',
                            lowercase = True,
                            analyzer = 'word',
                            smooth_idf = True,
                            use_idf = True)
    X = tfidf.fit_transform(train_set)
    test_point = tfidf.transform([test_set])
    # print_test_point(vocab, test_point)

if __name__=="__main__":
    main()
    
