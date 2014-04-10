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
    paths = ['./train.csv', './test.csv']
    train = pd.read_csv(paths[0])
    test = pd.read_csv(paths[1])
    train_tweet = train['tweet']
    num_train = train_tweet.__len__()

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
    train_labels = np.zeros((num_train,), dtype=np.int)
    
    # Convert labels to a vector of classes
    for idx in range(0, 5):
        label_p = train[str_labels[idx]]
        for idx2 in range(0, num_train):
            if label_p[idx2] > 0.5:
                train_labels[idx2] = labels[idx]

    # Remove punctuation
    clean_tweets = sanitize_tweets(train_tweet)

    # (NOTE: Use get_vocab() this for debugging purposes.) 
    # Obtain words to see what they are.
    # vocab = get_vocab(train_set)

    # Obtain vectors which count how often a word has occurred in the list of
    # possible vocab words (formed by looking at all samples) and removes all
    # the common words, much like TfidfVectorizer
    cv = CountVectorizer(max_features = 1000,
                         stop_words = 'english',
                         lowercase = True,
                         analyzer = 'word')
    clean_tweets = cv.fit_transform(clean_tweets)
    clean_tweets = np.array(clean_tweets.toarray())

    # Split Data into training and cross-validation: 75% training, 25% cross-validation
    num_train_cv = num_train/4
    cv_threshold = num_train - num_train_cv
    train_set = clean_tweets[0:cv_threshold]
    train_set_cv = clean_tweets[cv_threshold:]
    train_set_labels = train_labels[0:cv_threshold]
    train_set_labels_cv = train_labels[cv_threshold:]

    # Multinomial Naive Bayes
    # Classification is about 57%. Perhaps use bigram or trigram for more
    # resolution?
    mnb = naive_bayes.MultinomialNB()
    mnb.fit(train_set, train_set_labels)
    print "Classification accuracy of Multinomial Naive Bayes = ",\
        mnb.score(train_set_cv, train_set_labels_cv)

if __name__=="__main__":
    main()
    
