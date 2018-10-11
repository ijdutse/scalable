#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



class Scale:
    """Class constructor may be activated later, for now it is muted!
    def __init__(self, dataframe): # defines the constructor:
         self.dataframe = dataframe
     """
    pass

    def stream(self,dataframe): # iteratively yield the tweets in a given file
        self.dataframe=dataframe
        with open(dataframe, 'r') as csv:
            next(csv) # this skips the header in the file
            for line in csv:
                cdate, tweet = line[2:21], line[22:-1]
                yield cdate, tweet
    def batch_sim(self,tweets_stream, batch_size=10): #1 compute similarity between pair of tweets iteratively
            self.tweets_stream = tweets_stream
            self.batch_size = batch_size
            all_other_tweets = []
            tweets = []
            similarity_scores = []
            try:
                for _ in range(batch_size):
                    cdate, tweet = next(tweets_stream)
                    tweets.append(tweet)
            except StopIteration:
                return None, None
            for i in range(2):
                try:
                    for anchor_tweet, other_tweet in itertools.zip_longest(tweets[1:], tweets[2:]): #2 zip tweets
                        tweet_pair = [] #3 return pair of tweets
                        anchor_tweet = tweets[1]
                        all_other_tweets.append(other_tweet)
                        tweet_pair.append(anchor_tweet),tweet_pair.append(other_tweet)
                        vectorizer = TfidfVectorizer()
                        vectorised_tweets = vectorizer.fit_transform(tweet_pair) #4 tfidf matrix of tweets pair
                        # compute cosine similarity between the tweet pair:
                        cosim = np.round(cosine_similarity(vectorised_tweets.toarray()[0].reshape(1,-1),\
                                                           vectorised_tweets.toarray()[1].reshape(1,-1)).flatten(),2)
                        if cosim[0] == 0.0 or cosim[0]== 1.0:
                            continue
                        similarity_scores.append(cosim[0])#yield cosim, #tweet_pair
                        del tweet_pair
                except:
                    continue
            return similarity_scores, anchor_tweet, all_other_tweets#tweet_pair
if __name__=='__main__':
    p = Scale()

    tweet_stream = p.stream('test.csv')
    sim_scores, anchor_tweet, other_tweets = p.batch_sim(tweet_stream, batch_size=51)
    print(sim_scores)
    print(anchor_tweet)
    print(other_tweets)
    print(len(sim_scores))
    print(len(other_tweets))
    plt.plot(sim_scores)
    plt.plot(range(0,len(sim_scores)),sim_scores)
    locator = matplotlib.ticker.MultipleLocator(4) # format the x-axis to print integers:
    plt.gca().xaxis.set_major_locator(locator)
    #formatter =  matplotlib.ticker.StrMethodFormatter("{x:0f}") # format sring if required
    #plt.gca().xaxis.set_major_formatter(formatter)
    plt.xlabel('Index')
    plt.ylabel('Scores')
    plt.title('Similarity between anchor and other tweets')
    plt.show()
