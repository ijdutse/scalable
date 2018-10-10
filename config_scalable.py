#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from datetime import datetime
import preprocessor as p
import random, os, utils, smart_open, json, codecs, pickle, time
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument, LabeledSentence
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.fftpack import fft


# Example of list of Raw Data: data_sources = ['diverse_tweets1.json']

#def main():
 #   spd = Spd(data_sources) #class instantiation
  #  start = time.clock()
   # relevant_tweets = spd.detector(data_sources)
    #stop = time.clock()
    #return relevant_tweets




class Semantics:
    """Class constructor may be activated later, for now it is muted!
    def __init__(self, dataframe): # defines the constructor:
         self.dataframe = dataframe
     """
    pass

    def get_tweets_stream(self, dataframe): # iteratively yield the tweets in a given file
        with open(dataframe, 'r') as csv:
            next(csv) # this skips the header in the file
            for line in csv:
                cdate, tweet = line[2:21], line[22:-1]
                yield cdate, tweet
    def get_tweet_batch(self, tweets_stream, batch_size=50): #return tweets in batches of n tweets, default is 50
        self.tweets_stream = tweets_stream
        self.batch_size = batch_size
        tweets = []
        try:
            for _ in range(batch_size):
                _, tweet = next(tweets_stream)
                tweets.append(tweet)
        except StopIteration:
            return None, None
        return tweets

    def get_tfidf_matrix(self, tweets): # convert tweets from get_tweet_batch() to numeric using tfidf scheme ...
        self.tweets = tweets  # or call the member function returning list of tweets
        vectorizer = TfidfVectorizer()
        vectorised_tweets = vectorizer.fit_transform(tweets)
        return vectorised_tweets.toarray()

    # pairwise similarity using the usual approach and approach proposed for the study
    def pairwise_sim1(self, vectorised_tweets): # compute pairwise similarity using usual cosine similarity
        self.vectorised_tweets = vectorised_tweets
        tfidf_matrix = vectorised_tweets
        sim_scores = []
        for row in tfidf_matrix:
            cosim = np.round(cosine_similarity(row.reshape(1,-1), tfidf_matrix).flatten(),2)
            sim_scores.append(cosim)
        return sim_scores

    def pairwise_sim2(self, vectorised_tweets): # compute pairwise similarity using modified row-wise
        self.vectorised_tweets = vectorised_tweets
        tfidf_matrix = vectorised_tweets
        sim_scores = []
        y_axis = []
        j = tfidf_matrix.shape[0]
        while j > 1:
            i = 0
            c = 1
            for index in range(i, j):
                if index == 0:
                    continue # avoid multiplying a row by itself
                cosim = np.round(cosine_similarity(tfidf_matrix[i:c].reshape(1,-1),\
                                tfidf_matrix[index:]).flatten() ,2)
                #y_axis.append(cosim)
                sim_scores.append(cosim)
                # update indices ...
                j -=1
                i +=1
                c +=1
                # plot the results, if necessary
        for score in sim_scores: # return scores as single array/list
            y_axis.extend(score)
        return y_axis
    # get most similar or top scores ... a relevent function:
    def top_scores2(self, sim_scores): # using the argpartition function of numpy for the operation
        self.sim_scores = sim_scores
        sim_score_index = []
        for array in sim_scores:
            index = np.argpartition(array, -array.argmax())[-9:] # partition on highest score to return last 9
            sim_score = array[index]
            sim_score_index.append((sim_score, index))
            # visualisation using the score_index to plot the values:
        return sim_score_index
    def top_scores1(self, sim_scores): # more accurate procedure  based on elimination trick ...
        self.sim_scores = sim_scores
        sim_scores_and_indices = []
        for array in sim_scores:
            #scores_and_indices = []
            while array.shape[0] > 1:
                sim_scores_and_indices.append((array.max(), array.argmax()))
                array = np.delete(array, array.argmax())
        # visualisation using the scores_and_indices ...
        return sim_scores_and_indices
    def ff_transform(self, sim_scores): #Transformation of sim_scores in time domain to frequency domain ..
        self.sim_scores = sim_scores
        time_scores = np.array(sim_scores)
        frequency_scores = fft(time_scores)
        # plot the result:
        return frequency_scores
#if __name__ =='__main__':
 #   main()
