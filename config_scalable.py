#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker
import itertools
from itertools import zip_longest as zipper
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Scalable (object):
    """Python class to compute similarity between n tweets in a finite window that can be scaled to accommodate tweets efficiently.
    Initialise/trigger the class to create instances by supplying path to the file containing your data and the
    the window size or batch size to return"""

    def __init__(self, dataframe, window_size): # defines the constructor:
         self.dataframe = dataframe
         self.window_size = window_size

    def tweets_stream(self):
        """ THis function iteratively yield tweets from a csv file containing tweets"""
        with open(self.dataframe, 'r') as csv:
            next(csv) # this skips the header in the file
            for line in csv:
                cdate, tweet = line[2:21], line[22:-1]
                yield cdate, tweet

    def tweets_batch(self,stream, window_size = 50):
        """This function utilises the tweets_stream generator function to return m number of tweets as list"""
        self.stream = stream
        self.window_size = window_size
        tweets = [] # stores batch of tweets
        try:
            for _ in range(self.window_size):
                cdate, tweet = next(stream)
                tweets.append(tweet)
        except StopIteration:
            return None, None
        return tweets

    def tweets_sim(self, tweets_list):
        """The following function computes the pairwise similarity between paair of random tweets using cosine similarity"""
        self.tweets_list = tweets_list
        similarity_scores = [] # #keeps track of similarity scores > 0.0 and < 1.0
        discarded = 0 # keeps tract of pairs with simialrity of 0.0 or 1.0
        for anchor_tweet, other_tweet in zipper(tweets_list[1:], tweets_list[2:]):
            tweet_pair = [] #  only stores the pair of tweets for simialrity computation. This is being created and deleted continously
            if other_tweet in tweets_list[2:][-1]: # avoids TypeError on reaching the final tweet in the list
                break
            anchor_tweet = tweets_list[1] # make the anchor tweet constant for each iteration in the window
            tweet_pair.append(anchor_tweet), tweet_pair.append(other_tweet)# stores pairs for simialrity computation only
            vectorizer = TfidfVectorizer(lowercase = False)
            vectorised_pair = vectorizer.fit_transform(tweet_pair) # convert tweet_pair to numeric using tfidf scheme
            cosim = np.round(cosine_similarity(vectorised_pair.toarray()[0].reshape(1,-1), vectorised_pair.toarray()[1].reshape(1,-1)).flatten(),2)
            if cosim[0] == 0.0 or cosim[0] == 1.0:
                discarded +=1
                continue
            similarity_scores.append(cosim[0])
        return similarity_scores
    def tweets_sim1(self, tweets_list):
        """The following function computes the pairwise similarity between paair of random tweets using cosine similarityself.
        This function dffers from the the above (tweet_sim) as it considers all the possible anchors in the given window. for
        instance, a window of size 11 will have 10 possible anchors, hence 10 different windows of disproportionate sizes"""
        self.tweets_list = tweets_list
        anchor_index = 0
        other_index = 1
        all_similarity_scores = []
        tracker = []
        while other_index < len(tweets_list)-1:
            #anchor = anchor
            similarity_scores = [] # #keeps track of similarity scores > 0.0 and < 1.0
            discarded = 0 # keeps tract of pairs with simialrity of 0.0 or 1.0
            for anchor_tweet, other_tweet in zipper(tweets_list[anchor_index:], tweets_list[other_index:]):
                tweet_pair = [] #  only stores the pair of tweets for simialrity computation. This is being created and deleted continously
                if other_tweet in tweets_list[1:][-1]: # avoids TypeError on reaching the final tweet in the list
                    break
                anchor_tweet = tweets_list[anchor_index] # make the anchor tweet constant for each iteration in the window
                tweet_pair.append(anchor_tweet), tweet_pair.append(other_tweet)# stores pairs for simialrity computation only
                vectorizer = TfidfVectorizer(lowercase = False)
                vectorised_pair = vectorizer.fit_transform(tweet_pair) # convert tweet_pair to numeric using tfidf scheme
                cosim = np.round(cosine_similarity(vectorised_pair.toarray()[0].reshape(1,-1), vectorised_pair.toarray()[1].reshape(1,-1)).flatten(),2)
                if cosim[0] == 0.0 or cosim[0] == 1.0:
                    discarded +=1
                    continue
                similarity_scores.append(cosim[0])
                all_similarity_scores.append(cosim[0])
            tracker.append((anchor_index, len(similarity_scores))) # keep tracks of the anchor index and the number of similar tweets to it in the window
            anchor_index +=1
            other_index +=1
        return all_similarity_scores, tracker

    def frame(self, tweets_list):
        """This function takes list of tweets and return set of windows. A frame is a unit of computation that can take very
        large file, break it into finite windows and return the associated metrics in each window. Each window consist of list of Top Scores
        and a dictionary of tracker consisting of Top scores for each anchor tweet and the corresponding number of the top scores"""
        self.tweets_list = tweets_list
        frame = {} # initialise empty frame to store all window instances and associated values
        for r in range(5):
            frame['Window_'+str(r)] = {'Top Scores':[],'Tracker':[]} # for each window store its top scores, tuple of top scores and indices
            anchor_index = 0
            other_index = 1
            while other_index < len(tweets_list)-1:
                discarded_pair = 0 # keeps track of number of pairs with simialrity of 0.0 or 1.0
                retained_pair = 0 # keeps track of number of pairs greater than 0.0 and less than 1.0
                for anchor_tweet, other_tweet in zipper(tweets_list[anchor_index:], tweets_list[other_index:]):
                    tweet_pair = [] #  only stores the pair of tweets for simialrity computation. This is being created and deleted continously
                    if other_tweet in tweets_list[1:][-1]: # avoids TypeError on reaching the final tweet in the list
                        break
                    anchor_tweet = tweets_list[anchor_index] # make the anchor tweet constant for each iteration in the window
                    tweet_pair.append(anchor_tweet), tweet_pair.append(other_tweet)# stores pairs for simialrity computation only
                    vectorizer = TfidfVectorizer(lowercase = False)
                    vectorised_pair = vectorizer.fit_transform(tweet_pair) # convert tweet_pair to numeric using tfidf scheme
                    cosim = np.round(cosine_similarity(vectorised_pair.toarray()[0].reshape(1,-1), vectorised_pair.toarray()[1].reshape(1,-1)).flatten(),2)
                    if cosim[0] == 0.0 or cosim[0] == 1.0:
                        discarded_pair +=1
                        continue
                    frame['Window_'+str(r)]['Top Scores'].append(cosim[0]) # updtae the frame data structure with the anchor tweet and
                    retained_pair +=1
                frame['Window_'+str(r)]['Tracker'].append((anchor_index, retained_pair)) # updtae the frame data structure with the anchor tweet and
                anchor_index +=1 # pick the next tweet as the next anchor
                other_index +=1 # shrink the window size by a factor of 1

        return frame

    def view_results0(self, scores):
        self.scores = scores
        """This function visualises the pairwise simialrity scores between pair of tweets in a window"""
        plt.plot(range(len(scores)),scores)
        # format the x-axis to print integers at certain intervals:
        locator = matplotlib.ticker.MultipleLocator(10)
        plt.gca().xaxis.set_major_locator(locator)
        #formatter =  matplotlib.ticker.StrMethodFormatter("{x:0f}")
        #plt.gca().xaxis.set_major_formatter(formatter)
        plt.xlabel('Index')#, rotation=90)
        plt.ylabel('Scores')
        plt.title('Similarity between anchor and other tweets')
        plt.xticks(rotation=90)
        plt.show()

    def view_results(self, frame):
        self.frame = frame
        """This function visualises the pairwise simialrity scores for a number of windows in a frame"""
        for key in frame:
            scores = [] # store top scores for every window in the frame
            for score in frame[key]['Top Scores']:
                scores.append(score)
            plt.plot(range(len(scores)),scores)
            # format the x-axis to print integers at certain intervals:
            locator = matplotlib.ticker.MultipleLocator(10)
            plt.gca().xaxis.set_major_locator(locator)
            #formatter =  matplotlib.ticker.StrMethodFormatter("{x:0f}")
            #plt.gca().xaxis.set_major_formatter(formatter)
            plt.xlabel('Index')#, rotation=90)
            plt.ylabel('Scores')
            plt.title('Similarity between anchor and other tweets in '+key)
            plt.xticks(rotation=90)
            plt.show()


if __name__=='__main__':
    """The main function to initialise/trigger the Scalable class to make its methods available"""
    trigger = Scalable('test.csv',100) # the window_size here is being overshadowed by the window_size in tweets_batch function
    stream = trigger.tweets_stream() # stream of tweets to pull out m batches
    batch = trigger.tweets_batch(stream)#, 313) # batch of tweets from stream of tweets
    sim_scores = trigger.tweets_sim(batch) # print(sim_scores)
    frame = trigger.frame(batch)
    #view_result0 = trigger.view_results(sim_scores) # visualise simialriy scores
    print(frame)
    view_result = trigger.view_results(frame) # visualise simialriy scores
