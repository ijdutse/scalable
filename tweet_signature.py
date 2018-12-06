#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import matplotlib.ticker
import itertools
from random import randint
from itertools import zip_longest as zipper
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
sns.set()
from collections import defaultdict


class TweetSignature(object):
    """Python class to compute similarity between n tweets in a finite window that can be scaled to accommodate tweets efficiently.
    Initialise/trigger the class to create instances by supplying path to the file containing your data and the
    the window size or batch size to return"""

    def data_stream(self,datapath): # accpet the file path
        """ This function iteratively yield tweets from a csv file containing tweets"""
        self.datapath=datapath
        data =pd.read_csv(datapath)
        values = data.values # all values/columns in the DataFrame
        for ptime, tweet, period, fav, fol, friend in zip(values[:,1], values[:,2],values[:,3],values[:,4],values[:,5],values[:,6]):# extract value in columns
            yield ptime,tweet.strip(' ,'), period, fav, fol, friend

    def tweets_batch(self,stream, window_size = 50):
        """This function utilises the tweets_stream generator function to return m number of tuples of posting times and tweets as list"""
        self.stream = stream
        self.window_size = window_size
        data_tuple = [] # stores posting time, tweet, favourite, followers and friend counts of each tweet in batches
        try:
            for _ in range(self.window_size):
                ptime, tweet, period, fav, fol,friend = next(stream)
                data_tuple.append((ptime, tweet, period, fav,fol,friend)) # store tuple of items
        except StopIteration:
            return None
        return data_tuple

    def frame(self, tweets_list, frame_size = 3):
        """This function takes list of tweets and return set of windows. A frame is a unit of computation that can take very
        large file, break it into finite windows and return the associated metrics in each window. Each window consist of list of Top Scores
        and a dictionary of tracker consisting of Top scores for each anchor tweet and the corresponding number of the top scores
        The frame also contains top scores and corresponding indices in each window"""
        self.tweets_list = tweets_list # list of tweets to be broken into n windows
        self.frame_size = frame_size # number of windows in each frame
        window_size = int(len(tweets_list)/frame_size) # size of each widnow in the frame
        window_tweets = np.array_split(tweets_list, frame_size) # split tweets into window tweets according to the frame size, default is 3
        frame = {} # initialise empty frame to store all window instances and associated values
        k = 0 # initialise stopping criteria
        while k < frame_size:
            for window, window_tweet in zip(range(window_size),window_tweets):
                frame['Window_'+str(window)] = {'AnchorID':[],'AnchorTweet':[],'OtherTweet':[],'PostingTimes':[], 'RelativeTime':[],\
                'PairIndices':[],'CoSim':[],'Discards':[],'FavCount':[],'FolCount':[],'Friends':[],'Period':[],'Counter':[]}
                anchor_index = 0
                other_index = 1
                while other_index < len(window_tweet)-1:
                    discarded_pair = 0 # keeps track of number of pairs with simialrity of 0.0 or 1.0
                    retained_pair = 0 # keeps track of number of pairs greater than 0.0 and less than 1.0
                    tracked_index = 2# track all other indices
                    for anchor_tweet, other_tweet in zipper(window_tweet[anchor_index:], window_tweet[other_index:]):
                        try:
                            tweet_pair = [] #  only stores the pair of tweets for simialrity computation. This is being created and deleted continously
                            # POTENTIAL FAILURE POINT!
                            #if other_tweet in window_tweet[1:][-1]: # avoids TypeError on reaching the final tweet in the list ...
                            #    break
                            anchor_tweet = window_tweet[anchor_index] # make the anchor tweet constant for each iteration in the window
                            tweet_pair.append(anchor_tweet[1]), tweet_pair.append(other_tweet[1])# stores pairs for simialrity computation only
                            vectorizer = TfidfVectorizer(max_features= 3,lowercase = False)
                            vectorised_pair = vectorizer.fit_transform(tweet_pair) # convert tweet_pair to numeric using tfidf scheme
                            cosim = np.round(cosine_similarity(vectorised_pair.toarray()[0].reshape(1,-1), vectorised_pair.toarray()[1].reshape(1,-1)).flatten(),2)
                            if cosim[0] == 0.0 or cosim[0] == 1.0:
                                discarded_pair +=1 # keep track of less important scores
                                continue
                            frame['Window_'+str(window)]['AnchorID'].append('A_'+str(anchor_index)) # stores the id of the anchor
                            frame['Window_'+str(window)]['AnchorTweet'].append(anchor_tweet[1]) # stores the anchor tweet
                            frame['Window_'+str(window)]['OtherTweet'].append(other_tweet[1]) # stores the tweet to compare with
                            frame['Window_'+str(window)]['PostingTimes'].append((anchor_tweet[0],other_tweet[0])) # stores posting times of the pair
                            frame['Window_'+str(window)]['RelativeTime'].append((int(other_tweet[0])-int(anchor_tweet[0]))) # relative posting time/time difference
                            frame['Window_'+str(window)]['PairIndices'].append((anchor_index,tracked_index)) # stores the indices of the pair
                            frame['Window_'+str(window)]['CoSim'].append(cosim[0]) # stores the indices of the pair
                            frame['Window_'+str(window)]['Discards'].append(discarded_pair) # stores the indices of the pair
                            frame['Window_'+str(window)]['FavCount'].append(anchor_tweet[3]) # stores the favourite counts history
                            frame['Window_'+str(window)]['FolCount'].append(anchor_tweet[4]) # stores the number of followers
                            frame['Window_'+str(window)]['Friends'].append(anchor_tweet[5]) # stores the number of friends ... specific time of the day also needed
                            frame['Window_'+str(window)]['Period'].append(anchor_tweet[2]) # period of the day, e.g. morning or afternoon
                            retained_pair +=1
                            tracked_index+=1 # update the index of the inner tweet being compared with the anchor
                        except:
                            continue
                    frame['Window_'+str(window)]['Counter'].append((anchor_index, retained_pair)) # updtae the frame data structure with the anchor tweet and
                    # update indices of anchor and other tweets
                    anchor_index +=1 # pick the next tweet as the next anchor
                    other_index +=1 # shrink the window size by a factor of 1
            k+=1 # update stopping criteria:
        return frame

# MAIN ... run all:
if __name__=='__main__':
    """The main function to initialise/trigger the Scalable class to make its methods available"""
    trigger = TweetSignature()
    stream = trigger.data_stream('sbt_all_extracts_numeric_columns.csv') # stream of tweets to pull out m batches in a file with numeric columns
    batch = trigger.tweets_batch(stream, window_size=90) # batch of tweets from stream of tweets
    frame = trigger.frame(batch, frame_size=3) #  many windows and multiple anchor tweets for each window
    #test functionality ....
    #print(frame) or print(pd.DataFrame(frame))
